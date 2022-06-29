import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from utills import over_flow_battery

class Qnet(torch.nn.Module):
    def __init__(self, feature):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(feature, 512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 21)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DualingQnet(torch.nn.Module):
    def __init__(self, feature):
        super(DualingQnet, self).__init__()
        self.fc1 = torch.nn.Linear(feature, 512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.relu = torch.nn.ReLU()
        
        self.output_value = torch.nn.Linear(512, 1) # V(s)
        self.output_layer = torch.nn.Linear(512, 21) # A(s, a)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        a = self.output_layer(x)
        v = self.output_value(x).expand_as(a)

        q = v + a - a.mean(0, keepdim=True).expand_as(a)
        return q


class DQN():
    def __init__(self, 
                 replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 training_strategy_fn,
                 cal_price,
                 next_state,
                 batch_size,
                 gamma,
                 battery_max,
                 ):
        """
        replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                cal_price,
                next_state,
                batch_size,
                gamma,
                battery_max
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.cal_price = cal_price
        self.next_state = next_state

    def optimize_model(self, Tf):
        """
        online 신경망 학습
        """
        s, a, r, s_prime = self.replay_buffer.sample(self.batch_size)
        q_a = self.online_model(s).gather(1,a)
    
        # target 신경망
        min_q_prime = self.target_model(s_prime).detach()
        Q_target = over_flow_battery(self.batch_size, self.battery_max, s_prime, min_q_prime, Tf).min(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        
        # r + discount * max(q)
        target = r + self.gamma * Q_target
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max)
        return action
    
    def update_network(self):
        """
        target 신경망 업데이트
        """
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, 
            feature, 
            epochs,    
            start_day, 
            end_day, 
            summer_TOU, 
            Tf,
            pD,
            load_data, 
            generation_data, 
            days, 
            T, 
            start_size, 
            freq,
            init_state,
            
            ):

        # online target 신경망 할당
        self.online_model = self.value_model_fn(feature)
        self.target_model = self.value_model_fn(feature)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        
        cost_history, battery_history, action_history = [],[],[]
        
        state = init_state

        # init 배터리, 행동
        battery, action = 0, 0

        with tqdm(range(epochs), unit="Run") as runing_bar:
            for n_epi in runing_bar:
                cum_cost, day_action, day_battery = 0, [] , []
                #if (start_day + n_epi % end_day) % 365 < 90 or (start_day + n_epi % end_day) % 365 >= 273:TOU = winter_TOU
                #else: TOU = summer_TOU
                TOU = summer_TOU
                if n_epi % (end_day-start_day) == 0: state[0] = 0
                for time in range(24):
                    battery = state[0] - state[50] + state[51+Tf]
                    if battery > self.battery_max: battery = self.battery_max  # generation 초과 제한
                    
                    # online 신경망 행동
                    action = self.interaction_step(torch.from_numpy(state).float(), battery)
                    
                    # 하루 행동 및 배터리 추가
                    day_action.append(action)
                    day_battery.append(state[0])
                    
                    # cost 계산
                    cost = self.cal_price(time=time,charge=action,day_charge=day_action,
                                    TOU=TOU, pD=pD)
                    cum_cost = cum_cost + cost
                    
                    # 다음 state 계산
                    state_prime = self.next_state(n_epi=n_epi,time=time,
                                                battery=battery,charge=action,day_charge=day_action,
                                                TOU=TOU,load_data=load_data,generation_data=generation_data,  
                                                feature=feature, Tf=Tf, days=days, T=T)
                    
                    # buffer에 추가
                    self.replay_buffer.put((state, action, cost, state_prime))
                    
                    # 갱신
                    state = state_prime

                cost_history.append(cum_cost)
                action_history.append(day_action)
                battery_history.append(day_battery)
                
                if n_epi > days and n_epi % 1000 == 0:
                    runing_bar.set_postfix(cost=sum(cost_history[n_epi-days:n_epi]))
                
                # 버퍼가 start_size 이상이면 online 모델 업데이트
                if len(self.replay_buffer) > start_size:
                    self.optimize_model(Tf)

                # freq 마다 target 신경망 업데이트 
                if n_epi % freq == 0 and n_epi != 0:
                    self.update_network()

        return cost_history, action_history, battery_history
    

# DDQN
class DDQN():
    def __init__(self, 
                 replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 training_strategy_fn,
                 cal_price,
                 next_state,
                 batch_size,
                 gamma,
                 battery_max,
                 ):
        """
        replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                cal_price,
                next_state,
                batch_size,
                gamma,
                battery_max
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.cal_price = cal_price
        self.next_state = next_state

    def optimize_model(self, Tf):
        """
        online 신경망 학습
        """
        s, a, r, s_prime = self.replay_buffer.sample(self.batch_size)

        q_a = self.online_model(s_prime)
        # min index 뽑기
        min_q_a = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_a, Tf).min(1)[1]
        
        q_sp = self.target_model(s_prime).detach()
        min_q_prime = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_sp, Tf)
        Q_target = min_q_prime[np.arange(self.batch_size), min_q_a]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        # r + discount * max(q)
        target = r + self.gamma * Q_target

        q_a = self.online_model(s).gather(1,a)

        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max)
        return action
    
    def update_network(self):
        """
        target 신경망 업데이트
        """
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)

    def train(self, 
            feature, 
            epochs,    
            start_day, 
            end_day, 
            summer_TOU, 
            Tf,
            pD,
            load_data, 
            generation_data, 
            days, 
            T, 
            start_size, 
            freq,
            init_state,
            
            ):

        # online target 신경망 할당
        self.online_model = self.value_model_fn(feature)
        self.target_model = self.value_model_fn(feature)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        
        cost_history, battery_history, action_history = [],[],[]
        
        state = init_state

        # init 배터리, 행동
        battery, action = 0, 0
        
        with tqdm(range(epochs), unit="Run") as runing_bar:
            for n_epi in runing_bar:
                cum_cost, day_action, day_battery = 0, [] , []
                #if (start_day + n_epi % end_day) % 365 < 90 or (start_day + n_epi % end_day) % 365 >= 273:TOU = winter_TOU
                #else: TOU = summer_TOU
                TOU = summer_TOU
                if n_epi % (end_day-start_day) == 0: state[0] = 0
                for time in range(24):
                    battery = state[0] - state[50] + state[51+Tf]
                    if battery > self.battery_max: battery = self.battery_max  # generation 초과 제한
                    
                    # online 신경망 행동
                    action = self.interaction_step(torch.from_numpy(state).float(), battery)
                    
                    # 하루 행동 및 배터리 추가
                    day_action.append(action)
                    day_battery.append(state[0])
                    
                    # cost 계산
                    cost = self.cal_price(time=time,charge=action,day_charge=day_action,
                                    TOU=TOU, pD=pD)
                    cum_cost = cum_cost + cost
                    
                    # 다음 state 계산
                    state_prime = self.next_state(n_epi=n_epi,time=time,
                                                battery=battery,charge=action,day_charge=day_action,
                                                TOU=TOU,load_data=load_data,generation_data=generation_data,  
                                                feature=feature, Tf=Tf, days=days, T=T)
                    
                    # buffer에 추가
                    self.replay_buffer.put((state, action, cost, state_prime))
                    
                    # 갱신
                    state = state_prime

                cost_history.append(cum_cost)
                action_history.append(day_action)
                battery_history.append(day_battery)
                
                if n_epi > days and n_epi % 1000 == 0:
                    runing_bar.set_postfix(cost=sum(cost_history[n_epi-days:n_epi]))
                
                # 버퍼가 start_size 이상이면 online 모델 업데이트
                if len(self.replay_buffer) > start_size:
                    self.optimize_model(Tf)

                # freq 마다 target 신경망 업데이트 
                if n_epi % freq == 0 and n_epi != 0:
                    self.update_network()

        return cost_history, action_history, battery_history


class DualingDDQN():
    def __init__(self, 
                 replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 training_strategy_fn,
                 cal_price,
                 next_state,
                 batch_size,
                 gamma,
                 battery_max,
                 tau
                 ):
        """
        replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                cal_price,
                next_state,
                batch_size,
                gamma,
                battery_max
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.cal_price = cal_price
        self.next_state = next_state
        self.tau = tau

    def optimize_model(self, Tf):
        """
        online 신경망 학습
        """
        s, a, r, s_prime = self.replay_buffer.sample(self.batch_size)

        q_a = self.online_model(s_prime)
        # min index 뽑기
        min_q_a = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_a, Tf).min(1)[1]
        
        q_sp = self.target_model(s_prime).detach()
        min_q_prime = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_sp, Tf)
        Q_target = min_q_prime[np.arange(self.batch_size), min_q_a]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        # r + discount * max(q)
        target = r + self.gamma * Q_target

        q_a = self.online_model(s).gather(1,a)

        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max)
        return action
    
    def update_network(self, tau=None):
        """
        target 신경망 업데이트
        """
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):

            # target 신경망에 너무 급격한 변화를 막기 위해서 폴샤크 평균을 사용 tau는 파라미터 변화율
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, 
            feature, 
            epochs,    
            start_day, 
            end_day, 
            summer_TOU, 
            Tf,
            pD,
            load_data, 
            generation_data, 
            days, 
            T, 
            start_size, 
            freq,
            init_state,   
            ):

        # online target 신경망 할당
        self.online_model = self.value_model_fn(feature)
        self.target_model = self.value_model_fn(feature)
        self.update_network(tau=1.0)

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        
        cost_history, battery_history, action_history = [],[],[]
        
        state = init_state

        # init 배터리, 행동
        battery, action = 0, 0
        
        with tqdm(range(epochs), unit="Run") as runing_bar:
            for n_epi in runing_bar:
                cum_cost, day_action, day_battery = 0, [] , []
                #if (start_day + n_epi % end_day) % 365 < 90 or (start_day + n_epi % end_day) % 365 >= 273:TOU = winter_TOU
                #else: TOU = summer_TOU
                TOU = summer_TOU
                if n_epi % (end_day-start_day) == 0: state[0] = 0
                for time in range(24):
                    battery = state[0] - state[50] + state[51+Tf]
                    if battery > self.battery_max: battery = self.battery_max  # generation 초과 제한
                    
                    # online 신경망 행동
                    action = self.interaction_step(torch.from_numpy(state).float(), battery)
                    
                    # 하루 행동 및 배터리 추가
                    day_action.append(action)
                    day_battery.append(state[0])
                    
                    # cost 계산
                    cost = self.cal_price(time=time,charge=action,day_charge=day_action,
                                    TOU=TOU, pD=pD)
                    cum_cost = cum_cost + cost
                    
                    # 다음 state 계산
                    state_prime = self.next_state(n_epi=n_epi,time=time,
                                                battery=battery,charge=action,day_charge=day_action,
                                                TOU=TOU,load_data=load_data,generation_data=generation_data,  
                                                feature=feature, Tf=Tf, days=days, T=T)
                    
                    # buffer에 추가
                    self.replay_buffer.put((state, action, cost, state_prime))
                    
                    # 갱신
                    state = state_prime

                cost_history.append(cum_cost)
                action_history.append(day_action)
                battery_history.append(day_battery)
                
                if n_epi > days and n_epi % 1000 == 0:
                    runing_bar.set_postfix(cost=sum(cost_history[n_epi-days:n_epi]))
                
                # 버퍼가 start_size 이상이면 online 모델 업데이트
                if len(self.replay_buffer) > start_size:
                    self.optimize_model(Tf)

                # freq 마다 target 신경망 업데이트 
                if n_epi % freq == 0 and n_epi != 0:
                    self.update_network()

        return cost_history, action_history, battery_history


class PER():
    def __init__(self, 
                 replay_buffer_fn, 
                 value_model_fn, 
                 value_optimizer_fn, 
                 value_optimizer_lr,
                 max_gradient_norm,
                 training_strategy_fn,
                 cal_price,
                 next_state,
                 batch_size,
                 gamma,
                 battery_max,
                 tau
                 ):
        """
        replay_buffer_fn,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                cal_price,
                next_state,
                batch_size,
                gamma,
                battery_max
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.max_gradient_norm = max_gradient_norm
        self.training_strategy_fn = training_strategy_fn
        self.cal_price = cal_price
        self.next_state = next_state
        self.tau = tau

    def optimize_model(self, experiences, Tf):
        """
        online 신경망 학습
        """
        idxs, weights, \
        (s, a, r, s_prime) = experiences #self.replay_buffer.sample(self.batch_size)

        q_a = self.online_model(s_prime)
        # min index 뽑기
        min_q_a = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_a, Tf).min(1)[1]
        
        q_sp = self.target_model(s_prime).detach()
        min_q_prime = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_sp, Tf)
        Q_target = min_q_prime[np.arange(self.batch_size), min_q_a]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        # r + discount * max(q)
        target = r + self.gamma * Q_target

        q_a = self.online_model(s).gather(1,a)

        td_error = q_a - target
        loss = (weights * td_error).pow(2).mul(0.5).mean()
        #loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.max_gradient_norm)
        self.value_optimizer.step()

        priorities = np.abs(loss.detach().cpu().numpy())
        self.replay_buffer.update(idxs, priorities)
        
    def interaction_step(self, state, battery):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max)
        return action
    
    def update_network(self, tau=None):
        """
        target 신경망 업데이트
        """
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):

            # target 신경망에 너무 급격한 변화를 막기 위해서 폴샤크 평균을 사용 tau는 파라미터 변화율
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)

    def train(self, 
            feature, 
            epochs,    
            start_day, 
            end_day, 
            summer_TOU, 
            Tf,
            pD,
            load_data, 
            generation_data, 
            days, 
            T, 
            start_size, 
            freq,
            init_state,   
            ):

        # online target 신경망 할당
        self.online_model = self.value_model_fn(feature)
        self.target_model = self.value_model_fn(feature)
        self.update_network(tau=1.0)

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        
        cost_history, battery_history, action_history = [],[],[]
        
        state = init_state

        # init 배터리, 행동
        battery, action = 0, 0
        
        with tqdm(range(epochs), unit="Run") as runing_bar:
            for n_epi in runing_bar:
                cum_cost, day_action, day_battery = 0, [] , []
                #if (start_day + n_epi % end_day) % 365 < 90 or (start_day + n_epi % end_day) % 365 >= 273:TOU = winter_TOU
                #else: TOU = summer_TOU
                TOU = summer_TOU
                if n_epi % (end_day-start_day) == 0: state[0] = 0
                for time in range(24):
                    battery = state[0] - state[50] + state[51+Tf]
                    if battery > self.battery_max: battery = self.battery_max  # generation 초과 제한
                    
                    # online 신경망 행동
                    action = self.interaction_step(torch.from_numpy(state).float(), battery)
                    
                    # 하루 행동 및 배터리 추가
                    day_action.append(action)
                    day_battery.append(state[0])
                    
                    # cost 계산
                    cost = self.cal_price(time=time,charge=action,day_charge=day_action,
                                    TOU=TOU, pD=pD)
                    cum_cost = cum_cost + cost
                    
                    # 다음 state 계산
                    state_prime = self.next_state(n_epi=n_epi,time=time,
                                                battery=battery,charge=action,day_charge=day_action,
                                                TOU=TOU,load_data=load_data,generation_data=generation_data,  
                                                feature=feature, Tf=Tf, days=days, T=T)
                    
                    # buffer에 추가
                    experience = (state, action, cost, state_prime)
                    self.replay_buffer.put(experience)
                    
                    # 갱신
                    state = state_prime

                cost_history.append(cum_cost)
                action_history.append(day_action)
                battery_history.append(day_battery)
                
                if n_epi > days and n_epi % 1000 == 0:
                    runing_bar.set_postfix(cost=sum(cost_history[n_epi-days:n_epi]))
                
                # 버퍼가 start_size 이상이면 online 모델 업데이트
                if len(self.replay_buffer) > start_size:
                    experiences = self.replay_buffer.sample()
                    idxs, weights, samples = experiences
                    experiences = (idxs, weights) + (samples,)
                    self.optimize_model(Tf=Tf, experiences=experiences)

                # freq 마다 target 신경망 업데이트 
                if n_epi % freq == 0 and n_epi != 0:
                    self.update_network()

        return cost_history, action_history, battery_history