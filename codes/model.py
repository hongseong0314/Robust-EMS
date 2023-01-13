from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from codes.utills import over_flow_battery, create_directory, over_flow_battery_PER

class Qnet(torch.nn.Module):
    def __init__(self, nS):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(nS, 128)
        self.fc2 = torch.nn.Linear(128,128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, 21)
        self.relu = torch.nn.ReLU()
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DualingQnet(torch.nn.Module):
    def __init__(self, nS):
        super(DualingQnet, self).__init__()
        self.fc1 = torch.nn.Linear(nS, 128)
        self.fc2 = torch.nn.Linear(128,128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.relu = torch.nn.ReLU()
        
        self.output_value = torch.nn.Linear(128, 1) # V(s)
        self.output_layer = torch.nn.Linear(128, 21) # A(s, a)
        self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        a = self.output_layer(x)
        v = self.output_value(x).expand_as(a)

        q = v + a - a.mean(-1, keepdim=True).expand_as(a)
        return q

class DQN():
    def __init__(self, cfg):
        self.battery_max = cfg.EMS.battery_max
        self.start_day = cfg.EMS.start_day  
        self.end_day = cfg.EMS.end_day  
        self.summer_TOU = cfg.EMS.summer_TOU  
        self.winter_TOU = cfg.EMS.winter_TOU 
        self.days = cfg.EMS.days 

        self.env = cfg.env
        
        self.Tf = cfg.model.Tf 
        self.batch_size = cfg.model.batch_size
        self.gamma = cfg.model.gamma
        self.replay_buffer_fn = cfg.model.replay_buffer_fn
        self.value_model_fn = cfg.model.value_model_fn
        self.value_optimizer_fn = cfg.model.value_optimizer_fn
        self.value_optimizer_lr = cfg.model.value_optimizer_lr
        self.training_strategy_fn = cfg.model.training_strategy_fn
        self.nS = cfg.model.nS  
        self.epochs = cfg.model.epochs  
        self.start_size = 3650 
        self.freq = cfg.model.freq 
        self.violation = cfg.model.violation

    def optimize_model(self):
        """
        online 신경망 학습
        """
        s, a, r, s_prime, _, day = self.replay_buffer.sample(self.batch_size)
        times = np.argmax(s[:, 2:26], axis=1)
        q_a = self.online_model(s).gather(1,a)

        # target 신경망
        min_q_prime = self.target_model(s_prime).detach()
        Q_target = over_flow_battery(self.batch_size, self.battery_max, s_prime, min_q_prime, self.Tf, day, times, self.env.market_limit, self.days).min(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        
        # r + discount * max(q)
        target = r + self.gamma * Q_target
        
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[self.epochs_completed % self.end_day][time])
        return action
    
    def update_network(self):
        """
        target 신경망 업데이트
        """
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)
    
    def setup(self):
        self.cost_history = []
        self.action_history = []
        self.battery_history = []

        self.online_model = self.value_model_fn(self.nS)
        self.target_model = self.value_model_fn(self.nS)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, 
                                                        step_size=1, 
                                                        gamma=0.985)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0

        if self.violation:
            self.violation_list = []
            self.vio = 0

        pass
    
    def epoch(self, runing_bar):
        self.cum_cost, self.day_action, self.day_battery = 0, [] , []
        if (self.start_day + self.epochs_completed % self.end_day) % 365 < 90 or (self.start_day + self.epochs_completed % self.end_day) % 365 >= 273:
            self.TOU = self.winter_TOU
        else: 
            self.TOU = self.summer_TOU
        
        # 하루 시작
        self.time = 0
        for _ in range(24):
            next(self.stepper)
            self.time += 1
        self.epochs_completed += 1
        if self.epochs_completed % 1000 == 0:
            self.scheduler.step()

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), violation=self.vio)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        if len(self.replay_buffer) > self.start_size:
            for _ in range(20):
                self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()

        if self.epochs_completed >= 20000:
            online_model_name = "DQN/{}.pth".format(self.epochs_completed)
            torch.save(self.online_model.state_dict(), online_model_name)
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
 
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed % self.days, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation, self.epochs_completed % self.days))
            
            state = new_state

            if self.violation:
                self.vio += violation
            yield battery
    
class DDQN():
    def __init__(self, cfg):
        self.battery_max = cfg.EMS.battery_max
        self.start_day = cfg.EMS.start_day  
        self.end_day = cfg.EMS.end_day  
        self.summer_TOU = cfg.EMS.summer_TOU  
        self.winter_TOU = cfg.EMS.winter_TOU 
        self.days = cfg.EMS.days 

        self.env = cfg.env
        
        self.Tf = cfg.model.Tf 
        self.batch_size = cfg.model.batch_size
        self.gamma = cfg.model.gamma
        self.replay_buffer_fn = cfg.model.replay_buffer_fn
        self.value_model_fn = cfg.model.value_model_fn
        self.value_optimizer_fn = cfg.model.value_optimizer_fn
        self.value_optimizer_lr = cfg.model.value_optimizer_lr
        self.training_strategy_fn = cfg.model.training_strategy_fn
        self.nS = cfg.model.nS  
        self.epochs = cfg.model.epochs  
        self.start_size = 3650 
        self.freq = cfg.model.freq 
        self.violation = cfg.model.violation

    def optimize_model(self):
        """
        online 신경망 학습
        """
        s, a, r, s_prime, _, day = self.replay_buffer.sample(self.batch_size)
        times = np.argmax(s[:, 2:26], axis=1)
        q_a = self.online_model(s).gather(1,a)
        min_q_a = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_a, self.Tf, day, times, self.env.market_limit, self.days).min(1)[1]
        
        q_sp = self.target_model(s_prime).detach()
        min_q_prime = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_sp, self.Tf, day, times, self.env.market_limit, self.days).min(1)[0]
        Q_target = min_q_prime[np.arange(self.batch_size), min_q_a]
        
        # r + discount * max(q)
        target = r + self.gamma * Q_target

        q_a = self.online_model(s).gather(1,a)
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[self.epochs_completed % self.end_day][time])
        return action
    
    def update_network(self):
        """
        target 신경망 업데이트
        """
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)
    
    def setup(self):
        self.cost_history = []
        self.action_history = []
        self.battery_history = []

        self.online_model = self.value_model_fn(self.nS)
        self.target_model = self.value_model_fn(self.nS)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, 
                                                        step_size=1, 
                                                        gamma=0.985)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0

        if self.violation:
            self.violation_list = []
            self.vio = 0

        pass
    
    def epoch(self, runing_bar):
        self.cum_cost, self.day_action, self.day_battery = 0, [] , []
        if (self.start_day + self.epochs_completed % self.end_day) % 365 < 90 or (self.start_day + self.epochs_completed % self.end_day) % 365 >= 273:
            self.TOU = self.winter_TOU
        else: 
            self.TOU = self.summer_TOU
        
        # 하루 시작
        self.time = 0
        for _ in range(24):
            next(self.stepper)
            self.time += 1
        self.epochs_completed += 1
        if self.epochs_completed % 1000 == 0:
            self.scheduler.step()

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), violation=self.vio)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        if len(self.replay_buffer) > self.start_size:
            for _ in range(20):
                self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()

        if self.epochs_completed >= 20000:
            online_model_name = "DDQN/{}.pth".format(self.epochs_completed)
            torch.save(self.online_model.state_dict(), online_model_name)
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
 
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed % self.days, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation, self.epochs_completed % self.days))
            
            state = new_state

            if self.violation:
                self.vio += violation
            yield battery

class DualingDDQN():
    def __init__(self, cfg):
        self.battery_max = cfg.EMS.battery_max
        self.start_day = cfg.EMS.start_day  
        self.end_day = cfg.EMS.end_day  
        self.summer_TOU = cfg.EMS.summer_TOU  
        self.winter_TOU = cfg.EMS.winter_TOU 
        self.days = cfg.EMS.days 

        self.env = cfg.env
        
        self.Tf = cfg.model.Tf 
        self.batch_size = cfg.model.batch_size
        self.gamma = cfg.model.gamma
        self.replay_buffer_fn = cfg.model.replay_buffer_fn
        self.value_model_fn = cfg.model.value_model_fn
        self.value_optimizer_fn = cfg.model.value_optimizer_fn
        self.value_optimizer_lr = cfg.model.value_optimizer_lr
        self.training_strategy_fn = cfg.model.training_strategy_fn
        self.nS = cfg.model.nS  
        self.epochs = cfg.model.epochs  
        self.start_size = 3650 
        self.freq = cfg.model.freq 
        self.violation = cfg.model.violation
        self.tau = cfg.model.tau

    def optimize_model(self):
        """
        online 신경망 학습
        """
        s, a, r, s_prime, _, day = self.replay_buffer.sample(self.batch_size)
        times = np.argmax(s[:, 2:26], axis=1)
        q_a = self.online_model(s).gather(1,a)
        min_q_a = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_a, self.Tf, day, times, self.env.market_limit, self.days).min(1)[1]
        
        q_sp = self.target_model(s_prime).detach()
        min_q_prime = over_flow_battery(self.batch_size, self.battery_max, s_prime, q_sp, self.Tf, day, times, self.env.market_limit, self.days)
        Q_target = min_q_prime[np.arange(self.batch_size), min_q_a]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        
        # r + discount * max(q)
        target = r + self.gamma * Q_target
        
        q_aa = self.online_model(s).gather(1,a)

        loss = F.smooth_l1_loss(q_aa, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[self.epochs_completed % self.end_day][time])
        return action
    
    def update_network(self, tau=None):
        """
        target 신경망 업데이트
        """
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target_ratio = (1.0 - tau) * target.data
            online_ratio = tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
    
    def setup(self):
        self.cost_history = []
        self.action_history = []
        self.battery_history = []

        self.online_model = self.value_model_fn(self.nS)
        self.target_model = self.value_model_fn(self.nS)
        self.update_network(1.0)

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, 
                                                        step_size=1, 
                                                        gamma=0.985)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0

        if self.violation:
            self.violation_list = []
            self.vio = 0

        pass
    
    def epoch(self, runing_bar):
        self.cum_cost, self.day_action, self.day_battery = 0, [] , []
        if (self.start_day + self.epochs_completed % self.end_day) % 365 < 90 or (self.start_day + self.epochs_completed % self.end_day) % 365 >= 273:
            self.TOU = self.winter_TOU
        else: 
            self.TOU = self.summer_TOU
        
        # 하루 시작
        self.time = 0
        for _ in range(24):
            next(self.stepper)
            self.time += 1
        self.epochs_completed += 1
        if self.epochs_completed % 1000 == 0:
            self.scheduler.step()

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), violation=self.vio)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        if len(self.replay_buffer) > self.start_size:
            for _ in range(20):
                self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()

        if self.epochs_completed >= 20000:
            online_model_name = "DualingDDQN/{}.pth".format(self.epochs_completed)
            torch.save(self.online_model.state_dict(), online_model_name)
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
 
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed % self.days, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation, self.epochs_completed % self.days))
            
            state = new_state

            if self.violation:
                self.vio += violation
            yield battery

class PER():
    def __init__(self, cfg):
        self.battery_max = cfg.EMS.battery_max
        self.start_day = cfg.EMS.start_day  
        self.end_day = cfg.EMS.end_day  
        self.summer_TOU = cfg.EMS.summer_TOU  
        self.winter_TOU = cfg.EMS.winter_TOU 
        self.days = cfg.EMS.days 

        self.env = cfg.env
        
        self.Tf = cfg.model.Tf 
        self.batch_size = cfg.model.batch_size
        self.gamma = cfg.model.gamma
        self.replay_buffer_fn = cfg.model.replay_buffer_fn
        self.value_model_fn = cfg.model.value_model_fn
        self.value_optimizer_fn = cfg.model.value_optimizer_fn
        self.value_optimizer_lr = cfg.model.value_optimizer_lr
        self.training_strategy_fn = cfg.model.training_strategy_fn
        self.nS = cfg.model.nS  
        self.epochs = cfg.model.epochs  
        self.start_size = 3650
        self.freq = cfg.model.freq 
        self.violation = cfg.model.violation
        self.tau = cfg.model.tau
        self.max_gradient_norm = cfg.model.max_gradient_norm

    def optimize_model(self):
        """
        online 신경망 학습
        """
        idxs, weights, \
        (s, a, r, s_prime, vi, day) = self.replay_buffer.sample(self.batch_size)
        time = np.argmax(s[:, 2:26], axis=1)

        q_a = self.online_model(s_prime)
        min_q_a = over_flow_battery_PER(self.batch_size, self.battery_max, s_prime, q_a, self.Tf, day, time, self.env.market_limit, self.days).min(1)[1]
        
        q_sp = self.target_model(s_prime).detach()
        min_q_prime = over_flow_battery_PER(self.batch_size, self.battery_max, s_prime, q_sp, self.Tf, day, time, self.env.market_limit, self.days)
        Q_target = min_q_prime[np.arange(self.batch_size), min_q_a]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        
        # r + discount * max(q)
        target = r + self.gamma * Q_target

        q_a = self.online_model(s).gather(1,a)

        td_error = q_a - target
        loss = (weights * td_error).pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), 
                                       self.max_gradient_norm)
        self.value_optimizer.step()

        priorities = np.abs(loss.detach().cpu().numpy())
        self.replay_buffer.update(idxs, priorities)
        
    def interaction_step(self, state, battery, time):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[self.epochs_completed % self.end_day][time])
        return action
    
    def update_network(self, tau=None):
        """
        target 신경망 업데이트
        """
        tau = self.tau if tau is None else tau
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target_ratio = (1.0 - self.tau) * target.data
            online_ratio = self.tau * online.data
            mixed_weights = target_ratio + online_ratio
            target.data.copy_(mixed_weights)
    
    def setup(self):
        self.cost_history = []
        self.action_history = []
        self.battery_history = []

        if self.violation:
            self.violation_list = []
            self.vio = 0

        self.online_model = self.value_model_fn(self.nS)
        self.target_model = self.value_model_fn(self.nS)
        self.update_network(1.0)

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, 
                                                        step_size=1, 
                                                        gamma=0.985)

        self.stepper = self.step_run()
        self.epochs_completed = 0

        if self.violation:
            self.violation_list = []
            self.vio = 0

        pass
    
    def epoch(self, runing_bar):
        self.cum_cost, self.day_action, self.day_battery = 0, [] , []
        if (self.start_day + self.epochs_completed % self.end_day) % 365 < 90 or (self.start_day + self.epochs_completed % self.end_day) % 365 >= 273:
            self.TOU = self.winter_TOU
        else: 
            self.TOU = self.summer_TOU
        
        # 하루 시작
        self.time = 0
        for _ in range(24):
            next(self.stepper)
            self.time += 1
        self.epochs_completed += 1
        if self.epochs_completed % 1000 == 0:
            self.scheduler.step()

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), violation=self.vio)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfi(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))
        
        # 버퍼가 start_size 이상이면 online 모델 업데이트
        if len(self.replay_buffer) > self.start_size:
            for _ in range(20):
                self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()

        if self.epochs_completed >= 20000:
            online_model_name = "PER/{}.pth".format(self.epochs_completed)
            torch.save(self.online_model.state_dict(), online_model_name)
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max
            
            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
            
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed % self.days, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation, self.epochs_completed % self.days))
            
            # 갱신
            state = new_state

            if self.violation:
                self.vio += violation
            yield battery

class LRDQN():
    def __init__(self, cfg):
        self.battery_max = cfg.EMS.battery_max
        self.start_day = cfg.EMS.start_day  
        self.end_day = cfg.EMS.end_day  
        self.summer_TOU = cfg.EMS.summer_TOU  
        self.winter_TOU = cfg.EMS.winter_TOU 
        self.days = cfg.EMS.days 

        self.env = cfg.env
        
        self.Tf = cfg.model.Tf 
        self.batch_size = cfg.model.batch_size
        self.gamma = cfg.model.gamma
        self.replay_buffer_fn = cfg.model.replay_buffer_fn
        self.value_model_fn = cfg.model.value_model_fn
        self.value_optimizer_fn = cfg.model.value_optimizer_fn
        self.value_optimizer_lr = cfg.model.value_optimizer_lr
        self.training_strategy_fn = cfg.model.training_strategy_fn
        self.nS = cfg.model.nS  
        self.epochs = cfg.model.epochs  
        self.start_size = 3650
        self.freq = cfg.model.freq 
        self.violation = cfg.model.violation
        self.p = 0.1

    def optimize_model(self):
        """
        online 신경망 학습
        """
        s, a, r, s_prime, vi, day = self.replay_buffer.sample(self.batch_size)
        times = np.argmax(s[:, 2:26], axis=1)
        q_a = self.online_model(s).gather(1,a)

        min_q_prime = self.target_model(s_prime).detach()
        Q_target = over_flow_battery(self.batch_size, self.battery_max, s_prime, min_q_prime, self.Tf, day, times, self.env.market_limit, self.days).min(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        
        # 라그랑주 패널티 부여
        target = r + self.gamma * Q_target
        target[vi.to(torch.bool)] += torch.tensor(self.lagrange_lambda, dtype=torch.float32)
        
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time):
        """
        online 신경망 행동
        """
        action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[self.epochs_completed % self.end_day][time])      
        return action
    
    def update_network(self):
        """
        target 신경망 업데이트
        """
        for target, online in zip(self.target_model.parameters(), 
                                  self.online_model.parameters()):
            target.data.copy_(online.data)
    
    def setup(self):
        self.cost_history = []
        self.action_history = []
        self.battery_history = []

        self.online_model = self.value_model_fn(self.nS)
        self.target_model = self.value_model_fn(self.nS)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.value_optimizer, 
                                                        step_size=1, 
                                                        gamma=0.985)
        self.replay_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0

        # 라그랑주 init 상수
        self.lagrange_lambda = 100

        if self.violation:
            self.violation_list = []
            self.vio = 0

        pass
    
    def epoch(self, runing_bar):
        self.cum_cost, self.day_action, self.day_battery = 0, [] , []
        if (self.start_day + self.epochs_completed % self.end_day) % 365 < 90 or (self.start_day + self.epochs_completed % self.end_day) % 365 >= 273:
            self.TOU = self.winter_TOU
        else: 
            self.TOU = self.summer_TOU
        
        # 하루 시작
        self.time = 0
        for _ in range(24):
            next(self.stepper)
            self.time += 1
        self.epochs_completed += 1
        if self.epochs_completed % 1000 == 0:
            self.scheduler.step()

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), violation=self.vio)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))
        
        # 버퍼가 start_size 이상이면 online 모델 업데이트
        if len(self.replay_buffer) > self.start_size:
            for _ in range(20):
                self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()

        if self.epochs_completed >= 20000:
            online_model_name = "LRDQN/{}.pth".format(self.epochs_completed)
            torch.save(self.online_model.state_dict(), online_model_name)
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max

            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time)

            self.day_action.append(action)
            self.day_battery.append(state[0])
          
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed % self.days, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)
            
            self.cum_cost += reward
            
            self.Multiplier_update(self.epochs_completed*24 + self.time, violation)
            self.replay_buffer.put((state, action, reward, new_state, violation, self.epochs_completed % self.days))

            state = new_state

            if self.violation:
                self.vio += violation
            yield battery

    def Multiplier_update(self, timestep, violation_f):
        self.lagrange_alpha = 1 / (timestep + 1)
        self.lagrange_lambda = np.max([self.lagrange_lambda + self.lagrange_alpha * (violation_f - self.p), 0])