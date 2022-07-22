import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange

from codes.utills import over_flow_battery, over_flow_battery_max, over_flow_battery_3000, target_scale
from codes.buffer import ReplayBuffer_vi
from codes.real_dynamics import T_hat
from src.dynamics import BatchedGaussianEnsemble

class SDQN():
    def __init__(self, 
                 replay_buffer_fn,
                env,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                batch_size,
                gamma,
                battery_max,
                feature, 
                epochs, 
                start_day, 
                end_day, 
                summer_TOU, 
                winter_TOU,
                Tf,
                pD,
                load_data, 
                generation_data, 
                days,
                T, 
                start_size,
                freq,
                violation,
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
        self.env = env

        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.feature = feature  
        self.epochs = epochs  
        self.start_day = start_day  
        self.end_day = end_day  
        self.summer_TOU = summer_TOU  
        self.winter_TOU = winter_TOU 
        self.Tf = Tf 
        self.pD = pD 
        self.load_data = load_data  
        self.generation_data = generation_data  
        self.days = days 
        self.T = T  
        self.start_size = start_size 
        self.freq = freq 
        self.violation = violation
        self.buffer_min = 2000
        self.check_violation = lambda states, action, time : env.check_violation(states.cpu().numpy(), action.cpu().numpy(), time)
        self.model_cfg = BatchedGaussianEnsemble.Config()

        self.rollout_batch_size = 100
        self.horizon = 5

    def optimize_model(self):
        """
        online 신경망 학습
        """
        n_real = int(self.fraction * self.batch_size)
        real_samples = self.replay_buffer.sample(n_real)
        virt_samples = self.virt_buffer.sample(self.batch_size - n_real)
        combined_samples = [
            torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)
        ]

        s, a, r, s_prime, vi = combined_samples
        q_a = self.online_model(s).gather(1,a)

        # target 신경망
        min_q_prime = self.target_model(s_prime).detach()
        min_q_prime = target_scale(min_q_prime)
        Q_target = over_flow_battery_3000(self.batch_size, self.battery_max, s_prime, min_q_prime, self.Tf).min(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        # print("Q : ",Q_target)

        # r + discount * max(q)
        target = r + self.gamma * Q_target
        target[vi.to(torch.bool)] = self.violation_cost / (1. - self.gamma) 

        # print("vi",target[vi.to(torch.bool)])
        # print("not vi",target[~vi.to(torch.bool)])
        # import time
        # time.sleep(3)
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time, roll=False):
        """
        online 신경망 행동
        """
        if state.ndim == 2:
            action_list = []
            for s, b, t in zip(state, battery, time):
                action = self.training_strategy.select_action(self.online_model, s, b, self.battery_max, self.env.market_limit[t], roll)
                action_list.append(action)
            return np.vstack(action_list)
        else:
            action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[time], roll)
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

        self.online_model = self.value_model_fn(self.feature)
        self.target_model = self.value_model_fn(self.feature)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.virt_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, self.feature, 1)
        self.fraction = 0.3
        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0
        self.model_update_period = 24 * 5
        
        if self.violation:
            self.violation_list = []
            self.vio = 0
            self.violation_cost=0

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

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), 
                                    violation=self.vio,
                                    T_loss=self.model_loss)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        
        # 버퍼가 start_size 이상이면 online 모델 업데이트
        if len(self.virt_buffer) > self.start_size and self.epochs_completed % 1 == 0:
            self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max
            
            if len(self.replay_buffer) > self.buffer_min:
                if self.step_count % self.model_update_period == 0:
                    self.update_models()
                if self.step_count % 3 == 0:
                    self.rollout()
                # 버퍼가 start_size 이상이면 online 모델 업데이트
                # if len(self.virt_buffer) > self.start_size and self.epochs_completed % 1 == 0:
                #     self.optimize_model()

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
            
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation))
            
            # 갱신
            state = new_state

            if self.violation:
                self.vio += violation
            
            self.step_count += 1
            yield battery

    def rollout(self, initial_states=None):
        if initial_states is None:
            initial_states = np.random.choice(self.replay_buffer.ss_mem[:self.replay_buffer.size], size=self.rollout_batch_size)

        buffer = ReplayBuffer_vi(self.rollout_batch_size * self.horizon, batch_size=self.batch_size)
        states = torch.from_numpy(np.vstack(initial_states).astype(np.float32)).float()
        ob_time = np.argmax(states[:, 2:26], axis=1).numpy()
        
        for t in range(self.horizon):
            battery = (states[:, 0] - states[:, 50] + states[:, 51]).numpy()
            battery = np.clip(battery, a_min=None, a_max=self.battery_max)
            #ob_time = np.argmax(states[:, 2:26], axis=1).numpy()

            with torch.no_grad():
                actions = self.interaction_step(states, battery, ob_time, roll=True) # roll=True
                next_states, rewards = self.model_ensemble.sample(states, torch.from_numpy(actions))
                next_states, rewards = next_states.detach(), rewards.detach()

            violations = self.env.check_violation(next_states, actions.squeeze(-1), ob_time)
            
            for s,a,r,s_,vi in zip(states.numpy(), actions.squeeze(-1), rewards.numpy(), next_states.numpy(), violations):
                buffer.put((s, a, r, s_, vi))
                self.virt_buffer.put((s, a, r, s_, vi))
            
            continues = ~(violations | (ob_time+1 > 23))
            if continues.sum() == 0:
                break
            states = next_states[continues]
            battery = battery[continues]
            ob_time = ob_time[continues] + 1

        return buffer
    
    def update_models(self, steps=10):
        model_losses = self.fit(steps=steps)
        self.model_loss = np.mean(model_losses)
        buffer_reward = np.random.choice(self.replay_buffer.rs_mem[:self.replay_buffer.size], size=self.replay_buffer.size)
        r_min = buffer_reward.min()
        r_max = buffer_reward.max()
        self.update_r_bounds(r_min, r_max) 
    
    def fit(self, steps=10, progress_bar=False):
        n = self.replay_buffer.size
        
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(n)
        self.model_ensemble.state_normalizer.fit(states)
        targets = torch.cat([next_states, rewards], dim=1)
        
        losses = []
        for _ in (trange if progress_bar else range)(steps):
            # indices = random_indices(n, size=self.total_batch_size, replace=False)
            indices = torch.randint(n, [self.model_ensemble.total_batch_size])
            loss = self.model_ensemble.compute_loss(states[indices], actions[indices], targets[indices])
            losses.append(loss.item())
            self.model_ensemble.optimizer.zero_grad()
            loss.backward()
            self.model_ensemble.optimizer.step()
        return losses
    
    def update_r_bounds(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        self.violation_cost = (r_max - r_min) / self.gamma**self.horizon - r_max

class ESDQN():
    def __init__(self, 
                 replay_buffer_fn,
                env,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                batch_size,
                gamma,
                battery_max,
                feature, 
                epochs, 
                start_day, 
                end_day, 
                summer_TOU, 
                winter_TOU,
                Tf,
                pD,
                load_data, 
                generation_data, 
                days,
                T, 
                start_size,
                freq,
                violation,
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
        self.env = env

        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.feature = feature  
        self.epochs = epochs  
        self.start_day = start_day  
        self.end_day = end_day  
        self.summer_TOU = summer_TOU  
        self.winter_TOU = winter_TOU 
        self.Tf = Tf 
        self.pD = pD 
        self.load_data = load_data  
        self.generation_data = generation_data  
        self.days = days 
        self.T = T  
        self.start_size = start_size 
        self.freq = freq 
        self.violation = violation
        self.buffer_min = 2000
        self.check_violation = lambda states, action, time : env.check_violation(states.cpu().numpy(), action.cpu().numpy(), time)
        self.T_hat = T_hat(load_data, 
                            generation_data, 
                            feature, 
                            summer_TOU, 
                            T, 
                            days, 
                            pD,)

        self.rollout_batch_size = 100
        self.horizon = 10

    def optimize_model(self):
        """
        online 신경망 학습
        """
        n_real = int(self.fraction * self.batch_size)
        real_samples = self.replay_buffer.sample(n_real)[:-1]
        virt_samples = self.virt_buffer.sample(self.batch_size - n_real)
        
        combined_samples = [
            torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)
        ]

        s, a, r, s_prime, vi = combined_samples
        q_a = self.online_model(s).gather(1,a)

        # target 신경망
        min_q_prime = self.target_model(s_prime).detach()
        #min_q_prime = target_scale(min_q_prime)
        Q_target = over_flow_battery_3000(self.batch_size, self.battery_max, s_prime, min_q_prime, self.Tf).min(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        # print("Q : ",Q_target)

        # r + discount * max(q)
        target = r + self.gamma * Q_target
        target[vi.to(torch.bool)] = self.violation_cost / (1. - self.gamma) 
        print(f"C {self.violation_cost / (1. - self.gamma)}")
        print("vi",target[vi.to(torch.bool)])
        print("not vi",target[~vi.to(torch.bool)])
        # import time
        # time.sleep(3)
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time, roll=False):
        """
        online 신경망 행동
        """
        if state.ndim == 2:
            action_list = []
            for s, b, t in zip(state, battery, time):
                action = self.training_strategy.select_action(self.online_model, s, b, self.battery_max, self.env.market_limit[t], roll)
                action_list.append(action)
            return np.vstack(action_list)
        else:
            action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[time], roll)
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

        self.online_model = self.value_model_fn(self.feature)
        self.target_model = self.value_model_fn(self.feature)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.virt_buffer = ReplayBuffer_vi(15000, batch_size=self.batch_size)
        self.training_strategy = self.training_strategy_fn()

        self.fraction = 0.2
        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0
        self.model_update_period = 24
        
        if self.violation:
            self.violation_list = []
            self.vio = 0
            self.violation_cost=0

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

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), 
                                    violation=self.vio
                                    )

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        
        # # 버퍼가 start_size 이상이면 online 모델 업데이트
        if len(self.virt_buffer) > self.start_size and self.epochs_completed % 1 == 0:
            self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max
            
            if len(self.replay_buffer) > self.buffer_min:
                if self.step_count % self.model_update_period == 0:
                    self.update_models()
                if self.step_count % 3 == 0:
                    self.rollout()
                # 버퍼가 start_size 이상이면 online 모델 업데이트
                # if len(self.virt_buffer) > self.start_size and self.epochs_completed % 1 == 0:
                #     self.optimize_model()

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
            
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation, self.epochs_completed % self.days))
            
            # 갱신
            state = new_state

            if self.violation:
                self.vio += violation
            
            self.step_count += 1
            yield battery

    def rollout(self, idxs=None):
        if idxs is None:
            idxs = np.random.choice(len(self.replay_buffer), size=self.rollout_batch_size)

        buffer = ReplayBuffer_vi(self.rollout_batch_size * self.horizon, batch_size=self.batch_size)
        states = np.vstack(self.replay_buffer.ss_mem[idxs]).astype(np.float32)
        ep = self.replay_buffer.e_mem[idxs].astype(np.int64)
        ob_time = np.argmax(states[:, 2:26], axis=1)
     
        for t in range(self.horizon):
            battery = states[:, 0] - states[:, 50] + states[:, 51]
            battery = np.clip(battery, a_min=None, a_max=self.battery_max)
        
            with torch.no_grad():
                actions = self.interaction_step(torch.from_numpy(states).float(), battery, ob_time, roll=True) # roll=True
            
            next_states, rewards = self.T_hat.transition(states, actions.squeeze(-1), ob_time, ep, battery)

            violations = self.env.check_violation(next_states, actions.squeeze(-1), ob_time)
            
            for s,a,r,s_,vi in zip(states, actions.squeeze(-1), rewards, next_states, violations):
                buffer.put((s, a, r, s_, vi))
                self.virt_buffer.put((s, a, r, s_, vi))
            
            # max day and max hour check 
            logic = np.logical_or((ob_time+1 > 23), (ep+1 >= self.days))
            continues = ~(violations | logic) 
            
            #print(violations)
            if continues.sum() == 0:
                break
            states = next_states[continues]
            battery = battery[continues]
            ob_time = ob_time[continues] + 1
            ep = ep[continues] + 1

        return buffer
    
    def update_models(self, steps=10):
        buffer_reward = np.random.choice(self.replay_buffer.rs_mem[:self.replay_buffer.size], size=self.replay_buffer.size)
        r_min = buffer_reward.min()
        r_max = buffer_reward.max()
        self.update_r_bounds(r_min, r_max) 
    
    def update_r_bounds(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        self.violation_cost = (r_max - r_min) / self.gamma**self.horizon - r_max



class MSDQN():
    def __init__(self, 
                 replay_buffer_fn,
                env,
                value_model_fn,
                value_optimizer_fn,
                value_optimizer_lr,
                training_strategy_fn,
                batch_size,
                gamma,
                battery_max,
                feature, 
                epochs, 
                start_day, 
                end_day, 
                summer_TOU, 
                winter_TOU,
                Tf,
                pD,
                load_data, 
                generation_data, 
                days,
                T, 
                start_size,
                freq,
                violation,
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
        self.env = env

        self.batch_size = batch_size
        self.gamma = gamma
        self.battery_max = battery_max
        self.replay_buffer_fn = replay_buffer_fn
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.training_strategy_fn = training_strategy_fn
        self.feature = feature  
        self.epochs = epochs  
        self.start_day = start_day  
        self.end_day = end_day  
        self.summer_TOU = summer_TOU  
        self.winter_TOU = winter_TOU 
        self.Tf = Tf 
        self.pD = pD 
        self.load_data = load_data  
        self.generation_data = generation_data  
        self.days = days 
        self.T = T  
        self.start_size = start_size 
        self.freq = freq 
        self.violation = violation
        self.check_violation = lambda states, action, time : env.check_violation(states.cpu().numpy(), action.cpu().numpy(), time)
        self.model_cfg = BatchedGaussianEnsemble.Config()

        self.rollout_batch_size = 100
        self.horizon = 10

    def optimize_model(self):
        """
        online 신경망 학습
        """
        n_real = int(self.fraction * self.batch_size)
        real_samples = self.replay_buffer.sample(n_real)
        virt_samples = self.virt_buffer.sample(self.batch_size - n_real)
        combined_samples = [
            torch.cat([real, virt]) for real, virt in zip(real_samples, virt_samples)
        ]

        s, a, r, s_prime, vi = combined_samples
        q_a = self.online_model(s).gather(1,a)

        # target 신경망
        min_q_prime = self.target_model(s_prime).detach()
        Q_target = over_flow_battery_max(self.batch_size, self.battery_max, s_prime, min_q_prime, self.Tf).max(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)
        
        # r + discount * max(q)
        target = -r + self.gamma * Q_target
        #target = (target - target.min()) / (target.max() - target.min()) * 2000
        target[vi.to(torch.bool)] = -self.violation_cost / (1. - self.gamma) 

        # print("vi",target[vi.to(torch.bool)])
        # print("not vi",target[~vi.to(torch.bool)])
        # import time
        # time.sleep(3)
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery):
        """
        online 신경망 행동
        """
        if state.ndim == 2:
            action_list = []
            for s, b in zip(state, battery):
                action = self.training_strategy.select_action(self.online_model, s, b, self.battery_max)
                action_list.append(action)
            return np.vstack(action_list)
        else:
            action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max)
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

        self.online_model = self.value_model_fn(self.feature)
        self.target_model = self.value_model_fn(self.feature)
        self.update_network()

        self.value_optimizer = self.value_optimizer_fn(self.online_model, 
                                                       self.value_optimizer_lr)

        self.replay_buffer = self.replay_buffer_fn()
        self.virt_buffer = self.replay_buffer_fn()
        self.training_strategy = self.training_strategy_fn()

        self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, self.feature, 1)
        self.fraction = 0.1
        self.stepper = self.step_run()
        self.epochs_completed = 0
        self.step_count = 0
        self.model_update_period = 24 * 50
        if self.violation:
            self.violation_list = []
            self.vio = 0
            self.violation_cost=0

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

        self.cost_history.append(self.cum_cost)
        self.action_history.append(self.day_action)
        self.battery_history.append(self.day_battery)
        
        if self.violation:
            self.violation_list.append(self.vio)

        if self.epochs_completed > self.days and self.epochs_completed % 1000 == 0 and self.violation:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), 
                                    violation=self.vio,
                                    T_loss=self.model_loss)

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        
        # 버퍼가 start_size 이상이면 online 모델 업데이트
        if len(self.virt_buffer) > self.start_size and self.epochs_completed % 1 == 0:
            self.optimize_model()

        # freq 마다 target 신경망 업데이트 
        if self.epochs_completed % self.freq == 0 and self.epochs_completed != 0:
            self.update_network()
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max
            
            if self.step_count > 5000:
                if self.step_count % self.model_update_period == 0:
                    self.update_models()
                if self.step_count % 24 == 0:
                    self.rollout()

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery)
            
            # 하루 행동 및 배터리 추가
            self.day_action.append(action)
            self.day_battery.append(state[0])
            
            # step
            new_state, reward, violation = self.env.step(n_epi=self.epochs_completed, time=self.time, battery=battery, charge=action, 
                                day_charge=self.day_action, TOU=self.TOU)

            
            self.cum_cost += reward
            
            # buffer에 추가
            self.replay_buffer.put((state, action, reward, new_state, violation))
            
            # 갱신
            state = new_state

            if self.violation:
                self.vio += violation
            
            self.step_count += 1
            yield battery

    def rollout(self, initial_states=None):
        if initial_states is None:
            initial_states = np.random.choice(self.replay_buffer.ss_mem[:self.replay_buffer.size], size=self.rollout_batch_size)

        buffer = ReplayBuffer_vi(self.rollout_batch_size * self.horizon, batch_size=self.batch_size)
        states = torch.from_numpy(np.vstack(initial_states).astype(np.float32)).float()

        for t in range(self.horizon):
            battery = (states[:, 0] - states[:, 50] + states[:, 51]).numpy()
            battery = np.clip(battery, a_min=None, a_max=self.battery_max)
            ob_time = np.argmax(states[:, 2:26], axis=1).numpy()

            with torch.no_grad():
                actions = self.interaction_step(states, battery)
                next_states, rewards = self.model_ensemble.sample(states, torch.from_numpy(actions))
                next_states, rewards = next_states.detach(), rewards.detach()

            violations = self.env.check_violation(next_states, actions.squeeze(-1), ob_time)
            
            for s,a,r,s_,vi in zip(states.numpy(), actions.squeeze(-1), rewards.numpy(), next_states.numpy(), violations):
                buffer.put((s, a, r, s_, vi))
                self.virt_buffer.put((s, a, r, s_, vi))
            
            continues = ~(violations | (ob_time+1 > 23))
            if continues.sum() == 0:
                break
            states = next_states[continues]
            battery = battery[continues]
            ob_time = ob_time[continues] + 1
    
    def update_models(self):
        model_losses = self.fit(steps=10)
        # start_loss_average = np.mean(model_losses[:10])
        # end_loss_average = np.mean(model_losses[-10:])
        # print(f'Loss statistics:')
        # print(f'\tFirst {10}: {start_loss_average}')
        # print(f'\tLast {10}: {end_loss_average}')
        self.model_loss = np.mean(model_losses)
        buffer_reward = np.random.choice(self.replay_buffer.rs_mem[:self.replay_buffer.size], size=self.replay_buffer.size)
        r_min = buffer_reward.min()
        r_max = buffer_reward.max()
        self.update_r_bounds(r_min, r_max) 
    
    def fit(self, steps=10, progress_bar=False):
        n = self.replay_buffer.size
        
        states, actions, rewards, next_states, _ = self.replay_buffer.sample(n)
        self.model_ensemble.state_normalizer.fit(states)
        targets = torch.cat([next_states, rewards], dim=1)
        
        losses = []
        for _ in (trange if progress_bar else range)(steps):
            # indices = random_indices(n, size=self.total_batch_size, replace=False)
            indices = torch.randint(n, [self.model_ensemble.total_batch_size])
            loss = self.model_ensemble.compute_loss(states[indices], actions[indices], targets[indices])
            losses.append(loss.item())
            self.model_ensemble.optimizer.zero_grad()
            loss.backward()
            self.model_ensemble.optimizer.step()
        return losses
    
    def update_r_bounds(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        self.violation_cost = (r_max - r_min) / self.gamma**self.horizon - r_max