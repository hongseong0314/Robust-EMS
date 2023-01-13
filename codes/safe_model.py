import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from codes.real_dynamics import T_hat
from codes.utills import over_flow_battery, create_directory
from codes.buffer import ReplayBuffer_vi, ReplayBuffer_vi_day

class REMS():
    def __init__(self, cfg):
        self.battery_max = cfg.EMS.battery_max
        self.start_day = cfg.EMS.start_day
        self.end_day = cfg.EMS.end_day
        self.summer_TOU = cfg.EMS.summer_TOU
        self.winter_TOU = cfg.EMS.winter_TOU
        self.days = cfg.EMS.days 

        self.env = cfg.env
        self.T_hat = T_hat(cfg)

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
        self.start_size = cfg.model.batch_size * 10 
        self.freq = cfg.model.freq 
        self.violation = cfg.model.violation
        self.buffer_min = cfg.model.batch_size * 5
        self.rollout_batch_size = 100
        self.horizon = cfg.model.H
        self.q = cfg.model.q
        self.p = cfg.model.p
        self.model_update_interval = 1
        self.obs = cfg.model.obs

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

        s, a, r, s_prime, vi, day = combined_samples
        time = np.argmax(s[:, 2:26], axis=1)

        q_a = self.online_model(s).gather(1,a)
        min_q_prime = self.target_model(s_prime).detach()
        Q_target = over_flow_battery(self.batch_size, self.battery_max, s_prime, min_q_prime, self.Tf, day, time, self.env.market_limit, self.days).min(1)[0]
        Q_target = torch.tensor((Q_target)).resize(self.batch_size, 1)

        # r + discount * max(q)
        target = r + self.gamma * Q_target
        target[vi.to(torch.bool)] = self.violation_cost / (1. - self.gamma) 
        
        loss = F.smooth_l1_loss(q_a, target)
        self.value_optimizer.zero_grad()
        loss.backward()
        self.value_optimizer.step()
        
    def interaction_step(self, state, battery, time, day, roll=False):
        """
        online 신경망 행동
        """
        if state.ndim == 2:
            action_list = []
            for s, b, t, d in zip(state, battery, time, day):
                action = self.training_strategy.select_action(self.online_model, s, b, self.battery_max, self.env.market_limit[d % self.days][t % 24], roll)
                action_list.append(action)
            return action_list
        else:
            action = self.training_strategy.select_action(self.online_model, state, battery, self.battery_max, self.env.market_limit[day % self.days][time % 24], roll)
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
        self.virt_buffer = ReplayBuffer_vi_day(100000, batch_size=self.batch_size)
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

        self.roll_time = []
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
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]), 
                                    violation=self.vio
                                    )

        elif self.epochs_completed > self.days and self.epochs_completed % 1000 == 0:
            runing_bar.set_postfix(cost=sum(self.cost_history[self.epochs_completed-self.days:self.epochs_completed]))

        # 에피소드마다 target 업데이트
        if self.epochs_completed != 0:
            self.update_network()
        
        if self.epochs_completed >= 20000:
            online_model_name = "M/{}.pth".format(self.epochs_completed)
            torch.save(self.online_model.state_dict(), online_model_name)
    
    def step_run(self):
        state = self.env.initialize_state(self.start_day)
        while(True):
            battery = state[0] - state[50] + state[51+self.Tf]
            if battery > self.battery_max: battery = self.battery_max

            if len(self.replay_buffer) > self.buffer_min:
                if self.step_count % self.model_update_period == 0:
                    self.update_models()
                if self.step_count % 6 == 0:
                    self.rollout()
                    if len(self.virt_buffer) > self.start_size:
                        for _ in range(20):
                            self.optimize_model()

            # online 신경망 행동
            action = self.interaction_step(torch.from_numpy(state).float(), battery, self.time, self.epochs_completed % self.days)
            
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
            
            self.step_count += 1
            yield battery

    def rollout(self, idxs=None):
        if idxs is None:
            idxs = np.random.choice(len(self.replay_buffer), size=self.rollout_batch_size)
        buffer = ReplayBuffer_vi(self.rollout_batch_size * self.horizon, batch_size=self.batch_size)
        states = np.vstack(self.replay_buffer.ss_mem[idxs]).astype(np.float32)
        ep = self.replay_buffer.day_mem[idxs].astype(np.int64)
        ob_time = np.argmax(states[:, 2:26], axis=1)

        for t in range(1, self.horizon+1):
            battery = states[:, 0] - states[:, 50] + states[:, 51]
            battery = np.clip(battery, a_min=None, a_max=self.battery_max)
        
            with torch.no_grad():
                actions = self.interaction_step(torch.from_numpy(states).float(), battery, ob_time, ep, roll=True) # roll=True
            
            next_states, rewards = self.T_hat.transition(states, actions, ob_time, ep, 
                                                            battery, t)

            violations = self.env.check_violation(next_states, actions, ob_time)
            
            for s,a,r,s_,vi, d in zip(states, actions, rewards, next_states, violations, ep):
                buffer.put((s, a, r, s_, vi))
                self.virt_buffer.put((s, a, r, s_, vi, d))
            
            over_hour = ob_time + 1 > 23
            ob_time[over_hour] = 0
            ep[over_hour] += 1
            over_day = ep >= self.days
            continues = ~(violations | over_day)

            if continues.sum() == 0:
                break
            states = next_states[continues]
            battery = battery[continues]
            ob_time = ob_time[continues] + 1
            ep = ep[continues]
            
        return buffer
    
    def update_models(self, steps=10):
        # buffer_reward = np.random.choice(self.replay_buffer.rs_mem[:self.replay_buffer.size], size=self.replay_buffer.size)
        buffer_reward = self.replay_buffer.rs_mem[:self.replay_buffer.size]
        r_min = buffer_reward.min()
        r_max = buffer_reward.max()
        # if (self.replay_buffer.size > 0) & (self.virt_buffer.size > 0):
        #     real_violoation = self.replay_buffer.vi_mem[:self.replay_buffer.size]

        #     H_step_violation = self.virt_buffer.vi_mem[:self.virt_buffer.size]
        #     full_violation = np.r_[real_violoation, H_step_violation]

        #     p_new = real_violoation.sum() / len(real_violoation)
        #     q_new = H_step_violation.sum() / len(H_step_violation)

        #     self.p, self.q = (p_new, q_new) if p_new > q_new else (self.p, self.q)
        self.update_r_bounds(r_min, r_max) 
    
    def update_r_bounds(self, r_min, r_max):
        self.r_min, self.r_max = r_min, r_max
        alpha1 = ((1.-self.p)*r_max - (1. - self.q*self.gamma**self.horizon)*r_min) / (self.p - self.q * self.gamma**self.horizon)
        self.violation_cost = np.max([alpha1, 0])