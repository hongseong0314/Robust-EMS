import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange
from src.dynamics import BatchedGaussianEnsemble

class T_hat():
    def __init__(self, 
                 load_data, 
                 generation_data, 
                 feature, 
                 summer_TOU, 
                 winter_TOU,
                 T, 
                 days, 
                 pD,
                 obs,
                ):
        self.load_data = load_data
        self.generation_data = generation_data
        self.feature = feature
        self.TOU = winter_TOU
        self.T = T
        self.days = days
        self.pD = pD
        self.obs = obs
        
    def next_state(self, time, e, battery, charge, day_charge, iter):
        s_ = np.zeros(self.feature)
        s_[0] = battery + charge
        
        if time==23: s_[1] = 0
        else: s_[1] = day_charge

        s_[2:26] = self.T[(time + 1) % 24]

        for i in range(24):
            s_[26+i] = self.TOU[(time+1+i)%24]

        s_[50] = self.load_data[(e)%self.days][(time+1)%24]
        s_[51] = self.generation_data[(e)%self.days][(time+1)%24]
        
        if self.obs < iter:
            # pr = iter - self.obs + 1
            sd = (1.31)**(iter-1) - 1
            s_[50] = np.clip(np.int(np.round(np.random.normal(s_[50], sd, 1))), 0, None)
            s_[51] = np.clip(np.int(np.round(np.random.normal(s_[51], sd, 1))), 0, None)
        return s_
    
    def cal_price(self, time, charge, day_charge):
        """
        cost 계산
        """
        cost = charge * self.TOU[time]
        if time == 23: cost = cost + self.pD * day_charge
        return cost
    
    def transition(self, state, charge, time, ep, battery, iter):
        if state.ndim == 2:
            day_max = state[:, 1]
            day_max = np.c_[day_max, charge]
            day_max = np.max(day_max, axis=1)
            
            reward_list = []
            next_state = []
            for c, t, b, e, de in zip(charge, time, battery, ep, day_max):
                reward = self.cal_price(t, c, de)
                next_s = self.next_state(t, e, b, c, de, iter)
                
                reward_list.append(reward)
                next_state.append(next_s)
                
            return np.array(next_state), np.array(reward_list)
        else:
            day_max = state[1] if state[1] > charge else charge
            reward = self.cal_price(time, charge, day_max)
            next_s = self.next_state(time, ep, battery, charge, day_max, iter)
            return next_s, reward


class T_hat_Gaussian():
    def __init__(self, 
                 load_data, 
                 generation_data, 
                 feature, 
                 summer_TOU, 
                 winter_TOU,
                 T, 
                 days, 
                 pD,
                 start_day,
                 end_day,
                ):
        self.load_data = load_data
        self.generation_data = generation_data
        self.feature = feature
        self.summer_TOU = summer_TOU
        self.winter_TOU = winter_TOU
        self.T = T
        self.days = days
        self.pD = pD
        self.start_day = start_day
        self.end_day = end_day
        self.model_cfg = BatchedGaussianEnsemble.Config()
    
    def setup(self):
        self.model_ensemble = BatchedGaussianEnsemble(self.model_cfg, self.feature, 20)

    def next_state(self, s_, time, e, battery, charge, day_charge):
        s_[0] = battery + charge
        
        if time==23: s_[1] = 0
        else: s_[1] = day_charge

        s_[2:26] = self.T[(time + 1) % 24]

        for i in range(24):
            s_[26+i] = self.TOU[(time+1+i)%24]

        # s_[50] = self.load_data[(e)%self.days][(time+1)%24]
        # s_[51] = self.generation_data[(e)%self.days][(time+1)%24]
        return s_
    
    def cal_price(self, time, charge, day_charge):
        """
        cost 계산
        """
        cost = charge * self.TOU[time]
        if time == 23: cost = cost + self.pD * max(day_charge)
        return cost
    
    def transition(self, state, charge, time, ep, battery):
        one_hot_a = self.onehot_charge(charge)
        next_s, _ = self.model_ensemble.sample(torch.from_numpy(state), one_hot_a)
        next_s = next_s.detach().cpu().numpy()
        if state.ndim() == 2:
            day_max = state[:, 1]
            day_max = np.c_[day_max, charge]
            day_max = np.max(day_max, axis=1)
            
            reward_list = []
            next_state = []

            for s_, c, t, b, e, de in zip(next_s, charge, time, battery, ep, day_max):
                self.TOU_cal(e)
                reward = self.cal_price(t, c, de)
                next_s = self.next_state(s_, t, e, b, c, de)
                
                reward_list.append(reward)
                next_state.append(next_s)
                
            return np.array(next_state), np.array(reward_list)
        else:
            self.TOU_cal(ep)
            day_max = state[1] if state[1] > charge else charge
            reward = self.cal_price(time, charge, day_max)
            next_s = self.next_state(next_s, time, ep, battery, charge, day_max)
            return next_s, reward
    
    def TOU_cal(self, epoch):
        if (self.start_day + epoch % self.end_day) % 365 < 90 or (self.start_day + epoch % self.end_day) % 365 >= 273:
            self.TOU = self.winter_TOU
        else: 
            self.TOU = self.summer_TOU
    
    def onehot_charge(self, charge):
        return F.one_hot(torch.from_numpy(charge), num_classes=20)

    def fit(self, replay_buffer, steps=10, progress_bar=False):
        n = replay_buffer.size
        
        states, actions, rewards, next_states, _ = replay_buffer.sample(n)
        actions = self.onehot_charge(actions)
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
        
