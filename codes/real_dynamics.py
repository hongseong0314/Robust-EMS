import pandas as pd
import numpy as np

class T_hat():
    def __init__(self, 
                 load_data, 
                 generation_data, 
                 feature, 
                 summer_TOU, 
                 T, 
                 days, 
                 pD,
                ):
        self.load_data = load_data
        self.generation_data = generation_data
        self.feature = feature
        self.TOU = summer_TOU
        self.T = T
        self.days = days
        self.pD = pD
        
    def next_state(self, time, e, battery, charge, day_charge):
        s_ = np.zeros(self.feature)
        s_[0] = battery + charge
        
        if time==23: s_[1] = 0
        else: s_[1] = day_charge

        s_[2:26] = self.T[(time + 1) % 24]

        for i in range(24):
            s_[26+i] = self.TOU[(time+1+i)%24]

        s_[50] = self.load_data[(e)%self.days][(time+1)%24]
        s_[51] = self.generation_data[(e)%self.days][(time+1)%24]
        return s_
    
    def cal_price(self, time, charge, day_charge):
        """
        cost 계산
        """
        cost = charge * self.TOU[time]
        if time == 23: cost = cost + self.pD * max(day_charge)
        return cost
    
    def transition(self, state, charge, time, ep, battery):
        if state.ndim() == 2:
            day_max = state[:, 1]
            day_max = np.c_[day_max, charge]
            day_max = np.max(day_max, axis=1)
            
            reward_list = []
            next_state = []
            for c, t, b, e, de in zip(charge, time, battery, ep, day_max):
                reward = self.cal_price(t, c, de)
                next_s = self.next_state(t, e, b, c, de)
                
                reward_list.append(reward)
                next_state.append(next_s)
                
            return np.array(next_state), np.array(reward_list)
        else:
            day_max = state[1] if state[1] > charge else charge
            reward = self.cal_price(time, charge, day_max)
            next_s = self.next_state(time, ep, battery, charge, day_max)
            return next_s, reward