import numpy as np
import pickle

class Environment_minc():
    def __init__(self, cfg):
        self.nS = cfg.model.nS
        self.load_data = cfg.EMS.load_data
        self.generation_data = cfg.EMS.generation_data
        self.days = cfg.EMS.days
        self.T = cfg.EMS.T 
        self.pD = cfg.EMS.pD
        self.Tf = cfg.model.Tf
        self.summer_TOU = cfg.EMS.summer_TOU
        self.winter_TOU = cfg.EMS.winter_TOU
        self.battery_max = cfg.EMS.battery_max
        self.market_limit = pickle.load(open("market_sample.pkl", 'rb'))

    def cal_price(self, time, charge, day_charge, TOU):
        """
        cost 계산
        """
        cost = charge * TOU[time]
        if time == 23: cost = cost + self.pD * max(day_charge)
        return cost

    def initialize_state(self, start_day):
        state = np.zeros(self.nS)
        if start_day % 365 < 90 or start_day % 365 >= 273: TOU = self.winter_TOU
        else: TOU = self.summer_TOU
        state[0] = 0  
        state[1] = 0   
        state[2:2+24] = self.T[0][:24]
        state[26:26+24] = TOU
        state[50:50+self.Tf+1] = self.load_data[0][0:self.Tf+1]
        state[51+self.Tf:51 + self.Tf + self.Tf+1] = self.generation_data[0][0:self.Tf+1]
        return state

    def next_state(self, n_epi, time, battery, charge, day_charge, TOU):
        """
        다음 state 계산
        """
        state_prime = np.zeros(self.nS)
        state_prime[0] = battery + charge
        
        if time==23: state_prime[1] = 0
        else: state_prime[1] = max(day_charge) 
        
        for i in range(24):
            state_prime[2 + i] = self.T[(time + 1) % 24][i]
            state_prime[26 + i] = TOU[(time+1+i)%24]
        
        for i in range(self.Tf+1):
            if time+1+i >= 24:
                state_prime[50+i] = self.load_data[(n_epi+1)%self.days][(time+1+i)%24]
                state_prime[51+self.Tf+i] = self.generation_data[(n_epi+1)%self.days][(time+1+i)%24]
            else:
                state_prime[50+i] = self.load_data[(n_epi)%self.days][(time+1+i)%24]
                state_prime[51+self.Tf+i] = self.generation_data[(n_epi)%self.days][(time+1+i)%24]

        return state_prime
    

    def step(self, n_epi, time, battery, charge, day_charge, TOU):
        next_s = self.next_state(n_epi, time, battery, charge, day_charge, TOU)
        reward = self.cal_price(time, charge, day_charge, TOU)
        violation = self.check_violation(next_s)
        return next_s, reward, violation

    def check_violation(self, next_state):
        if next_state.ndim == 2:
            vio_list = next_state[..., 0] <= 0
            return np.array(vio_list)
        else:
            return next_state[0] <= 0 