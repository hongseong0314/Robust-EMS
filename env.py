import numpy as np

from utills import cal_price

class Environment():
    def __init__(self, feature,
                 load_data, 
                 generation_data,
                 days, 
                 T, pD):
        self.feature = feature
        self.load_data = load_data
        self.generation_data = generation_data
        self.day = days
        self.T = T 
        self.pD = pD

    def cal_price(self, time, charge, day_charge,TOU):
        """
        cost 계산
        """
        cost = charge * TOU[time]
        if time == 23: cost = cost + self.pD * max(day_charge)
        return cost

    def next_state(self, n_epi,time,battery,charge,day_charge,TOU, Tf):
        """
        다음 state 계산
        """
        state_prime = np.zeros(self.feature)
        state_prime[0] = battery + charge

        load_data = self.load_data
        generation_data = self.generation_data
        days = self.day
        T = self.T
        if time==23: state_prime[1] = 0
        else: state_prime[1] = max(day_charge)
        for i in range(24):
            state_prime[2 + i] = T[(time + 1) % 24][i]
            state_prime[26 + i] = TOU[(time+1+i)%24]
        for i in range(Tf+1):
            if time+1+i >= 24:
                state_prime[50+i] = load_data[(n_epi+1)%days][(time+1+i)%24]
                state_prime[51+Tf+i] = generation_data[(n_epi+1)%days][(time+1+i)%24]
            else:
                state_prime[50+i] = load_data[(n_epi)%days][(time+1+i)%24]
                state_prime[51+Tf+i] = generation_data[(n_epi)%days][(time+1+i)%24]

        return state_prime

    def step(self, n_epi, time, battery, charge, day_charge, TOU, Tf):
        next_s = self.next_state(n_epi, time, battery, charge, day_charge, TOU, Tf)
        reward = self.cal_price(time, charge, day_charge,TOU)
        return next_s, -reward
