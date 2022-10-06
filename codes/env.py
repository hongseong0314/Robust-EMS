import numpy as np
import pickle

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
    
    def initialize_state(self, feature, T, load_data, generation_data, Tf, start_day, summer_TOU, winter_TOU):
        state = np.zeros(feature)
        if start_day % 365 < 90 or start_day % 365 >= 273: TOU = winter_TOU
        else: TOU = summer_TOU
        state[0] = 0    # battery level
        state[1] = 0    #demand charge
        state[2:2+24] = T[0][0:24]
        state[26:26+24] = TOU
        state[50:50+Tf+1] = load_data[0][0:Tf+1]
        state[51+Tf:51 + Tf + Tf+1] = generation_data[0][0:Tf+1]
        return state

    def step(self, n_epi, time, battery, charge, day_charge, TOU, Tf):
        next_s = self.next_state(n_epi, time, battery, charge, day_charge, TOU, Tf)
        reward = self.cal_price(time, charge, day_charge,TOU)
        return next_s, -reward


class Environment_minc():
    def __init__(self, 
                 feature,
                 load_data, 
                 generation_data,
                 days, 
                 T,
                 Tf, 
                 pD,
                 summer_TOU, winter_TOU,
                 battery_max,
                 ):
        self.feature = feature
        self.load_data = load_data
        self.generation_data = generation_data
        self.day = days
        self.T = T 
        self.pD = pD
        self.Tf = Tf
        self.summer_TOU = summer_TOU
        self.winter_TOU = winter_TOU
        self.battery_max = battery_max

        # self.limit_c = 0
        # self.e = 0
        # # self.update_limit(winter_TOU)
        self.market_limit = pickle.load(open("market_sample.pkl", 'rb'))

    def cal_price(self, time, charge, day_charge, TOU):
        """
        cost 계산
        """
        cost = charge * TOU[time]
        if time == 23: cost = cost + self.pD * max(day_charge)
        return cost

    def initialize_state(self, start_day):
        Tf = self.Tf
        T = self.T
        state = np.zeros(self.feature)
        if start_day % 365 < 90 or start_day % 365 >= 273: TOU = self.winter_TOU
        else: TOU = self.summer_TOU
        state[0] = 0    # battery level
        state[1] = 0    #demand charge
        state[2:2+24] = T[0][0:24]
        state[26:26+24] = TOU
        state[50:50+Tf+1] = self.load_data[0][0:Tf+1]
        state[51+Tf:51 + Tf + Tf+1] = self.generation_data[0][0:Tf+1]
        return state

    def next_state(self, n_epi,time,battery,charge,day_charge,TOU):
        """
        다음 state 계산
        """
        state_prime = np.zeros(self.feature)
        state_prime[0] = battery + charge

        load_data = self.load_data
        generation_data = self.generation_data
        days = self.day
        T = self.T
        Tf = self.Tf
        
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
    

    def step(self, n_epi, time, battery, charge, day_charge, TOU):
        next_s = self.next_state(n_epi, time, battery, charge, day_charge, TOU)
        reward = self.cal_price(time, charge, day_charge, TOU)

        # if self.e < n_epi:
        #     self.update_limit(TOU)

        violation = self.check_violation(next_s, charge, time)
        return next_s, reward, violation


    # def update_limit(self, TOU):
    #     # day_load = self.load_data[self.limit_c % self.day]
    #     sd = np.log(TOU) # day_load
    #     market_limit = [np.random.uniform(20-sd[i], 20, 1) for i in range(24)]
    #     # market_limit = [40 - time if time >= 20 else time for time in market_limit]
    #     self.limit_c += 1

    #     self.market_limit = np.vstack(market_limit)
        
    #     pass

    def check_violation(self, next_state, charge, time):
        if next_state.ndim == 2:
            vio_list = []
            for s, a, t in zip(next_state, charge, time):
                #distbattery = s[0] + s[51] - s[50]
                vio_list.append(s[0] <= 0)#s[0] > self.battery_max
            return np.array(vio_list)
        else:
            battery = next_state[0]

            # t+1 배터리 + 생산 배터리 - load 량
            #distbattery = battery + next_state[51] - next_state[50]

            # market 제한
            #charge > self.market_limit[time][0] 
            return battery <= 0 #or next_state[0] > self.battery_max