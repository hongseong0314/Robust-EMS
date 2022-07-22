import numpy as np
import torch

class GreedyStrategy():
    """
    greedy 전략
    """
    def __init__(self):
        self.exploratory_action_taken = False

    def select_action(self, model, state, battery, battery_max):
        with torch.no_grad():
            q_values = model(state).cpu().detach()

        q_list ,idx = [], []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max:
                q_list.append(q_values[i].item())
                idx.append(i)
            else: q_list.append(np.float('inf'))
        
        return q_list.index(min(q_list))

class EGreedyStrategy():
    """
    입실론 그리디 전략
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state, battery, battery_max, market_limit, roll=False):
        #self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach()

        q_list ,idx = [], []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max :
                q_list.append(q_values[i].item())
                idx.append(i)
            else: q_list.append(np.float('inf')) #np.float('inf')
        # 입실론 그리디??탐색 
        if len(idx) == 0:
            action = 0

        else:
            if np.random.rand() >= self.epsilon:
                action = q_list.index(min(q_list))
            else: 
                action = np.random.choice(idx)

        #self.exploratory_action_taken = action != np.argmax(q_values)
        return action

class EGreedyStrategy_vi():
    """
    입실론 그리디 전략
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state, battery, battery_max):
        #self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach()

        q_list = [q_values[i].item() for i in range(0,21,1)]
        # for i in range(0,21,1):
        #     q_list.append(q_values[i].item())
        #     idx.append(i)
        # print("idx", idx)
        # 입실론 그리디??탐색 
        if np.random.rand() >= self.epsilon:
            action = q_list.index(min(q_list))
        else: 
            action = np.random.choice(range(0,21,1))

        #self.exploratory_action_taken = action != np.argmax(q_values)
        return action

class EGreedyStrategy_max():
    """
    입실론 그리디 전략
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.exploratory_action_taken = None

    def select_action(self, model, state, battery, battery_max):
        #self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach()

        q_list ,idx = [], []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max:
                q_list.append(q_values[i].item())
                idx.append(i)
            else: q_list.append(np.float('-inf')) #np.float('inf')
        # 입실론 그리디??탐색 
        if np.random.rand() >= self.epsilon:
            action = q_list.index(max(q_list))
        else: 
            if len(idx) == 0:
                action = np.random.choice(range(1, 21, 1))     
            else:
                action = np.random.choice(idx)

        #self.exploratory_action_taken = action != np.argmax(q_values)
        return action

class EGreedyLinearStrategy():
    """
    입실론 그리디 선형 감소 전략
    """
    def __init__(self, init_epsilon=1.0, min_epsilon=0.001, decay_steps=20000):
        self.t = 0
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.min_epsilon = min_epsilon
        self.decay_steps = decay_steps
        self.exploratory_action_taken = None
        
    def _epsilon_update(self):
        epsilon = 1 - self.t / self.decay_steps
        epsilon = (self.init_epsilon - self.min_epsilon) * epsilon + self.min_epsilon
        epsilon = np.clip(epsilon, self.min_epsilon, self.init_epsilon)
        self.t += 1
        return epsilon

    def select_action(self, model, state, battery, battery_max, market_limit, roll=False):
        #self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach()

        q_list ,idx = [], []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max:
                q_list.append(q_values[i].item())
                idx.append(i)
            else: q_list.append(np.float('inf'))

        if len(idx) == 0:
            action = 0
        else:
            if np.random.rand() >= self.epsilon:
                action = q_list.index(min(q_list))
            else: 
                action = np.random.choice(idx)

        if roll == False:
            self.epsilon = self._epsilon_update()
        #self.exploratory_action_taken = action != np.argmax(q_values)
        return action


class EGreedyExpStrategy():
    """
    입실론 그리디 exp 감소 전략
    """
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0
        self.exploratory_action_taken = None

    def _epsilon_update(self):
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        return self.epsilon

    def select_action(self, model, state, battery, battery_max, market_limit):
        #self.exploratory_action_taken = False
        with torch.no_grad():
            q_values = model(state).cpu().detach()

        q_list ,idx = [], []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max and i <= market_limit:
                q_list.append(q_values[i].item())
                idx.append(i)
            else: q_list.append(np.float('inf'))

        if np.random.rand() >= self.epsilon:
            action = q_list.index(min(q_list))
        else: 
            action = np.random.choice(idx)

        self._epsilon_update()
        #self.exploratory_action_taken = action != np.argmax(q_values)
        return action

class SoftMaxStrategy():
    """
    소프트맥스 전략
    """
    def __init__(self, 
                 init_temp=1.0, 
                 min_temp=0.01, 
                 exploration_ratio=0.8, 
                 max_steps=25000):
        self.t = 0
        self.init_temp = init_temp
        self.exploration_ratio = exploration_ratio
        self.min_temp = min_temp
        self.max_steps = max_steps
        self.exploratory_action_taken = None
        
    def _update_temp(self):
        temp = 1 - self.t / (self.max_steps * self.exploration_ratio)
        temp = (self.init_temp - self.min_temp) * temp + self.min_temp
        temp = np.clip(temp, self.min_temp, self.init_temp)
        self.t += 1
        return temp

    def select_action(self, model, state, battery, battery_max, market_limit):
        #self.exploratory_action_taken = False
        temp = self._update_temp()

        with torch.no_grad():
            q_values = model(state)
        
        idx = []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max and i <= market_limit:
                idx.append(i)
    
        q_values = q_values[idx]
        scaled_qs = q_values/temp
        norm_qs = scaled_qs - scaled_qs.max()            
        e = np.exp(norm_qs)
        probs = e / np.sum(e)
        assert np.isclose(probs.sum(), 1.0)

        action = np.random.choice(np.arange(len(probs)), size=1, p=probs)[0]
        #self.exploratory_action_taken = action != np.argmax(q_values)
        return action