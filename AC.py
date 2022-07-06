import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os

from utills import over_flow_battery

class FCDAP(nn.Module):
    def __init__(self, 
                 feature):
        super(FCDAP, self).__init__()

        self.fc1 = torch.nn.Linear(feature, 512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.relu = torch.nn.ReLU()
        
        self.output_layer = torch.nn.Linear(512, 21) # A(s, a)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        a = self.output_layer(x)
        return a

    def full_pass(self, state, battery, battery_max):
        logits = self.forward(state)
        idx = []
        for i in range(0,21,1):
            if battery + i>= 0 and battery + i <= battery_max:
                idx.append(i)
        #print(f"battery : {battery}")
        #print(f"a : {logits[idx]}")
        
        logits = logits[idx]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy

    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()
    
    def select_greedy_action(self, state):
        logits = self.forward(state)
        return np.argmax(logits.detach().numpy())

class FCV(nn.Module):
    def __init__(self, 
                 feature):
        super(FCV, self).__init__()
        self.fc1 = torch.nn.Linear(feature, 512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.relu = torch.nn.ReLU()
        
        self.output_layer = torch.nn.Linear(512, 1) # A(s, a)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))

        return self.output_layer(x)

class VPG():
    def __init__(self, 
                 policy_model_fn, 
                 policy_model_max_grad_norm, 
                 policy_optimizer_fn, 
                 policy_optimizer_lr,
                 value_model_fn, 
                 value_model_max_grad_norm, 
                 value_optimizer_fn, 
                 value_optimizer_lr, 
                 entropy_loss_weight,
                 battery_max,
                 ):

        self.policy_model_fn = policy_model_fn
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.value_model_fn = value_model_fn
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        
        self.entropy_loss_weight = entropy_loss_weight

        self.battery_max = battery_max

    def optimize_model(self):
        T = len(self.rewards)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * self.rewards[t:]) for t in range(T)])
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)

        self.logpas = torch.cat(self.logpas)
        self.entropies = torch.cat(self.entropies) 
        self.values = torch.cat(self.values)

        value_error = (-returns - self.values)
        policy_loss = -(discounts * value_error.detach() * self.logpas).mean()
        entropy_loss = -self.entropies.mean()
        loss = (policy_loss + self.entropy_loss_weight * entropy_loss)
        self.policy_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        self.policy_optimizer.step()

        # 
        value_loss = value_error.pow(2).mul(0.5).mean()
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        self.value_optimizer.step()
        
    def interaction_step(self, state, env, n_epi, time, battery, TOU, Tf):
        action, is_exploratory, logpa, entropy = self.policy_model.full_pass(state, battery, self.battery_max)

        self.day_action.append(action)
        self.day_battery.append(state[0])

        new_state, reward = env.step(n_epi=n_epi, time=time, battery=battery, charge=action, 
                                        day_charge=self.day_action, TOU=TOU, Tf=Tf)
        
        self.cum_cost = self.cum_cost + reward
        self.logpas.append(logpa)
        self.entropies.append(entropy)
        self.rewards.append(reward)
        self.values.append(self.value_model(state))

        return new_state

    def train(self, make_env_fn, epochs, gamma,feature, 
              load_data, generation_data, pD, winter_TOU,
              summer_TOU, start_day, end_day, days, T, Tf):

        self.make_env_fn = make_env_fn
        self.gamma = gamma
        
        env = self.make_env_fn(feature, 
                                load_data, 
                                generation_data,
                                days,
                                T,
                                pD,)
    

        self.policy_model = self.policy_model_fn(feature)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model, 
                                                         self.policy_optimizer_lr)
        
        self.value_model = self.value_model_fn(feature)
        self.value_optimizer = self.value_optimizer_fn(self.value_model, 
                                                       self.value_optimizer_lr)
       # training_time = 0
        cost_history, battery_history, action_history = [],[],[]
        
        state = env.initialize_state(feature, T, load_data, generation_data, Tf, start_day, summer_TOU, winter_TOU)

        # init 배터리
        battery = 0

        with tqdm(range(epochs), unit="Run") as runing_bar:
            for n_epi in runing_bar:

                # collect rollout
                self.logpas, self.entropies, self.rewards, self.values = [], [], [], []
                
                self.cum_cost, self.day_action, self.day_battery = 0, [] , []
                # if (start_day + n_epi % end_day) % 365 < 90 or (start_day + n_epi % end_day) % 365 >= 273:TOU = winter_TOU
                # else: TOU = summer_TOU
                TOU = summer_TOU
                
                if n_epi % (end_day-start_day) == 0: state[0] = 0
                
                # 에피소드 step -> 24시간
                for time in range(24):
                    battery = state[0] - state[50] + state[51+Tf]
                    if battery > self.battery_max: battery = self.battery_max  # generation 초과 제한
                    # next state
                    state = self.interaction_step(torch.from_numpy(state).float(), env, n_epi, time, battery, TOU, Tf)
                    #print(state)
                
                
                next_value = self.value_model(torch.from_numpy(state).float()).detach().item()
                self.rewards.append(next_value)
                self.optimize_model()
                
                cost_history.append(self.cum_cost)
                action_history.append(self.day_action)
                battery_history.append(self.day_battery)
            
            
                if n_epi > days and n_epi % 1000 == 0:
                    runing_bar.set_postfix(cost=-sum(cost_history[n_epi-days:n_epi]))
        return cost_history, action_history, battery_history