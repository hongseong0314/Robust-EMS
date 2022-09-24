import collections
import random
import numpy as np
import torch.nn.functional as F
import torch
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

class ReplayBuffer():
    """
    경험재현 버퍼 큐 방식
    """
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst = [], [], [], []
        for transition in mini_batch:
            s, a, r, s_prime = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            
        s_lst = torch.tensor(np.array(s_lst),dtype=torch.float32)
        a_lst = torch.tensor(np.array(a_lst),dtype=torch.int64)
        r_lst = torch.tensor(np.array(r_lst),dtype=torch.int64)    
        s_prime_lst = torch.tensor(np.array(s_prime_lst),dtype=torch.float32)
        
        return s_lst, a_lst, r_lst, s_prime_lst

    def size(self):
        return len(self.buffer)


def sample_action(model, s, epsilon, battery, battery_max):
    with torch.no_grad():
        out = model(s)
    q_list ,idx = [], []
    for i in range(0,21,1):
        if battery + i>= 0 and battery + i <= battery_max:
            q_list.append(out[i].item())
            idx.append(i)
        else: q_list.append(np.float('inf'))

    # 입실론 그리디??탐색 
    coin = random.random()
    if (coin < epsilon): return random.choice(idx)
    else: return q_list.index(min(q_list))

def train(memory, q_target, q, optimizer, batch_size, battery_max, gamma, Tf):
    s, a, r, s_prime = memory.sample(batch_size)
    # 탐색 신경망
    q_a = q(s).gather(1,a)
    
    # target 신경망
    min_q_prime = q_target(s_prime).detach()
    Q_target = over_flow_battery(batch_size, battery_max, s_prime, min_q_prime, Tf).min(1)[0]

    Q_target = torch.tensor((Q_target)).resize(batch_size,1)
    # r + discount * max(q)
    target = r + gamma * Q_target
    loss = F.smooth_l1_loss(q_a, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def initialize_state(feature, T, load_data, generation_data, Tf, start_day, summer_TOU, winter_TOU):
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

def cal_price(time, charge, day_charge,TOU, pD):
    """
    cost 계산
    """
    cost = charge * TOU[time]
    if time == 23: cost = cost + pD * max(day_charge)
    return cost

def next_state(n_epi,time,battery,charge,day_charge,TOU,load_data,generation_data, feature, Tf, days, T):
    """
    다음 state 계산
    """
    state_prime = np.zeros(feature)
    state_prime[0] = battery + charge
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

def over_flow_battery(batch_size, battery_max, s_prime, min_q, Tf):
    """
    모델 output에 배터리 맥스 넘는지 체크
    """
    for i in range(batch_size):
        battery = s_prime[i][0].item() - s_prime[i][50].item() + s_prime[i][51+Tf].item()
        
        # battery 용량 넘으면 max로 고정
        if battery > battery_max: battery = battery_max

        for j in range(0, 21, 1):
            if battery + j >= 0 and battery + j <= battery_max:
                pass
            else:
                min_q[i, j] = np.float('inf')
    return min_q

def over_flow_battery_3000(batch_size, battery_max, s_prime, min_q, Tf):
    """
    모델 output에 배터리 맥스 넘는지 체크
    """
    for i in range(batch_size):
        battery = s_prime[i][0].item() - s_prime[i][50].item() + s_prime[i][51+Tf].item()
        
        # battery 용량 넘으면 max로 고정
        if battery > battery_max: battery = battery_max

        for j in range(0, 21, 1):
            if battery + j >= 0 and battery + j <= battery_max:
                pass
            else:
                min_q[i, j] = 3000.
    return torch.tensor(min_q)

def over_flow_battery_max(batch_size, battery_max, s_prime, min_q, Tf):
    """
    모델 output에 배터리 맥스 넘는지 체크
    """
    for i in range(batch_size):
        battery = s_prime[i][0].item() - s_prime[i][50].item() + s_prime[i][51+Tf].item()
        
        # battery 용량 넘으면 max로 고정
        if battery > battery_max: battery = battery_max

        for j in range(0, 21, 1):
            if battery + j >= 0 and battery + j <= battery_max:
                pass
            else:
                min_q[i, j] = np.float('-inf')
    return min_q

def target_scale(out, range=(500,1500)):
    return MinMaxScaler(feature_range=range).fit_transform(out)


def name_check(coder):
    """
    model 이름 체크
    """
    if coder in ['DQN', 'dqn', 'DDQN', 'ddqn', 'DualingDQN', 'dualingdqn', 'DualingDDQN', 'dualingDDQN', 'PER', 'AC', 'SDQN', 'ESDQN', 'LR-DQN']:
        return False
    else: return True


def load_and_generte(load_df, sun_df, pos, start_day, end_day):
    """
    load 데이터와 generate 데이터를 날자별로 합치고 발전기면으로 indexing하여 load와 gen 데이터를 추출
    """
    sun, load = sun_df.copy(), load_df.copy()

    sun['time'] = pd.to_datetime(sun['년월일'])
    sun.drop('년월일', axis=1, inplace=True)
    sun.set_index(sun['time'], inplace=True)

    load['total'] = (load['total']/100).astype("int")
    load['time'] = pd.to_datetime(load['timestamp'])
    load['hour'] = load['time'].dt.hour
    load['time'] = load['time'].apply(lambda x:x.strftime("%Y-%m-%d")) 
    load.drop('timestamp', axis=1,  inplace=True)
    load_pivot = load.pivot_table(index="time", columns="hour", values='total')
    load_pivot['time'] = pd.to_datetime(load_pivot.index)
    
    sun.reset_index(drop = True, inplace = True)
    load_pivot.reset_index(drop = True, inplace = True)
    
    Big_df = pd.merge(load_pivot, sun, left_on='time',right_on='time', how='left')
    sub_df = Big_df[Big_df['발전기명'] == pos].reset_index(drop=True)
    
    load_data = sub_df.iloc[start_day:end_day, :24].to_numpy()
    generation_data = sub_df.iloc[start_day:end_day, 27:].to_numpy() 
    generation_data = (generation_data / 2659200.).astype("int")
    del Big_df, sub_df, load_pivot
    
    return load_data, generation_data