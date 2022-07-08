import collections
import random
import numpy as np
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import pandas as pd

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

class ReplayBuffer_list():
    """
    경험 재현 버퍼 list 버전
    """
    def __init__(self, 
                 buffer_limit=10000, 
                 batch_size=64):
        self.ss_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.as_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.rs_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        self.ps_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)
        #self.ds_mem = np.empty(shape=(buffer_limit), dtype=np.ndarray)

        self.max_size = buffer_limit
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0
    
    def put(self, sample):
        s, a, r, p = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        #self.ds_mem[self._idx] = d
        
        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size

        idxs = np.random.choice(
            self.size, batch_size, replace=False)

        s_lst = torch.tensor(np.vstack(self.ss_mem[idxs]),dtype=torch.float32)
        a_lst = torch.tensor(np.vstack(self.as_mem[idxs]),dtype=torch.int64)
        r_lst = torch.tensor(np.vstack(self.rs_mem[idxs]),dtype=torch.int64)    
        s_prime_lst = torch.tensor(np.vstack(self.ps_mem[idxs]),dtype=torch.float32)

        experiences = s_lst, \
                      a_lst, \
                      r_lst, \
                      s_prime_lst, \
                      
        return experiences

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer():
    """
    경험 재현 버퍼 우선순위 방식
    """
    def __init__(self, 
                 buffer_limit=15000, 
                 batch_size=64, 
                 rank_based=False,
                 alpha=0.6, 
                 beta0=0.1, 
                 beta_rate=0.99992):
        self.buffer_limit = buffer_limit
        self.memory = np.empty(shape=(self.buffer_limit, 2), dtype=np.ndarray)
        self.batch_size = batch_size
        self.n_entries = 0
        self.next_index = 0
        self.td_error_index = 0
        self.sample_index = 1
        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate

    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based:
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def put(self, sample):
        priority = 1.0
        if self.n_entries > 0:
            priority = self.memory[
                :self.n_entries, 
                self.td_error_index].max()
        self.memory[self.next_index, 
                    self.td_error_index] = priority
        self.memory[self.next_index, 
                    self.sample_index] = np.array(sample)
        self.n_entries = min(self.n_entries + 1, self.buffer_limit)
        self.next_index += 1
        self.next_index = self.next_index % self.buffer_limit

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        return self.beta

    def sample(self, batch_size=None):
        # beta 조절 후 0으로 채워진 행 삭제
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        # rank 기반 순위 결정
        if self.rank_based:
            priorities = 1/(np.arange(self.n_entries) + 1)
        else: # proportional
            priorities = entries[:, self.td_error_index] + 1e-6
        scaled_priorities = priorities**self.alpha       

        # 순위 확률화 
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64)

        # 중요도 및 정규화 계산
        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        
        # 샘플링
        samples = np.array([entries[idx] for idx in idxs])
        
        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = torch.tensor(np.vstack(normalized_weights[idxs]))

        samples_stacks[0] = torch.tensor(samples_stacks[0],dtype=torch.float32)
        samples_stacks[1] = torch.tensor(samples_stacks[1], dtype=torch.int64)
        samples_stacks[2] = torch.tensor(samples_stacks[2], dtype=torch.int64)    
        samples_stacks[3] = torch.tensor(samples_stacks[3], dtype=torch.float32)

        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return str(self.memory[:self.n_entries])
    
    def __str__(self):
        return str(self.memory[:self.n_entries])




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
    # 환경과의 상호작용? -> DQN은 버퍼를 사용
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

def avg_cost_plot(cost_history, iteration, days, epochs, title):
    """
    시간 평균 그래프
    """
    temp = 0
    check_dqn = list()
    check_greedy = list()
    for i in range(0,iteration):
        temp = temp + sum(cost_history[i*days : i*days+days])
        temp2 = (temp/(i+1))
        check_dqn.append(temp2)
    with plt.style.context('ggplot'):
        plt.figure(figsize=(15,10))
        x = range(0, int(epochs / days))
        y1 = [check_dqn[v] for v in x] #dqn
        plt.plot(x,y1,label='DQL time avgerage cost', color='r')
        plt.plot([0, x[-1]], [y1[-1], y1[-1]], "b:")
        plt.plot(x[-1], y1[-1], "bo")
        plt.text(x[-1]-0.04*x[-1], y1[-1]+0.04*y1[-1],"{}".format(y1[-1]))
        plt.title("DQL? Time Avegerage Cost / {}".format(title), fontsize=15)
        plt.xlabel('Epochs',fontsize=22)
        plt.ylabel('Cost(won)',fontsize=22)
        plt.legend()
        name_ = title.split("_")[0]
        print("{}_price : {}".format(name_, sum(cost_history[epochs-days:epochs])))
        print("{}_ave : {}".format(name_, check_dqn[int(epochs/days)-1]))
        # plt.show()
        plt.savefig("{}.png".format(title))


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


def name_check(coder):
    """
    model 이름 체크
    """
    if coder in ['DQN', 'dqn', 'DDQN', 'ddqn', 'DualingDQN', 'dualingdqn', 'DualingDDQN', 'dualingDDQN', 'PER', 'AC']:
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

def charge_graph(load_data, generation_data, TOU, action_history, battery_history, path):
    fig, ax = plt.subplots(2, 2, figsize=(2*7, 2*5))
    for iters, i in enumerate(range(4)):
        row = iters // 2
        col = iters % 2
        x = range(0, 24)
        y1 = [v for v in load_data[i]]
        y2 = [v for v in action_history[i]]
        y3 = [v for v in TOU[0:24]]
        y4 = [v for v in battery_history[i]]
        y5 = [v for v in generation_data[i]]

        ax[row, col].plot(x, y3, label='TOU', color='gray')
        ax[row, col].fill_between(x[0:24], y3[0:24], color='lightgray')

        #plt.plot(x, y4, linewidth=3, label='HM_charge', color='Red')
        ax[row, col].plot(x, y2, linewidth=3 ,label='DQL', color='Orange')
        #plt.plot(x, y5, linewidth=3 ,label='Optimal', color='orange')

        ax[row, col].plot(x, y5, '--',label='Generation',color='gray')
        ax[row, col].plot(x, y1,'-', label='Load', color='black')

        ax[row, col].set(title='{} Day sample'.format(i+1),
                         ylabel='Charge', xlabel='Time',
                        xticks=np.arange(0, 24), yticks=[0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40]) 
        ax[row, col].legend(loc='best')
        ax[row, col].grid(True)

    fig.suptitle('_'.join(path.split("_")[1:3]),fontweight ="bold")
    fig.tight_layout()
    plt.show()