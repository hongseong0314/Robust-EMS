import numpy as np
import pandas as pd
import torch
import os

def over_flow_battery(batch_size, battery_max, s_prime, min_q, Tf, day, time, market_limit, days):
    """
    모델 output에 배터리 맥스 넘는지 체크
    """
    B_t = s_prime[..., 0][..., np.newaxis].numpy()
    L_t = s_prime[..., 50][..., np.newaxis].numpy()
    G_t = s_prime[..., 51+Tf][..., np.newaxis].numpy()
    C = np.repeat([np.arange(21)],batch_size,axis=0)
    B_rate = 20

    con1 = np.where((-B_rate <= (C + G_t - L_t)) & ((C + G_t - L_t)<= B_rate), 1, np.float('inf'))
    con2 = np.where((0 <= (C + G_t - L_t + B_t)) & ((C + G_t - L_t + B_t) <= battery_max), 1, np.float('inf'))
    
    over_time = time == 23
    time[over_time] = 0
    day[over_time] = (day[over_time] + 1) % days
    market_mask = np.ones((batch_size,21))
    
    for i, (d, t) in enumerate(zip(day, time)):
        market_mask[i,int(market_limit[d][t][0])+1:] = np.float('inf')

    con = torch.from_numpy(con2 * con1 * market_mask).to(torch.float32)
    min_q *= con
    return torch.from_numpy(np.where(torch.isinf(min_q), float('inf'), min_q))

def name_check(coder):
    """
    model 이름 체크
    """
    if coder in ['DQN', 'dqn', 'DDQN', 'ddqn', 'DualingDQN', 'dualingdqn', 'DualingDDQN', 'dualingDDQN', 'PER', 'AC', 'REMS', 'LR-DQN', 'LRDQN']:
        return False
    else: 
        return True

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

def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)