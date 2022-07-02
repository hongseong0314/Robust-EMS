import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from train import trainer
from utills import load_and_generte

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# setup
f_load = pd.read_csv('1년치 소비데이터.csv')
# f_load= (f_load['total']/100).astype("int")

f_generation = pd.read_csv("태양광데이터1.csv",encoding='cp949')
# f_generation = f_generation[0:365]

start_day = 0 # 시작
end_day = 30 #int(365 * 0.8) # 80% train 사용
pos_name = '영암에프원태양광b' # 사용할 발전기명

T = np.identity(n=24, dtype=np.uint8) # 시간 인코딩
# T = np.array(pd.get_dummies(np.array([0,1,2,3,4,5,6,7,8,9,10,11,
#                                       12,13,14,15,16,17,18,19,20,21,22,23])))

# 여름 겨울 단위 전기료 가격?
winter_TOU = [5,5,5, 5,5,5, 5,15,15, 15,25,10, 10,10,10, 10,10,15, 15,5,5, 5,5,5]  # 겨울
summer_TOU = [5,5,5, 5,5,5, 5,10,10 ,10,10,15, 15,15,15, 15,15,10, 10,5,5,5,5,5]  #여름

# parameters
start_size = 3650
learning_rate = 0.0001
gamma = 0.98 # discount rate
Tf= 0 #미래 지수?
feature = 52 + 2*Tf # 52 + Tf*2
battery_max = 40 # 배터리 용량 
freq = 20 # target 모델 업데이트 폭
iteration = 50 # 반복
pD=30 # ??

# 탐색 전략
from strategy import GreedyStrategy, EGreedyStrategy, EGreedyLinearStrategy, EGreedyExpStrategy,SoftMaxStrategy
from utills import ReplayBuffer, ReplayBuffer_list, name_check

# 입실론 그리디
aa = lambda : EGreedyStrategy(epsilon=0.01)

# 입실론 그리디 선형 감소
bb = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
                                    min_epsilon=0.001, 
                                    decay_steps=20000)

# 입실론 그리디 exp 감소
cc = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                min_epsilon=0.001, 
                                decay_steps=20000)

# 입실론 그리디 softmax전략
dd = lambda: SoftMaxStrategy(init_temp=1.0, 
                            min_temp=0.1, 
                            exploration_ratio=0.8, 
                            max_steps=20000)
 
# 학습 설정값
config = {
    'f_load':f_load, 
    'f_generation':f_generation,
    'start_day': start_day,
    'end_day': end_day,
    'pos_name':pos_name,
    'iteration': iteration,
    'feature' : feature,
    'start_size': start_size,
    'summer_TOU':summer_TOU,
    'winter_TOU':winter_TOU,
    'battery_max':battery_max,
    'learning_rate':learning_rate,
    'T':T, 
    'Tf':Tf, 
    'pD':pD, 
    'gamma':gamma,
    'freq':freq,
    'coder':'AC',
    'verbose':True,
}

st_list = [bb] # bb, cc
batch_list = [128]
model_name = ['AC']

def run(cfg_dict):
    # hyper search 
    for sub_set in product(*[batch_list, st_list, model_name]):
        cfg = cfg_dict.copy()
        batch, train_st,  model_= sub_set
        replay_buffer_fn = lambda : ReplayBuffer_list(buffer_limit=15000, batch_size=batch) #7300
        cfg['batch_size'] = batch
        cfg['sample_action'] = train_st
        cfg['buffer'] = replay_buffer_fn
        cfg['coder'] = model_

        if name_check(cfg_dict['coder']):
            return print("모델 이름 이상함")
        
        # 모델을 학습
        trainer(**cfg)

if __name__ == '__main__':
    print("train start.........")
    run(config)






