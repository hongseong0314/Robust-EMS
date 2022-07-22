import pandas as pd
import numpy as np
from itertools import product

from codes.train import trainer
from codes.strategy import GreedyStrategy, EGreedyStrategy, EGreedyLinearStrategy, EGreedyExpStrategy,SoftMaxStrategy
from codes.utills import name_check
from codes.buffer import ReplayBuffer_list, ReplayBuffer_vi, ReplayBuffer_vi_e
from config import load_cfg

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 입실론 그리디
aa = lambda : EGreedyStrategy(epsilon=0.01)

# 입실론 그리디 선형 감소
bb = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
                                    min_epsilon=0.001, 
                                    decay_steps=20000 * 24)

# 입실론 그리디 exp 감소
cc = lambda: EGreedyExpStrategy(init_epsilon=1.0,  
                                min_epsilon=0.001, 
                                decay_steps=20000)

# 입실론 그리디 softmax전략
dd = lambda: SoftMaxStrategy(init_temp=1.0, 
                            min_temp=0.1, 
                            exploration_ratio=0.8, 
                            max_steps=20000)
 

SEEDS = (12, 34, 53)
def run():
    # 학습 설정값
    cfg_dict = load_cfg()
    st_list = [aa] # bb, cc
    batch_list = [32]
    model_name = ["DQN", "DDQN", "DualingDDQN", "PER"]

    for seed in SEEDS:
        # hyper search 
        for sub_set in product(*[batch_list, st_list, model_name]):
            cfg = cfg_dict.copy()
            batch, train_st,  model_= sub_set
            replay_buffer_fn = lambda : ReplayBuffer_vi(buffer_limit=15000, batch_size=batch)
            cfg['batch_size'] = batch
            cfg['sample_action'] = train_st
            cfg['buffer'] = replay_buffer_fn
            cfg['coder'] = model_
            cfg['seed'] = seed

            if name_check(cfg['coder']):
                return print("모델 이름 이상함")
            
            # 모델을 학습
            trainer(**cfg)

if __name__ == '__main__':
    print("train start.........")
    run()






