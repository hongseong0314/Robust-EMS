from config import config
import warnings
import os
from itertools import product
from codes.strategy import EGreedyStrategy, EGreedyLinearStrategy
from codes.trainer import trainer
from codes.utills import name_check
warnings.filterwarnings('ignore')
# os.environ['KMP_DUPLICATE_LIB_OK']='True'

eg = lambda : EGreedyStrategy(epsilon=0.001)

egl = lambda: EGreedyLinearStrategy(init_epsilon=1.0,
                                    min_epsilon=0.001, 
                                    decay_steps=15000*24)

SEEDS = [12] # 12, 34, 53, 44, 34, 53
def run():
    st_list = [egl] # eg
    batch_list = [32]
    model_name = ["REMS"] # "DQN", "PER", "DDQN", "DualingDDQN", "REMS", "LRDQN"
 
    for seed in SEEDS:
        # hyper search 
        for sub_set in product(*[batch_list, st_list, model_name]): 
            cfg = config()
            batch, train_st,  model_= sub_set
            print(f"Model : {model_}, batch : {batch}, train_st : {train_st().__class__.__name__}")
            if name_check(model_):
                return print("모델 이름 이상함")
            cfg.seed = seed
            cfg.coder = model_
            cfg.model.batch_size= batch
            cfg.model.training_strategy_fn = train_st
            if cfg.coder == 'REMS':
                cfg.model.p=0.9
                cfg.model.q=0.09
                cfg.model.H=5
                cfg.model.obs=1
            # 모델을 학습
            trainer(cfg)

if __name__ == '__main__':
    print("train start.........")
    run()