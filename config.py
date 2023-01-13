from ml_collections import config_dict
import pandas as pd
import numpy as np
from codes.utills import load_and_generte

def config():
    cfg = config_dict.ConfigDict()
    cfg.EMS = config_dict.ConfigDict()
    
    f_load = pd.read_csv('1년치 소비데이터.csv')
    f_generation = pd.read_csv("태양광데이터1.csv",encoding='cp949')
    pos_name = '영암에프원태양광b' # 사용할 발전기명

    cfg.EMS.start_day = 0 
    cfg.EMS.end_day = 30 
    cfg.EMS.T = np.identity(n=24, dtype=np.uint8)

    cfg.EMS.winter_TOU = [5,5,5, 5,5,5, 5,15,15, 15,25,10, 10,10,10, 10,10,15, 15,5,5, 5,5,5]  
    cfg.EMS.summer_TOU = [5,5,5, 5,5,5, 5,10,10 ,10,10,15, 15,15,15, 15,15,10, 10,5,5, 5,5,5]

    cfg.EMS.battery_max = 40
    cfg.EMS.pD = 30
    cfg.EMS.days = cfg.EMS.end_day - cfg.EMS.start_day
    cfg.EMS.load_data, cfg.EMS.generation_data = load_and_generte(f_load, 
                                                                f_generation, 
                                                                pos_name, 
                                                                cfg.EMS.start_day, 
                                                                cfg.EMS.end_day)

    cfg.model = config_dict.ConfigDict()
    cfg.model.value_optimizer_lr = 0.0001
    cfg.model.gamma = 0.95
    cfg.model.Tf= 0
    cfg.model.nS = 52 + 2*cfg.model.Tf
    cfg.model.freq = 15
    cfg.model.epochs = 30000

    del f_load, f_generation, pos_name
    return cfg