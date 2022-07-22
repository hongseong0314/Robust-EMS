import pandas as pd
import numpy as np

def load_cfg():
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
    iteration = 5000 # 반복
    pD=30 # ??
    return {
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
            'violation':True,
            'verbose':True,
            }