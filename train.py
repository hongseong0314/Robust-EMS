from msilib.schema import Feature
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import pickle

from utills import sample_action, initialize_state, avg_cost_plot, load_and_generte, cal_price, next_state
from model import Qnet, DQN, DDQN, DualingQnet, DualingDDQN, PER
from env import Environment_minc

def trainer(f_load, 
            f_generation, 
            start_day, 
            end_day, 
            iteration, 
            feature, 
            start_size, 
            learning_rate, 
            batch_size, 
            pos_name,
            summer_TOU, 
            winter_TOU, 
            battery_max,
            T, 
            Tf, 
            pD, 
            gamma, 
            freq, 
            sample_action, 
            buffer, 
            coder,
            verbose=True):
    """
    f_load : 소비량 데이터프레임
    f_generation : 발전량 데이터프레임
    start_day : 학습에 사용할 시작 날
    end_day : 학습에 사용할 마지막 날
    iteration : 반복수
    feature : 사용할 특성 수
    start_size : online model 업데이트 시작
    learning_rate : 학습률
    batch_size : 배치 사이즈
    pos_name : 사용할 발전기 명
    summer_TOU : 여름 TOU
    winter_TOU : 겨울 TOU
    battery_max : 배터리 맥스
    T : 24시간 ont-hot matrix
    Tf : 미래 지수?
    pD : ?
    gamma : discount rate
    freq : target model 업데이트 횟수
    sample_action : 학습 전략
    buffer : 사용할 경험 버퍼
    coder : 모델 이름
    verbose : mean cost 시각화 유무
    """
    days = end_day-start_day

    # load, generation 데이터 형성
    load_data, generation_data = load_and_generte(f_load, f_generation, pos_name, start_day, end_day)

    # 반복 수
    epochs = days * iteration #무조건 days의 배수, 아니면 그래프 오류발생
    
    # DQN 모델 선택
    if coder == 'DQN':
        # 학습 함수들
        value_model_fn = lambda Feature: Qnet(Feature)
        value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        value_optimizer_lr = learning_rate
        training_strategy_fn = sample_action
        replay_buffer_fn = buffer
        
        env = Environment_minc(
                            feature,
                            load_data, 
                            generation_data,
                            days, 
                            T,
                            Tf, 
                            pD,
                            summer_TOU, winter_TOU
                            )

        agent = DQN(replay_buffer_fn=replay_buffer_fn,
                    env = env,
                    value_model_fn=value_model_fn,
                    value_optimizer_fn=value_optimizer_fn,
                    value_optimizer_lr=value_optimizer_lr,
                    training_strategy_fn=training_strategy_fn,
                    batch_size=batch_size,
                    gamma=gamma,
                    battery_max=battery_max,
                    feature=feature, 
                    epochs=epochs, 
                    start_day=start_day, 
                    end_day=end_day, 
                    summer_TOU=summer_TOU, 
                    winter_TOU=winter_TOU,
                    Tf=Tf,
                    pD=pD,
                    load_data=load_data, 
                    generation_data=generation_data, 
                    days=days,
                    T=T, 
                    start_size=start_size,
                    freq=freq
                    )

        # 학습 시작
        agent.setup()

        with tqdm(range(epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)
        
        
        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        
        # model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}.pth".format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        # 학습 결과 저장 이름 : result_modelname_stategy_batchsize
        history_path = 'history/result_{}_{}_{}.pkl'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, 
                                                            batch_size)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

        
        title = '{}_{}_{}'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)
    
    # DDQN 모델 선택
    if coder == 'DDQN':
        # 학습 함수들
        value_model_fn = lambda Feature: Qnet(Feature)
        value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        value_optimizer_lr = learning_rate
        training_strategy_fn = sample_action
        replay_buffer_fn = buffer
        
        env = Environment_minc(
                            feature,
                            load_data, 
                            generation_data,
                            days, 
                            T,
                            Tf, 
                            pD,
                            summer_TOU, winter_TOU
                            )

        agent = DDQN(replay_buffer_fn=replay_buffer_fn,
                    env = env,
                    value_model_fn=value_model_fn,
                    value_optimizer_fn=value_optimizer_fn,
                    value_optimizer_lr=value_optimizer_lr,
                    training_strategy_fn=training_strategy_fn,
                    batch_size=batch_size,
                    gamma=gamma,
                    battery_max=battery_max,
                    feature=feature, 
                    epochs=epochs, 
                    start_day=start_day, 
                    end_day=end_day, 
                    summer_TOU=summer_TOU, 
                    winter_TOU=winter_TOU,
                    Tf=Tf,
                    pD=pD,
                    load_data=load_data, 
                    generation_data=generation_data, 
                    days=days,
                    T=T, 
                    start_size=start_size,
                    freq=freq
                    )

        # 학습 시작
        agent.setup()

        with tqdm(range(epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)
        
        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        
        # model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}.pth".format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        # 학습 결과 저장 이름 : result_modelname_stategy_batchsize
        history_path = 'history/result_{}_{}_{}.pkl'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, 
                                                            batch_size)
    if coder == 'DualingDDQN':
        # 학습 함수들
        value_model_fn = lambda Feature: DualingQnet(Feature)
        value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        value_optimizer_lr = learning_rate
        training_strategy_fn = sample_action
        replay_buffer_fn = buffer
        
        env = Environment_minc(
                            feature,
                            load_data, 
                            generation_data,
                            days, 
                            T,
                            Tf, 
                            pD,
                            summer_TOU, winter_TOU
                            )

        tau = 0.1
        agent = DualingDDQN(replay_buffer_fn=replay_buffer_fn,
                    env = env,
                    value_model_fn=value_model_fn,
                    value_optimizer_fn=value_optimizer_fn,
                    value_optimizer_lr=value_optimizer_lr,
                    training_strategy_fn=training_strategy_fn,
                    batch_size=batch_size,
                    gamma=gamma,
                    battery_max=battery_max,
                    feature=feature, 
                    epochs=epochs, 
                    start_day=start_day, 
                    end_day=end_day, 
                    summer_TOU=summer_TOU, 
                    winter_TOU=winter_TOU,
                    Tf=Tf,
                    pD=pD,
                    load_data=load_data, 
                    generation_data=generation_data, 
                    days=days,
                    T=T, 
                    start_size=start_size,
                    freq=freq,
                    tau=tau 
                    )

        # 학습 시작
        agent.setup()

        with tqdm(range(epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)
        
        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        
        # model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}.pth".format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        # 학습 결과 저장 이름 : result_modelname_stategy_batchsize
        history_path = 'history/result_{}_{}_{}.pkl'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, 
                                                            batch_size)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

        title = '{}_{}_{}'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)

    # PER 모델 사용
    elif coder == 'PER':
        value_model_fn = lambda Feature: DualingQnet(Feature)
        value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        value_optimizer_lr = learning_rate
        training_strategy_fn = sample_action

        from utills import PrioritizedReplayBuffer
        replay_buffer_fn = lambda : PrioritizedReplayBuffer(buffer_limit=15000, batch_size=batch_size)

        env = Environment_minc(
                            feature,
                            load_data, 
                            generation_data,
                            days, 
                            T,
                            Tf, 
                            pD,
                            summer_TOU, winter_TOU
                            )

        tau = 0.1
        max_gradient_norm = float('inf')

        agent = PER(replay_buffer_fn=replay_buffer_fn,
                    env = env,
                    value_model_fn=value_model_fn,
                    value_optimizer_fn=value_optimizer_fn,
                    value_optimizer_lr=value_optimizer_lr,
                    max_gradient_norm = max_gradient_norm,
                    training_strategy_fn=training_strategy_fn,
                    batch_size=batch_size,
                    gamma=gamma,
                    battery_max=battery_max,
                    feature=feature,
                    epochs=epochs,
                    start_day=start_day,
                    end_day=end_day,
                    summer_TOU=summer_TOU,
                    winter_TOU=winter_TOU,
                    Tf=Tf,
                    pD=pD,
                    load_data=load_data,
                    generation_data=generation_data,
                    days=days,
                    T=T,
                    start_size=start_size,
                    freq=freq,
                    tau=tau)
        # 학습 시작
        agent.setup()

        with tqdm(range(epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)
        
        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history

        # model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}.pth".format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        # 학습 결과 저장 이름 : result_modelname_stategy_batchsize
        history_path = 'history/result_{}_{}_{}.pkl'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, 
                                                            batch_size)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

        title = '{}_{}_{}'.format(agent.__class__.__name__, training_strategy_fn().__class__.__name__, batch_size)

    # 엑터-크리틱 모델
    elif coder == 'AC':
        from AC import FCDAP, FCV, VPG

        policy_model_fn = lambda feature: FCDAP(feature)
        policy_optimizer_fn  = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        policy_model_max_grad_norm = 1
        policy_optimizer_lr = 0.0005

        value_model_fn = lambda feature: FCV(feature)
        value_model_max_grad_norm = float('inf')
        value_optimizer_fn = lambda net, lr: optim.RMSprop(net.parameters(), lr=lr)
        value_optimizer_lr = 0.0007

        max_gradient_norm = float('inf')
        entropy_loss_weight = 0.001

        # env
        from env import Environment
        make_env_fn = lambda feature, load_data, generation_data, days, T, pD: Environment(feature,
                                                                                            load_data, 
                                                                                            generation_data,
                                                                                            days, 
                                                                                            T, pD)

        agent = VPG(policy_model_fn=policy_model_fn,
                    policy_model_max_grad_norm=policy_model_max_grad_norm,
                    policy_optimizer_fn=policy_optimizer_fn,
                    policy_optimizer_lr = policy_optimizer_lr,
                    value_model_fn=value_model_fn,
                    value_model_max_grad_norm = value_model_max_grad_norm,
                    value_optimizer_fn = value_optimizer_fn,
                    value_optimizer_lr = value_optimizer_lr,
                    entropy_loss_weight = entropy_loss_weight,
                    battery_max=battery_max,
                    )
        
        # 학습 시작
        cost_history, action_history, battery_history = agent.train(make_env_fn = make_env_fn, feature=feature, epochs=epochs, start_day=start_day, end_day=end_day, 
                                                                    Tf=Tf,pD=pD, summer_TOU=summer_TOU, winter_TOU=winter_TOU,
                                                                    load_data=load_data, generation_data=generation_data, 
                                                                    days=days, T=T,
                                                                    gamma= gamma,
                                                                    )
        create_directory("weight")                
        policy_model_name = "weight/q_{}.pth".format(agent.__class__.__name__)
        torch.save(agent.policy_model.state_dict(), policy_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        # 학습 결과 저장 이름 : result_modelname_stategy_batchsize
        history_path = 'history/result_{}.pkl'.format(agent.__class__.__name__)
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

        
        title = '{}'.format(agent.__class__.__name__)


    # 그래프 출력
    if verbose:
        avg_cost_plot(cost_history, iteration, days, epochs, title)
     

def tester(f_load, f_generation, pos_name, 
            start_day, end_day, a, b, feature, 
            summer_TOU, winter_TOU, battery_max,
            T, Tf, pD,
            MPG=False,
            path="weight/q_model.pth"): 
    """
    
    """
    days = end_day-start_day
    # load, generation 데이터 형성
    load_data, generation_data = load_and_generte(f_load, f_generation, pos_name, start_day, end_day)

    result_cost_history, result_battery_history, result_action_history = [],[],[]

    def get_model(model, m=MPG, pretrained=False):
        # multi-GPU일 경우, Data Parallelism
        mdl = torch.nn.DataParallel(model(feature)) if m else model(feature)
        if not pretrained:
            return mdl
        else:
            print("load pretrained model here...")
            # 기학습된 torch.load()로 모델을 불러온다
            mdl.load_state_dict(torch.load(pretrained))
            return mdl

    q = get_model(Qnet, pretrained=path)

    battery, action = 0, 0

    # train
    state = initialize_state(feature, summer_TOU, T, load_data, generation_data, Tf)
    with tqdm(range(days), unit="Test") as runing_bar:
        for n_epi in runing_bar:
            epsilon = 0.01
            cum_cost, day_action, day_battery = 0, [] , []
            #if (start_day + n_epi % end_day) % 365 < 90 or (start_day + n_epi % end_day) % 365 >= 273:TOU = winter_TOU
            #else: TOU = summer_TOU
            TOU = summer_TOU

            for time in range(24):
                battery = state[0] - state[50] + state[51+Tf]
                if battery > battery_max: battery = battery_max  # generation 초과 제한
                action = sample_action(q, torch.from_numpy(state).float(), epsilon, battery, battery_max)
                day_action.append(action)
                day_battery.append(state[0])
                cost = cal_price(time,action,day_action,TOU, pD)
                cum_cost = cum_cost + cost
                state_prime = next_state(n_epi,time,battery,action,day_action,TOU,load_data,generation_data,  feature, Tf, days, T)
                state = state_prime

            result_cost_history.append(cum_cost)
            result_action_history.append(day_action)
            result_battery_history.append(day_battery)
    
    return result_cost_history, result_action_history, result_battery_history



def create_directory(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)