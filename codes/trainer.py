from tqdm import tqdm
import torch.optim as optim
import os
import torch
import pickle
from codes.buffer import ReplayBuffer_vi_day, PrioritizedReplayBuffer
from codes.model import Qnet, DualingQnet, DQN, PER, LRDQN
from codes.env import Environment_minc
from codes.safe_model import REMS
from codes.utills import create_directory

def trainer(cfg):
    if cfg.coder == 'DQN':
        cfg.model.violation=True
        cfg.model.value_model_fn = lambda nS: Qnet(nS)
        cfg.model.value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        cfg.model.replay_buffer_fn = lambda : ReplayBuffer_vi_day(buffer_limit=30000, batch_size=cfg.model.batch_size)
        cfg.env = Environment_minc(cfg)
        agent = DQN(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)
        
        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        if cfg.model.violation:
            violation_list = agent.violation_list

        #model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}_{}.pth".format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        if cfg.model.violation:
            history['violation'] = violation_list

        # 학습 결과 저장 이름 : result-modelname-stategy-batchsize-seed num
        history_path = 'history/result_{}_{}_{}_{}.pkl'.format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

        create_directory("buffer")
        buffer_dict = {
            "s":agent.replay_buffer.ss_mem,
            "a":agent.replay_buffer.as_mem,
            "r":agent.replay_buffer.rs_mem,
            "s_":agent.replay_buffer.ps_mem,
            "vi":agent.replay_buffer.vi_mem,
        }
        
        buffer_path = 'buffer/result_{}_{}_{}_{}.pkl'.format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        
        with open(buffer_path,'wb') as f:
            pickle.dump(buffer_dict,f)
    
    elif cfg.coder == 'PER':
        cfg.model.violation = True
        cfg.model.tau = 0.1
        cfg.model.max_gradient_norm = float('inf')
        cfg.model.value_model_fn = lambda nS: DualingQnet(nS)
        cfg.model.value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        cfg.model.replay_buffer_fn = lambda : PrioritizedReplayBuffer(buffer_limit=15000, batch_size=cfg.model.batch_size, rank_based=True)
        cfg.env = Environment_minc(cfg)

        agent = PER(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        if cfg.model.violation:
            violation_list = agent.violation_list

        #model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}_{}.pth".format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        if cfg.model.violation:
            history['violation'] = violation_list

        # 학습 결과 저장 이름 : result-modelname-stategy-batchsize-seed num
        history_path = 'history/result_{}_{}_{}_{}.pkl'.format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

    elif cfg.coder == 'LRDQN':
        cfg.model.violation = True
        cfg.model.value_model_fn = lambda nS: Qnet(nS)
        cfg.model.value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        cfg.model.replay_buffer_fn = lambda : ReplayBuffer_vi_day(buffer_limit=30000, batch_size=cfg.model.batch_size)
        cfg.env = Environment_minc(cfg)

        agent = LRDQN(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)

        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        if cfg.model.violation:
            violation_list = agent.violation_list

        #model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}_{}.pth".format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        if cfg.model.violation:
            history['violation'] = violation_list

        # 학습 결과 저장 이름 : result-modelname-stategy-batchsize-seed num
        history_path = 'history/result_{}_{}_{}_{}.pkl'.format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    )
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

    elif cfg.coder == 'REMS':
        cfg.model.violation=True
        cfg.model.value_model_fn = lambda nS: Qnet(nS)
        cfg.model.value_optimizer_fn = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        cfg.model.replay_buffer_fn = lambda : ReplayBuffer_vi_day(buffer_limit=30000, batch_size=cfg.model.batch_size)
        cfg.env = Environment_minc(cfg)

        agent = REMS(cfg)
        agent.setup()
        with tqdm(range(cfg.model.epochs), unit="Run") as runing_bar:
            for _ in runing_bar:
                agent.epoch(runing_bar)
        
        cost_history, action_history, battery_history = agent.cost_history, agent.action_history, agent.battery_history
        if cfg.model.violation:
            violation_list = agent.violation_list

        #model save
        create_directory("weight")                
        online_model_name = "weight/q_{}_{}_{}_{}_{}_{}_{}.pth".format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    cfg.model.p,
                                                                    cfg.model.q, 
                                                                    cfg.model.H,
                                                                    )
        torch.save(agent.online_model.state_dict(), online_model_name)

        # result save
        create_directory("history")
        history = {
            "cost":cost_history,
            "action":action_history,
            "battery":battery_history,
        }
        if cfg.model.violation:
            history['violation'] = violation_list

        # 학습 결과 저장 이름 : result-modelname-stategy-batchsize-seed num
        history_path = 'history/result_{}_{}_{}_{}_{}_{}_{}.pkl'.format(agent.__class__.__name__, 
                                                                    cfg.model.training_strategy_fn().__class__.__name__, 
                                                                    cfg.model.batch_size, 
                                                                    cfg.seed, 
                                                                    cfg.model.p,
                                                                    cfg.model.q, 
                                                                    cfg.model.H,
                                                                    )
        with open(history_path,'wb') as f:
            pickle.dump(history,f)

def tester(cfg):
    q = cfg.m(cfg.model.nS)
    q.load_state_dict(torch.load(cfg.model.path))

    train_st = cfg.train_st_fc()
    state = cfg.EMS.env.initialize_state(cfg.EMS.start_day)
    battery, action = 0, 0
    cost_history, battery_history,action_history, violation_history = [],[],[],[]

    for day in range(cfg.EMS.days):
        TOU = cfg.EMS.winter_TOU
        cum_cost, day_action, day_battery, day_vio = 0, [] , [], []

        for time in range(0, 24):
            battery = state[0] - state[50] + state[51+cfg.model.Tf]
            if battery > cfg.EMS.battery_max: battery = cfg.EMS.battery_max
            action = train_st.select_action(q, torch.from_numpy(state).float(), battery, cfg.EMS.battery_max)

            day_action.append(action)
            day_battery.append(state[0])
            new_state, reward, violation = cfg.EMS.env.step(n_epi=day, time=time, battery=battery, charge=action, 
                                            day_charge=day_action, TOU=TOU)
            cum_cost = cum_cost + reward
            state = new_state
            day_vio.append(violation)

        cost_history.append(cum_cost)
        action_history.append(day_action)
        battery_history.append(day_battery)
        violation_history.append(day_vio)
    return cost_history, action_history, battery_history, violation_history