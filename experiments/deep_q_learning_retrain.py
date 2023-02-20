'cd to path'
'export PYTHONPATH=./' 'current path sumo rl'
'run python experiments/ql_single-intersection.py'
# bash run_exp.sh 
# bash run_exp.sh > output.txt
# sumo --help

"""
cd /usr/local/opt/sumo/share/sumo
python /usr/local/opt/sumo/share/sumotools/xml/xml2csv.py /Users/linlyu/Desktop/sumo/max_pressure/stat_interval_new_25_900.0_450.0.xml
"""

"""
python experiments/ql_single-intersection.py -h
usage: ql_single-intersection.py [-h] [-route ROUTE] [-a ALPHA] [-g GAMMA]
                                 [-e EPSILON] [-me MIN_EPSILON] [-d DECAY]
                                 [-mingreen MIN_GREEN] [-maxgreen MAX_GREEN]
                                 [-gui] [-fixed] [-ns NS] [-we WE]
                                 [-s SECONDS] [-v] [-runs RUNS]
"""
import math
from collections import Counter
import argparse
import os
import sys
from datetime import datetime
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
from pathlib import Path
from sumo_rl.exploration.ucb import UCB
from sumo_rl.exploration.ucb import dql_UCB
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.agents.deep_q.deep_q_agent import DeepQAgent
from sumo_rl.agents.deep_q.model import TrainModel, TestModel
from sumo_rl.agents.deep_q.memory import Memory
from sumo_rl.exploration import EpsilonGreedy
import json 
from tensorflow.keras import backend as K


def generate_routefile(p1,p2,epi,step,road_type):
    d = {}

    random.seed(42)  # make tests reproducible
    N = step  # number of time steps
    # print(N)
    # demand per second from different directions
    veh_n_0 = 0
    veh_n_1 = 0
    veh_w_0 = 0
    veh_w_1 = 0
    
    # The driver imperfection (0 denotes perfect driving)
    # print("road_type:", road_type)
    if road_type == "two_way":
        path = Path("nets/single-intersection/two_way/single-intersection_{}.rou.xml".format(epi)).absolute()
        with path.open("w") as routes:
            print("""<routes>
            <vType id="typeWE" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
    guiShape="passenger"/>
            <vType id="typeEW" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
    guiShape="passenger"/>
            <vType id="typeNS" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
            guiShape="passenger"/>
            <vType id="typeSN" accel="2.6" decel="4.5" sigma="0" length="5" minGap="2.5" maxSpeed="15.65" \
            guiShape="passenger"/>
            <route id="eastbound" edges="1i 2i" />
            <route id="westbound" edges="-2i -1i" />
            <route id="northbound" edges="-3i 4i" />
            <route id="southbound" edges="-4i 3i" />
            """, file=routes)
            vehNr = 0
            for i in range(N):
                if random.uniform(0, 1) < p2:
                    print('    <vehicle id="eastbound_%i" type="typeWE" route="eastbound" depart="%i" departSpeed="max" />' % (
                        vehNr, i), file=routes)
                    d['eastbound_{}'.format(vehNr)] = i
                    veh_w_0 += 1
                    vehNr += 1
                if random.uniform(0, 1) < p2:
                    print('    <vehicle id="westbound_%i" type="typeEW" route="westbound" depart="%i" departSpeed="max"/>' % (
                        vehNr, i), file=routes)
                    d['westbound_{}'.format(vehNr)] = i
                    veh_w_1 += 1
                    vehNr += 1
                if random.uniform(0, 1) < p1:
                    print('    <vehicle id="southbound_%i" type="typeNS" route="southbound" depart="%i" departSpeed="max"/>' % (
                        vehNr, i), file=routes)
                    d['southbound_{}'.format(vehNr)] = i
                    veh_n_0 += 1
                    vehNr += 1
                if random.uniform(0, 1) < p1:
                    print('    <vehicle id="northbound_%i" type="typeSN" route="northbound" depart="%i" departSpeed="max"/>' % (
                        vehNr, i), file=routes)
                    d['northbound_{}'.format(vehNr)] = i
                    veh_n_1 += 1
                    vehNr += 1
            print("</routes>", file=routes)
    
    return d, vehNr

# new_method = True # old state


new_method = False # old state
if __name__ == '__main__':
    # p1 = 0.08333 # 300veh/h/lane
    # p2 = 0.194 #per lane 700veh/h/lane
    # Path(args.route).open()
    # open(args.route)
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='nets/single-intersection/single-intersection.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=1e-3, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.7, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", default="0.05", required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0, required=False, help="Minimum epsilon.\n")
    # prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-decay", dest="decay", type=float, default=1, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=5, required=False, help="Minimum green time.\n", choices=list(range(0,31))) # max fftt 
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n") # several veh files, decrease random
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=300, help="Number of runs.\n")
    prs.add_argument("-state", dest="state", type=str, default='9ele', help="state choices", choices=["num_veh","4ele_den_tslg","6ele_den_tslg","5ele_mg_time_density_avg","6ele_mg_time_density_avg","4ele_density_avg","3ele_density_avg","9ele","5ele","3ele_queue","3ele_density"])
    prs.add_argument("-reward", dest="reward", type=str, default='acc_pressure', help="reward choices", choices=["r_waiting","pressure", "lane_based_delay","acc_pressure"])
    prs.add_argument("-delta_time", dest="delta_time", type=int, default=5, required=False, help="Time between signal change.\n") # 25 for fixed time; 5 for RL
    prs.add_argument("-fixed_time", dest="fixed_time", default=False, required=False, help="fixed time or not.\n")
    prs.add_argument("-road_type", dest="road_type", type=str, default="two_way", required=False, help="way type.\n", choices=["one_way","two_way"])
    prs.add_argument("-Generate_cars_step", dest="Generate_cars_step", type=int, default=1800, required=False, help="generating cars step.\n")
    prs.add_argument("-flow_ns", dest="flow_ns", type=str, default='300/3600', required=False, help="flow of n/s.\n")
    prs.add_argument("-flow_ew", dest="flow_ew", type=str, default='700/3600', required=False, help="flow of e/w.\n")
    prs.add_argument("-episode_time", dest="episode_time", type=int, default=1800, required=False, help="episode_time.\n")
    prs.add_argument("-constant_demand", dest="constant_demand",default=True, required=False, help="constant_demand.\n")
    prs.add_argument("-test_flow", dest="test_flow",default=True, required=False, help="constant_demand.\n")
    prs.add_argument("-explore_strategy", dest="explore_strategy",default="e-greedy", required=False, help="exploration strategy.\n", choices=["e-greedy","ucb"])
    prs.add_argument("-ucb_c", dest="ucb_c",default=2, type=float,required=False, help="constant c for UCB.\n") 
    prs.add_argument("-epsilon_change", dest="epsilon_change",type=str,default="False",required=False) 
    prs.add_argument("-alpha_change", dest="alpha_change",default="False",required=False) 
    prs.add_argument("-new_state", dest="new_state",default="True",required=False) 
    prs.add_argument("-folder_path", dest="folder_path",default="DDQN",required=False) 
    prs.add_argument("-update_every_c_step", dest="update_every_c_step",default=300,required=False) 
    prs.add_argument("-RLmodel", dest="RLmodel",default="DDQN",required=False) 
    prs.add_argument("-state_dimension", dest="state_dimension",default=10,required=False) 
    prs.add_argument("-action_dimension", dest="action_dimension",default=2,required=False) 
    prs.add_argument("-sample_size_max", dest="sample_size_max",default=10000,required=False) 
    prs.add_argument("-sample_size_min", dest="sample_size_min",default=300,required=False) 
    prs.add_argument("-batch_size", dest="batch_size",default=100,required=False) 
    prs.add_argument("-training_epochs", dest="training_epochs",default=100,required=False) 
    prs.add_argument("-num_layers", dest="num_layers",default=4,required=False) 
    prs.add_argument("-neurons", dest="neurons",default=64,required=False) 
    prs.add_argument("-epsilon_decay", dest="epsilon_decay",default="False",required=False)
    prs.add_argument("-lr_change", dest="lr_change",default="False",required=False)

    args = prs.parse_args()
    args.epsilon_decay = eval(args.epsilon_decay)
    args.lr_change = eval(args.lr_change)
    args.flow_ns = eval(args.flow_ns)
    args.flow_ew = eval(args.flow_ew)
    args.epsilon_change = eval(args.epsilon_change)
    args.epsilon = eval(args.epsilon)
    args.alpha_change = eval(args.alpha_change)
    args.new_state = eval(args.new_state)

    
    """path__ = "C:\\Users\\lvl5569\\Desktop\\newest_code\\outputs\\constant\\flow\\dql\\state_6ele_den_tslg_flow_500.0_900.0_e-greedy\\q_table\\"
    with open(path__+"/q_table1", "rb") as f:
        q_table_use_1 = json.load(f)
    with open(path__+"/q_table2", "rb") as f:
        q_table_use_2 = json.load(f)""" 

    if args.test_flow:
        folder = str(Path('./outputs/constant/flow/dql/adam/{}/state_{}_flow_{}_{}_{}_reward{}_e_decay{}_lr_decay{}_f{}/'.format(args.RLmodel,args.state,
        args.flow_ns*3600,args.flow_ew*3600,args.explore_strategy,args.reward,args.epsilon_decay,args.lr_change,args.decay)))
        print("folder:",folder)
        # breakpoint()
        file_name = os.path.join(folder, "logging")
        folders = os.path.join(folder, "model")
        #path for saving q table
        discount_r = Path(os.path.join(folders, "discount_r"))
        newpath_1 = Path(os.path.join(folders, "q_table1"))
        newpath_2 = Path(os.path.join(folders, "q_table2"))
        action_path = Path(os.path.join(folders, "action_dict"))
        episode_output_path = os.path.join(folder, "timeloss")
        exploration_path = Path(os.path.join(folders, "exploration_list"))
        uncertainty1 = Path(os.path.join(folders, "uncertainty1"))
        uncertainty2 = Path(os.path.join(folders, "uncertainty2"))

    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folders):
        os.makedirs(folders)
    logger = logging.getLogger(__file__)
    fh = logging.FileHandler(filename=file_name, mode="w", delay=False)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(logging.StreamHandler())
    
    #  python test.py -we=3 note:"-we" for assign value in terminal, dest for calling in code, required=False or no required -> if not given, default used/ True, must give value in cmd
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/single-intersection/{}_alpha{}_gamma{}_eps{}_decay{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay)

    discounted_return = []
    exploration_lst_all = []
    uncertainty_lst1 = []
    uncertainty_lst2 = []

    print("epsilon:",args.epsilon_change,"flow:",args.flow_ns*3600,args.flow_ew*3600,',runs:', args.runs, ",road_type:", args.road_type, ",is fixed time:", args.fixed_time, ",state:", args.state,
    ",reward:", args.reward, ",min_green:", args.min_green,",Generate_cars_step:", args.Generate_cars_step, ",episode_time:", args.episode_time,",decay:",args.decay)
    """
    {'gui': False, 'total_episodes': 100, 'max_steps': 5400, 
    'n_cars_generated': 1000, 'green_duration': 10, 'yellow_duration': 4, 
    'num_layers': 4, 'width_layers': 400, 'batch_size': 100, 
    'learning_rate': 0.001, 'training_epochs': 800, 'memory_size_min': 600, 
    'memory_size_max': 50000, 'num_states': 80, 'num_actions': 4, 'gamma': 0.75, 
    'models_path_name': 'models', 'sumocfg_file_name': 'sumo_config.sumocfg'}
    """
    # assert args.state == "9ele"
    dimension_of_state = args.state_dimension
    number_of_action = args.action_dimension
    ml_model = TrainModel(num_layers=args.num_layers, width=args.neurons, batch_size=args.batch_size, learning_rate=args.alpha, input_dim=dimension_of_state, output_dim=number_of_action)
    memory = Memory(size_max=args.sample_size_max, size_min=args.sample_size_min)

    # for epi in range(400,500):
    for epi in range(args.runs):

        d, vehNr = generate_routefile(args.flow_ns,args.flow_ew,epi,args.Generate_cars_step, args.road_type)
        print('vehNr:', vehNr)

        route_file_name = str(Path('nets/single-intersection/'+args.road_type+'/single-intersection_{}.rou.xml'.format(epi)).absolute())
        env = SumoEnvironment(net_file=str(Path('nets/single-intersection/'+args.road_type+'/single-intersection.net.xml').absolute()),
                        route_file=route_file_name,
                        out_csv_name=out_csv,
                        use_gui=args.gui,
                        num_seconds=args.seconds,
                        min_green=args.min_green,
                        max_green=args.max_green,
                        max_depart_delay=0,
                        epi = epi,
                        delta_time = args.delta_time,
                        road_type = args.road_type,
                        state = args.state,
                        reward = args.reward,
                        sim_max_time = args.episode_time,
                        veh_info = d,
                        gamma = args.gamma,
                        test_flow = args.test_flow,
                        flow_ns = args.flow_ns,
                        flow_ew = args.flow_ew,
                        strategy = args.explore_strategy,
                        c_ucb = args.ucb_c,
                        episode_output_path = episode_output_path,
                        new_state =args.new_state)
        
        initial_states = env.reset() # every episode, reset signal using reset, change self.traffic signal
        # self._compute_observations_9()[self.ts_ids[0]]
        # print('initial_states:', initial_states['0'])
        if args.epsilon_decay:
            args.epsilon = min(pow(args.decay,args.runs),0.05)
        if args.explore_strategy == "e-greedy":
            exploration = EpsilonGreedy(initial_epsilon=args.epsilon, 
                                        min_epsilon=args.min_epsilon, 
                                        decay=args.decay,
                                        epsilon_change =args.epsilon_change)
        elif args.explore_strategy == "ucb":
            exploration = dql_UCB(args.ucb_c)

        deep_q_agents = {ts: DeepQAgent(starting_state=initial_states[ts],
                                state_space=env.observation_space,
                                action_space=env.action_space,
                                model=ml_model,
                                memory=memory,
                                epsilon=args.epsilon,
                                gamma=args.gamma
                                ) for ts in env.ts_ids}
        # env.ts_ids a list of ids of all traffic lights 
        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            i = 0
            while not done['__all__']:
                actions = {ts: deep_q_agents[ts].act() for ts in deep_q_agents.keys()}
                s, r, selected_action, done, _ = env.step(action=actions) # postion arguments
                for agent_id in deep_q_agents.keys():
                    if not args.fixed_time: #fixed_time default: True, RL
                        use_action = selected_action
                    else: # fixed time
                        use_action = actions[agent_id] 
                    deep_q_agents[agent_id].get_data(next_state=s[agent_id], selected_action=use_action, reward=r[agent_id])
                        
                # for _ in tqdm(range(args.training_epochs-1)):
                # for _ in range(args.training_epochs-1):
                #     deep_q_agents[agent_id].replay(update=False,model=args.RLmodel)
                    if (i+1) % args.update_every_c_step == 0:
                        deep_q_agents[agent_id].replay(update=True,model=args.RLmodel) 
                    else:
                        deep_q_agents[agent_id].replay(update=False,model=args.RLmodel) 
                    i += 1
        if args.lr_change:
            ml_model.change_lr(args.decay,change=True) 
        # print("lr:",K.eval(ml_model._model.optimizer.lr))      
        env.close()
        # breakpoint()
        """end of one epi"""

        for ts in env.ts_ids:
            # one traffic signal
            if args.reward == "lane_based_delay" or args.reward == "acc_pressure":
                discounted_return.append(env.discounted_return)
                # print("delay every 5s",env.delay_container)
                print("discounted_return:",env.discounted_return)
            
            else:
                discounted_return.append(env.traffic_signals[ts].discounted_r)
                print("discounted_return list:",discounted_return)
            
            """discounted_r: acc reward of one epi"""
        if epi % 50 == 0:
            ml_model.save_model(folders, epi)
    ml_model.save_model(folders, epi)
    print("discounted_return:",discounted_return)
     
    with discount_r.open( "w") as f:
        json.dump(discounted_return, f)
    # # end of all epi
    
    print("__________________________end of all epi_____________________________________")
    


       