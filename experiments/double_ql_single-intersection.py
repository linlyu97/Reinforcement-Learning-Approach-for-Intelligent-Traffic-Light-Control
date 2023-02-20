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
from sumo_rl.agents import DoubleQLAgent
from sumo_rl.exploration import EpsilonGreedy
import json 


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
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.7, required=False, help="Gamma discount rate.\n", choices=[0.5,0.6,0.7,0.8,0.9,0.95,0.99])
    prs.add_argument("-e", dest="epsilon", default="0.05", required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0, required=False, help="Minimum epsilon.\n")
    # prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=0, required=False, help="Minimum green time.\n", choices=list(range(0,31))) # max fftt 
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n") # several veh files, decrease random
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=int, default=700, help="Number of runs.\n")
    prs.add_argument("-state", dest="state", type=str, default='6ele_den_tslg', help="state choices", choices=["4ele_den_tslg","6ele_den_tslg","5ele_mg_time_density_avg","6ele_mg_time_density_avg","4ele_density_avg","3ele_density_avg","9ele","5ele","3ele_queue","3ele_density"])
    prs.add_argument("-reward", dest="reward", type=str, default='lane_based_delay', help="reward choices", choices=["r_waiting","avg_delay", "lane_based_delay","acc_pressure"])
    prs.add_argument("-delta_time", dest="delta_time", type=int, default=5, required=False, help="Time between signal change.\n") # 25 for fixed time; 5 for RL
    prs.add_argument("-fixed_time", dest="fixed_time", default=False, required=False, help="fixed time or not.\n")
    prs.add_argument("-road_type", dest="road_type", type=str, default="two_way", required=False, help="way type.\n", choices=["one_way","two_way"])
    prs.add_argument("-Generate_cars_step", dest="Generate_cars_step", type=int, default=8*1800, required=False, help="generating cars step.\n")
    prs.add_argument("-flow_ns", dest="flow_ns", type=str, default='500/3600', required=False, help="flow of n/s.\n")
    prs.add_argument("-flow_ew", dest="flow_ew", type=str, default='900/3600', required=False, help="flow of e/w.\n")
    prs.add_argument("-episode_time", dest="episode_time", type=int, default=8*1800, required=False, help="episode_time.\n")
    prs.add_argument("-constant_demand", dest="constant_demand",default=True, required=False, help="constant_demand.\n")
    prs.add_argument("-test_flow", dest="test_flow",default=True, required=False, help="constant_demand.\n")
    prs.add_argument("-explore_strategy", dest="explore_strategy",default="e-greedy", required=False, help="exploration strategy.\n", choices=["e-greedy","ucb"])
    prs.add_argument("-ucb_c", dest="ucb_c",default=2, type=float,required=False, help="constant c for UCB.\n") 
    prs.add_argument("-epsilon_change", dest="epsilon_change",type=str,default="False",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-alpha_change", dest="alpha_change",default="True",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-new_state", dest="new_state",default="True",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-folder_path", dest="folder_path",default="no",required=False, help="constant c for UCB.\n") 
    
    args = prs.parse_args()
    
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
        folder = str(Path('./outputs/constant/flow/dql/{}/state_{}_flow_{}_{}_{}_new_state{}_epsilon{}/'.format(args.folder_path,args.state,args.flow_ns*3600,args.flow_ew*3600,args.explore_strategy,args.new_state,args.epsilon)))
        file_name = os.path.join(folder, "logging")
        folders = os.path.join(folder, "q_table")
        #path for saving q table
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
    
    # for epi in range(400,500):
    for epi in range(args.runs):

        d, vehNr = generate_routefile(args.flow_ns,args.flow_ew,epi,args.Generate_cars_step, args.road_type)
        print('vehNr:', vehNr)

        route_file_name = str(Path('nets/single-intersection/'+args.road_type+'/single-intersection_{}.rou.xml'.format(epi)).absolute())
        """net_file, sumo start simulation"""
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
        if epi == 0: # episode
            path_1 = None # initialize a blank dict for q table
            path_2 = None
            action_dict = {} 
            action_dict_1 = {}
            action_dict_2 = {} 
        else:
            path_1 = q_table_use_1 # use q table from previous episodes
            path_2 = q_table_use_2
            action_dict = action_dict_use
            action_dict_1 = action_dict_use_1
            action_dict_2 = action_dict_use_2
       
        if args.explore_strategy == "e-greedy":
            exploration = EpsilonGreedy(initial_epsilon=args.epsilon, 
                                        min_epsilon=args.min_epsilon, 
                                        decay=args.decay,
                                        epsilon_change =args.epsilon_change)
        elif args.explore_strategy == "ucb":
            exploration = dql_UCB(args.ucb_c)
        dql_agents = {ts: DoubleQLAgent(starting_state=env.encode(initial_states[ts], ts),
                                state_space=env.observation_space,
                                action_space=env.action_space,
                                alpha=args.alpha,
                                gamma=args.gamma,
                                path_1 = path_1,
                                path_2 = path_2,
                                exploration_strategy=exploration,
                                action_dict = action_dict,
                                action_dict_1 = action_dict_1,
                                action_dict_2 = action_dict_2,
                                alpha_change = args.alpha_change) for ts in env.ts_ids}
        # env.ts_ids a list of ids of all traffic lights 
        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:

                'choose action' # according to next state
                actions = {ts: dql_agents[ts].act() for ts in dql_agents.keys()}

                # ts is one of the traffic lights from above dict, if single traffic light, just one agent
                # actions: return a chosed action from al agent.py, dict
                'take action, get next state, get reward'
                s, r, selected_action, done, _ = env.step(action=actions) # postion arguments

                # selected_action, action actually taken
                'update q table'
                for agent_id in dql_agents.keys():
                    if not args.fixed_time: #fixed_time default: True, RL
                        use_action = selected_action
                    else: # fixed time
                        use_action = actions[agent_id] 
                    q_table_ = dql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), selected_action=use_action, reward=r[agent_id])
        """end of one epi"""
        """save exploration for one epi"""
        if args.explore_strategy == "e-greedy":
            exploration_lst_all.append(exploration.exploration_list)
            print("avg explore:",sum(exploration.exploration_list)/len(exploration.exploration_list))
        elif args.explore_strategy == "ucb":
            uncertainty_lst1.append(exploration.uncertainty1)
            uncertainty_lst2.append(exploration.uncertainty2)
            print("uncertainty:",sum(exploration.uncertainty1)/len(exploration.uncertainty1))
            print("uncertainty:",sum(exploration.uncertainty2)/len(exploration.uncertainty2))

        for agent_id in dql_agents.keys(): # save q table for next episode
            # print('q_table_{}:'.format(agent_id), ql_agents[agent_id].q_table)
            # print('len of q table',len(ql_agents[agent_id].q_table))
            q_table_use_1 = dql_agents[agent_id].q_table_1
            q_table_use_2 = dql_agents[agent_id].q_table_2
            action_dict_use = dql_agents[agent_id].freq_by_state_by_action
            action_dict_use_1 = dql_agents[agent_id].freq_by_state_by_action_q1
            action_dict_use_2 = dql_agents[agent_id].freq_by_state_by_action_q2

        for ts in env.ts_ids:
            # one traffic signal
            if args.reward == "lane_based_delay" or args.reward == "acc_pressure":
                discounted_return.append(env.discounted_return)
                # print("delay every 5s",env.delay_container)
            
            else:
                discounted_return.append(env.traffic_signals[ts].discounted_r)
            """discounted_r: acc reward of one epi"""
            
        for agent_id in dql_agents.keys():
            # print('q_table_{}:'.format(agent_id), ql_agents[agent_id].q_table)
            # print('q_table_freq{}:'.format(agent_id), ql_agents[agent_id].dict_freq)
            print('len of q table 1',len(dql_agents[agent_id].q_table_1))
            print('len of q table 2',len(dql_agents[agent_id].q_table_2))
            # print('action table',len(ql_agents[agent_id].freq_by_state_by_action))
        """end of one epi"""
        print("epi:", epi)
        env.close()

    # end of all epi

    # logger.error('discounted_return_{}'.format(discounted_return))

    # print('discounted_return',discounted_return)
    newpath_1 = Path(os.path.join(folders, "q_table1"))
    newpath_2 = Path(os.path.join(folders, "q_table2"))
    for agent_id in dql_agents.keys():       
        with newpath_1.open( "w") as f:
            json.dump( dql_agents[agent_id].q_table_1, f)
        with newpath_2.open( "w") as f:
            json.dump( dql_agents[agent_id].q_table_2, f)
        with action_path.open( "w") as f:
            json.dump( dql_agents[agent_id].freq_by_state_by_action, f)
        with exploration_path.open("w") as f:
            json.dump(exploration_lst_all, f)
        with uncertainty1.open("w") as f:
            json.dump(uncertainty_lst1, f)
        with uncertainty2.open("w") as f:
            json.dump(uncertainty_lst2, f)
    
    print("__________________________end of all epi_____________________________________")
    


       