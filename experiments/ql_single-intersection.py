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
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy
import json

def gen_rou_varying(epi):
    prob = []
    random.seed(42)
    path = str(Path("nets/single-intersection/two_way/single-intersection_{}.rou.xml".format(epi)).absolute())
    with open(path, "w") as routes:
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
        vehNr_ns = 0
        vehNr_ew = 0
        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="0" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,150/3600),file=routes)
            vehNr_ns += 150/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="0" end="1800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,0,rou,550/3600),file=routes)
            vehNr_ew += 550/3600*1800
        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="1800" end="3600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,250/3600),file=routes)
            vehNr_ns += 250/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="1800" end="3600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,1,rou,650/3600),file=routes)
            vehNr_ew += 650/3600*1800

        for rou in ["northbound","southbound"]:
            print('     <flow id="{}_{}" route="{}" begin="3600" end="5400" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,400/3600),file=routes)
            vehNr_ns += 400/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="3600" end="5400" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,2,rou,800/3600),file=routes)
            vehNr_ew += 800/3600*1800
            
        for rou in ["northbound","southbound"]:
            print('    <flow id="{}_{}" route="{}" begin="5400" end="7200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,3,rou,500/3600),file=routes)
            vehNr_ns += 500/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="5400" end="7200" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,3,rou,900/3600),file=routes)
            vehNr_ew += 900/3600*1800

        for rou in ["northbound","southbound"]:
            print('    <flow id="{}_{}" route="{}" begin="7200" end="9000" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,4,rou,450/3600),file=routes)
            vehNr_ns += 450/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="7200" end="9000" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,4,rou,850/3600),file=routes)
            vehNr_ew += 850/3600*1800

        for rou in ["northbound","southbound"]:
            print('    <flow id="{}_{}" route="{}" begin="9000" end="10800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,5,rou,350/3600),file=routes)
            vehNr_ns += 350/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="9000" end="10800" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,5,rou,750/3600),file=routes)
            vehNr_ew += 750/3600*1800

        for rou in ["northbound","southbound"]:
            print('    <flow id="{}_{}" route="{}" begin="10800" end="12600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,6,rou,200/3600),file=routes)
            vehNr_ns += 200/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="10800" end="12600" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,6,rou,600/3600),file=routes)
            vehNr_ew += 600/3600*1800

        for rou in ["northbound","southbound"]:
            print('    <flow id="{}_{}" route="{}" begin="12600" end="14400" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,7,rou,100/3600),file=routes)
            vehNr_ns += 100/3600*1800
        for rou in ["westbound","eastbound"]:
            print('    <flow id="{}_{}" route="{}" begin="12600" end="14400" probability="{}" type="typeWE" \
                departSpeed="max" departPos="base" departLane="best"/>'.format(rou,7,rou,500/3600),file=routes)
            vehNr_ew += 500/3600*1800
        print("</routes>", file=routes)
        print("ns:", vehNr_ns,"ew:",vehNr_ew)
    return    

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
    else:
        path = Path("nets/single-intersection/one_way/single-intersection_{}.rou.xml".format(epi)).absolute()
        with open("w") as routes:
            print("""<routes>
            <vType id="typeWE" accel="2.6" decel="4.5" length="5" minGap="2.5" maxSpeed="13.90" \
    guiShape="passenger"/>
            <vType id="typeNS" accel="2.6" decel="4.5" length="5" minGap="2.5" maxSpeed="13.90" \
            guiShape="passenger"/>
            <route id="eastbound" edges="w_t t_e" />
            <route id="northbound" edges="n_t t_s" />
            
            """, file=routes)
            vehNr = 0
            for i in range(N):
                if  random.uniform(0, 1) < p2:
                    print('    <vehicle id="eastbound_0_%i" type="typeWE" route="eastbound" depart="%i" departSpeed="max" departLane="0"/>' % (
                        vehNr, i), file=routes)
                    d['eastbound_0_{}'.format(vehNr)] = i
                    veh_w_0 += 1
                    vehNr += 1
                if  random.uniform(0, 1) < p2:
                    print('    <vehicle id="eastbound_1_%i" type="typeWE" route="eastbound" depart="%i" departSpeed="max" departLane="1"/>' % (
                        vehNr, i), file=routes)
                    d['eastbound_1_{}'.format(vehNr)] = i
                    veh_w_1 += 1
                    vehNr += 1
        
                if random.uniform(0, 1) < p1:
                    print('    <vehicle id="northbound_0_%i" type="typeNS" route="northbound" depart="%i" departSpeed="max" departLane="0"/>' % (
                        vehNr, i), file=routes)
                    d['northbound_0_{}'.format(vehNr)] = i
                    veh_n_0 += 1
                    vehNr += 1

                if random.uniform(0, 1) < p1:
                    print('    <vehicle id="northbound_1_%i" type="typeNS" route="northbound" depart="%i" departSpeed="max" departLane="1"/>' % (
                        vehNr, i), file=routes)
                    d['northbound_1_{}'.format(vehNr)] = i
                    veh_n_1 += 1
                    vehNr += 1
            print("</routes>", file=routes)
    print('veh_n_0:', veh_n_0, 'veh_n_1:', veh_n_1, 'veh_w_0:', veh_w_0, 'veh_w_1:', veh_w_1)

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
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
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
    prs.add_argument("-epsilon_change", dest="epsilon_change",default=True,required=False, help="constant c for UCB.\n") 
    prs.add_argument("-alpha_change", dest="alpha_change",default=True,required=False, help="constant c for UCB.\n") 
    args = prs.parse_args()
    
    args.flow_ns = eval(args.flow_ns)
    args.flow_ew = eval(args.flow_ew)

    if args.test_flow:
        folder = str(Path('./outputs/constant/flow/state_{}_flow_{}_{}_{}_new_state/'.format(args.state,args.flow_ns*3600,args.flow_ew*3600,args.explore_strategy)))
        file_name = str(Path('./outputs/constant/flow/state_{}_flow_{}_{}_{}_{}_new_state/logging/logging'.format(args.state,args.flow_ns*3600,args.flow_ew*3600, args.explore_strategy, args.ucb_c)))
        
        file_name = os.path.join(folder, "logging")
        folders = os.path.join(folder, "q_table")
        #path for saving q table
        newpath = Path(os.path.join(folders, "q_table"))
        action_path = Path(os.path.join(folders, "action_dict"))
        episode_output_path = os.path.join(folder, "timeloss")
        path_freq = Path(os.path.join(folders, "freq"))
        
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

    # q_table for plot
    q_table_plot = {0:[],1:[]}
    labels = ['action 1', 'action 2']
    discounted_return = []
    avg_delay_all_epi = []

    print("flow:",args.flow_ns*3600,args.flow_ew*3600,',runs:', args.runs, ",road_type:", args.road_type, ",is fixed time:", args.fixed_time, ",state:", args.state,
    ",reward:", args.reward, ",min_green:", args.min_green,",Generate_cars_step:", args.Generate_cars_step, ",episode_time:", args.episode_time,",decay:",args.decay)
    
    for epi in range(args.runs):
        #every episode, generate cars, save dict of cars, ini env, ini ts(use different veh dict), ini agent
        if args.constant_demand:
            d, vehNr = generate_routefile(args.flow_ns,args.flow_ew,epi,args.Generate_cars_step, args.road_type)
            print('vehNr:', vehNr)
        else:
            gen_rou_varying(epi)
            d = {}
   
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
                        episode_output_path = episode_output_path)
        
        initial_states = env.reset() # every episode, reset signal using reset, change self.traffic signal

        if epi == 0: # episode
            path_ = None # initialize a blank dict for q table
            dict_freq = None
            action_dict = None
            action_dict_new = None
        else:
            path_ = q_table_use # use q table from previous episodes
            dict_freq = dict_freq_use
            action_dict = action_dict_use
            action_dict_new = action_dict_use_new
        
        if args.explore_strategy == "e-greedy":
            exploration = EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay, epsilon_change=args.epsilon_change)
        elif args.explore_strategy == "ucb":
            exploration = UCB(args.ucb_c)
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                state_space=env.observation_space,
                                action_space=env.action_space,
                                alpha=args.alpha,
                                gamma=args.gamma,
                                path = path_,
                                exploration_strategy=exploration,
                                fixed_time=args.fixed_time, 
                                dict_freq = dict_freq, 
                                action_dict = action_dict,
                                alpha_change = args.alpha_change,
                                action_dict_new = action_dict_new) for ts in env.ts_ids}
        # breakpoint()
        # env.ts_ids a list of ids of all traffic lights 
        done = {'__all__': False}
        infos = []
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:
                'choose action' # according to next state
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
                # ts is one of the traffic lights from above dict, if single traffic light, just one agent
                # actions: return a chosed action from al agent.py, dict
                'take action, get next state, get reward'
                s, r, selected_action, done, _ = env.step(action=actions) # postion arguments
                # selected_action, action actually taken
                'update q table'
                for agent_id in ql_agents.keys():
                    if not args.fixed_time: #fixed_time default: True, RL
                        use_action = selected_action
                    else: # fixed time
                        use_action = actions[agent_id] 
                    q_table_ = ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), selected_action=use_action, reward=r[agent_id])
        """end of one epi"""           
        for agent_id in ql_agents.keys(): # save q table for next episode
            # print('q_table_{}:'.format(agent_id), ql_agents[agent_id].q_table)
            # print('len of q table',len(ql_agents[agent_id].q_table))
            q_table_use = ql_agents[agent_id].q_table
            dict_freq_use = ql_agents[agent_id].dict_freq
            action_dict_use = ql_agents[agent_id].freq_by_state_by_action
            action_dict_use_new = ql_agents[agent_id].freq_by_state_by_action_new

        if  not args.fixed_time:#only RL plot
            for ts in env.ts_ids:
                # one traffic signal
                if args.reward == "lane_based_delay" or args.reward == "acc_pressure":
                    discounted_return.append(env.discounted_return)
                    # print("delay every 5s",env.delay_container)
                
                else:
                    discounted_return.append(env.traffic_signals[ts].discounted_r)
            """discounted_r: acc reward of one epi"""
            # print('discounted_return',discounted_return)

        else:
            for ts in env.ts_ids:
                dict_veh_delay = env.traffic_signals[ts].veh_delay
                    
                # avg_delay_all_epi.append(sum(dict_veh_delay.values())/len(dict_veh_delay))
        for agent_id in ql_agents.keys():
            # print('q_table_{}:'.format(agent_id), ql_agents[agent_id].q_table)
            # print('q_table_freq{}:'.format(agent_id), ql_agents[agent_id].dict_freq)
            print('len of q table',len(ql_agents[agent_id].q_table))
            # print('action table',len(ql_agents[agent_id].freq_by_state_by_action))
            # print("tslg_dict:", env.tslg_dict)
    
        """end of one epi"""

        print("epi:", epi)
        env.close()

    # end of all epi

    logger.error('discounted_return_{}'.format(discounted_return))
    
    for agent_id in ql_agents.keys():       
        with newpath.open( "w") as f:
            json.dump( ql_agents[agent_id].q_table, f)
        with path_freq.open( "w") as f:
            json.dump( ql_agents[agent_id].dict_freq, f)
        with action_path.open( "w") as f:
            json.dump( ql_agents[agent_id].freq_by_state_by_action, f)
        
        logger.error("len of q table{}".format(len(ql_agents[agent_id].q_table)))
        logger.error("q table{}".format(ql_agents[agent_id].q_table))
        logger.error("freq q table{}".format(ql_agents[agent_id].dict_freq))
    

    print("__________________________end of all epi_____________________________________")
    


       