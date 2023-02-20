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
from sumo_rl.environment.env import SumoEnvironment
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
    #speedFactor="1" speedDev="0" departSpeed="speedLimit"
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

num_in_q = 0

def read_table(path):
    with open(Path(path), "rb") as f:
        q_table = json.load(f)
    return q_table
def find_close(state, q_table, num_not_in_q):
    if state in q_table:
        global num_in_q
        num_in_q += 1
        # print("-----------state in q table-----------")
        return state, num_not_in_q
    else:
        min_diff = float("inf")
        min_state = None
        num_not_in_q += 1
        for k in q_table.keys():
            diff = float(np.square(np.array(eval(k)) - np.array(eval(state))).sum())
            if min_state is None or diff < min_diff:
                min_state = k
                min_diff = diff
        print("-----------state not in q-----------")
    return min_state, num_not_in_q

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
    prs.add_argument("-runs", dest="runs", type=int, default=50, help="Number of runs.\n")
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
    prs.add_argument("-test", dest="test",default="True", type=str,required=False, help="constant c for UCB.\n") 
    prs.add_argument("-epsilon_change", dest="epsilon_change",default="True",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-alpha_change", dest="alpha_change",default="True",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-paths", dest="paths",default="new_state_e_lr/",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-new_state", dest="new_state",default="True",required=False, help="constant c for UCB.\n") 
    
    args = prs.parse_args()
    
    args.flow_ns = eval(args.flow_ns)
    args.flow_ew = eval(args.flow_ew)
    args.test = eval(args.test)
    args.new_state = eval(args.new_state)
    args.alpha_change = eval(args.alpha_change)
    args.epsilon_change = eval(args.epsilon_change)

    if args.test_flow:
        folder = str(Path('./outputs/constant/flow/dql/state_{}_flow_{}_{}_{}_'.format(args.state,args.flow_ns*3600,args.flow_ew*3600,args.explore_strategy)+args.paths))
        print("folder:",folder)
        file_name = os.path.join(folder, "logging")
        folders = os.path.join(folder, "q_table")
        #path for saving q table
        newpath_1 = os.path.join(folders, "q_table1")
        newpath_2 = os.path.join(folders, "q_table2")
        episode_output_path = os.path.join(folder, "timeloss")
        # episode_output_path = str(Path('./outputs/constant/flow/dql/state_{}_flow_{}_{}_{}_{}/timeloss/'.format(args.state, args.flow_ns*3600, args.flow_ew*3600, args.explore_strategy,args.ucb_c)))
        q_table1 = read_table(newpath_1)
        q_table2 = read_table(newpath_2)
        print("len of q1:",len(q_table1))
        print("len of q2:",len(q_table2))
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not os.path.exists(folders):
        os.makedirs(folders)
    
    
    #  python test.py -we=3 note:"-we" for assign value in terminal, dest for calling in code, required=False or no required -> if not given, default used/ True, must give value in cmd
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/single-intersection/{}_alpha{}_gamma{}_eps{}_decay{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay)
    total_freq_use = 0

    discounted_return = []

    print("flow:",args.flow_ns*3600,args.flow_ew*3600,',runs:', args.runs, ",road_type:", args.road_type, ",is fixed time:", args.fixed_time, ",state:", args.state,
    ",reward:", args.reward, ",min_green:", args.min_green,",Generate_cars_step:", args.Generate_cars_step, ",episode_time:", args.episode_time,",decay:",args.decay)
    actions = {'0': 0} # action initialization
    # for epi in range(400,500):
    for epi in range(args.runs):
        num_not_in_q = 0

        d, vehNr = generate_routefile(args.flow_ns,args.flow_ew,epi,args.Generate_cars_step, args.road_type)
        # print('vehNr:', vehNr)

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
                        test = args.test,
                        new_state =args.new_state)
        
        initial_states = env.reset() # every episode, reset signal using reset, change self.traffic signal
        
        if args.explore_strategy == "e-greedy":
            exploration = EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay)
        elif args.explore_strategy == "ucb":
            exploration = UCB(args.ucb_c)
        dql_agents = {ts: DoubleQLAgent(starting_state=env.encode(initial_states[ts], ts),
                                state_space=env.observation_space,
                                action_space=env.action_space,
                                alpha=args.alpha,
                                gamma=args.gamma,
                                path_1 = q_table1,
                                path_2 = q_table2,
                                exploration_strategy=exploration) for ts in env.ts_ids}
        # env.ts_ids a list of ids of all traffic lights 
        done = {'__all__': False}
        infos = []
        """file_name = file_name+"{}".format(epi)
        logger = logging.getLogger(__file__)
        fh = logging.FileHandler(filename=file_name, mode="w", delay=False)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        logger.addHandler(logging.StreamHandler())"""
        while not done['__all__']:
            'take action, get next state, get reward'
            s, r, selected_action, done, _ = env.step(action=actions) # postion arguments

            for agent_id in dql_agents.keys():
                if s[agent_id][-1] > 60:
                    print(s[agent_id][-1])
                next_state = env.encode(s[agent_id], agent_id) 
                close_state, num_not_in_q = find_close(next_state, q_table1, num_not_in_q)
                q1_action_values = np.array(q_table1[close_state])
                q2_action_values = np.array(q_table2[close_state])
                actions = {ts: np.argmax(q1_action_values+q2_action_values) for ts in dql_agents.keys()}
                # logger.error('{}_{}_{}'.format(s[agent_id],close_state,q1_action_values+q2_action_values))
                if next_state != close_state:
                    print("nest_state:",next_state)
                    print("close state:",close_state)
                    print("q values:", q1_action_values+q2_action_values)
            if num_not_in_q > 100:
                break
        """end of one epi"""

        print("epi:", epi)
        # if num_not_in_q > 50:
        env.close()

    # end of all epi
    
    print("__________________________end of all epi_____________________________________")
    


       