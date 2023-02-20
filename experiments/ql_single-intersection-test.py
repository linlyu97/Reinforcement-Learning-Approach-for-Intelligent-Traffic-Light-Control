import argparse
import os
import sys
from datetime import datetime
import random

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
import numpy as np
import pickle
from pathlib import Path
'export PYTHONPATH=./' 'current path sumo rl'
'run python experiments/ql_single-intersection-test.py'


def generate_routefile(p1,p2,epi,step,road_type,seeds):
    d = {}
    random.seed(seeds)  # make tests reproducible
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
        with open("nets/single-intersection/two_way/single-intersection_{}.rou.xml".format(epi), "w") as routes:
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
        with open('nets/single-intersection/one_way/single-intersection_{}.rou.xml'.format(epi), "w") as routes:
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

def gen_rou_varying(epi):
    prob = []
    random.seed(42)
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

num_in_q = 0
# global num_in_q


def find_close(state, q_table,num_in_q):
    if state in q_table:
        
        # print("-----------state in q table-----------")
        return state,num_in_q
    else:
        num_in_q += 1
        min_diff = float("inf")
        min_state = None
        for k in q_table.keys():
            diff = float(np.square(np.array(eval(k)) - np.array(eval(state))).sum())
            if min_state is None or diff < min_diff:
                min_state = k
                min_diff = diff
        print("-----------state not in q-----------")
    return min_state,num_in_q

# mean = np.array([0.69945355,0.43169399,0.92622951,0.44808743,0.96174863,22.71857923])
# std = np.array([0.45849567,0.52737195,1.01492711,0.51351585,1.0289053,17.23907856])

# mean = np.array([ 0.71917808,  0.41438356,  0.91780822 , 0.39726027,  0.92123288, 22.12328767])
# std = np.array([ 0.44940068,  0.49951891, 1.02373777 , 0.49628003 , 1.02567778, 16.21068074])

def norm_q_table(q_table, max):
    norm_q = {}
    for key in q_table:
        new_key = []
        new_key.append(eval(key)[0])
        k_norm = np.array(eval(key)[1:-1])/np.array([9,9,9,9])
        k_norm = k_norm.tolist()
        duration = eval(key)[-1]/max
        # print("k_nrom",k_norm)
        # print("duration",duration)
        # breakpoint()
        new_key.extend(k_norm)
        new_key.append(duration)
        norm_q[str(new_key)] = q_table[key]
    return norm_q

def norm_state(state,max):
    new_key = []
    new_key.append(eval(state)[0])
    k_norm = np.array(eval(state)[1:-1])/np.array([9,9,9,9])
    k_norm = k_norm.tolist()
    duration = eval(state)[-1]/max
    # print("k_nrom",k_norm)
    # print("duration",duration)
    # breakpoint()
    new_key.extend(k_norm)
    new_key.append(duration)
    return str(new_key)


def find_close_center(state, q_center, mean, std):
    state_norm = (np.array(eval(state)) - mean) /std
    # print("state from environment after norm:",state_norm)
    min_diff = float("inf")
    min_state = None
    for k in q_center.keys():
        diff = float(np.square(np.array(eval(k)) - state_norm).sum())
        if min_state is None or diff < min_diff:
            min_state = k
            min_diff = diff
    return min_state

def read_table(path):
    with open(Path(path+"/q_table"), "rb") as f:
        q_table = json.load(f)
    # with open(Path(path+"/freq"), "rb") as f:
    #     freq_q_table = json.load(f)
    # with open(Path(path+"/q_center"), "rb") as f:
    #     q_center = json.load(f)
    # with open(Path(path+"/final_q_table"), "rb") as f:
    #     tune_q = json.load(f) 
    # with open(Path(path+"/q_norm_tuned"), "rb") as f:
    #     q_norm_tuned = json.load(f) 
    # return q_table, freq_q_table, q_center,tune_q,q_norm_tuned 
    return q_table   
   

def load_file(floder_path, indicator):
    if indicator == "mean":
        file = open(Path(floder_path+"mean"), 'rb')
        # source, destination
        f = pickle.load(file)                     
        file.close()
    else:
        file = open(Path(floder_path+"std"), 'rb')
        # source, destination
        f = pickle.load(file)                     
        file.close()
    return f


new_method = False # old state
if __name__ == '__main__':
    # p1 = 0.08333 # 300veh/h/lane
    # p2 = 0.194 #per lane 700veh/h/lane
    

    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Q-Learning Single-Intersection""")
    prs.add_argument("-route", dest="route", type=str, default='nets/single-intersection/single-intersection.rou.xml', help="Route definition xml file.\n")
    prs.add_argument("-a", dest="alpha", type=float, default=0.1, required=False, help="Alpha learning rate.\n")
    prs.add_argument("-g", dest="gamma", type=float, default=0.7, required=False, help="Gamma discount rate.\n")
    prs.add_argument("-e", dest="epsilon", type=float, default=0.05, required=False, help="Epsilon.\n")
    prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    # prs.add_argument("-me", dest="min_epsilon", type=float, default=0.005, required=False, help="Minimum epsilon.\n")
    prs.add_argument("-d", dest="decay", type=float, default=1, required=False, help="Epsilon decay.\n")
    prs.add_argument("-mingreen", dest="min_green", type=int, default=0, required=False, help="Minimum green time.\n") # max fftt 
    prs.add_argument("-maxgreen", dest="max_green", type=int, default=50, required=False, help="Maximum green time.\n") # several veh files, decrease random
    prs.add_argument("-gui", action="store_true", default=False, help="Run with visualization on SUMO.\n")
    prs.add_argument("-fixed", action="store_true", default=False, help="Run with fixed timing traffic signals.\n")
    prs.add_argument("-ns", dest="ns", type=int, default=42, required=False, help="Fixed green time for NS.\n")
    prs.add_argument("-we", dest="we", type=int, default=42, required=False, help="Fixed green time for WE.\n")
    prs.add_argument("-s", dest="seconds", type=int, default=100000, required=False, help="Number of simulation seconds.\n")
    prs.add_argument("-v", action="store_true", default=False, help="Print experience tuple.\n")
    prs.add_argument("-runs", dest="runs", type=str, default="50", help="Number of runs.\n")
    prs.add_argument("-state", dest="state", type=str, default='6ele_den_tslg', help="state choices", choices=["4ele_den_tslg","6ele_den_tslg","5ele_mg_time_density_avg","6ele_mg_time_density_avg","4ele_density_avg","3ele_density_avg","9ele","5ele","3ele_queue","3ele_density"])
    prs.add_argument("-reward", dest="reward", type=str, default='lane_based_delay', help="reward choices", choices=["r_waiting","avg_delay", "lane_based_delay","acc_pressure"])
    prs.add_argument("-delta_time", dest="delta_time", type=int, default=5, required=False, help="Time between signal change.\n") # 25 for fixed time; 5 for RL
    prs.add_argument("-fixed_time", dest="fixed_time", default=False, required=False, help="fixed time or not.\n")
    prs.add_argument("-road_type", dest="road_type", type=str, default="two_way", required=False, help="way type.\n", choices=["one_way","two_way"])
    prs.add_argument("-Generate_cars_step", dest="Generate_cars_step", type=int, default=1800*8, required=False, help="generating cars step.\n")
    prs.add_argument("-flow_ns", dest="flow_ns", type=str, default="500/3600", required=False, help="flow of n/s.\n")
    prs.add_argument("-flow_ew", dest="flow_ew", type=str, default="900/3600", required=False, help="flow of e/w.\n")
    prs.add_argument("-episode_time", dest="episode_time", type=int, default=1800*8, required=False, help="episode_time.\n")
    prs.add_argument("-test", dest="test",default=True, required=False, help="test.\n")
    prs.add_argument("-cluster", dest="cluster",default="False", required=False, help="test.\n")
    prs.add_argument("-tune_q", dest="tune_q",default="False", required=False, help="test.\n")
    prs.add_argument("-test_path", dest="test_path",default="500.0_900.0_ucb_new_e_lr", required=False, help="test.\n")
    prs.add_argument("-seeds", dest="seeds",default="42", required=False, help="test.\n")
    prs.add_argument("-new_state", dest="new_state",default="False",required=False, help="constant c for UCB.\n") 
    prs.add_argument("-norm", dest="norm",default=True,required=False, help="constant c for UCB.\n") 
    
    args = prs.parse_args()
    args.flow_ns = eval(args.flow_ns)
    args.flow_ew = eval(args.flow_ew)
    args.seeds = eval(args.seeds)
    args.cluster = eval(args.cluster)
    args.tune_q = eval(args.tune_q)
    args.runs = eval(args.runs)
    args.new_state = eval(args.new_state)
    # print(args.tune_q)
    #  python test.py -we=3 note:"-we" for assign value in terminal, dest for calling in code, required=False or no required -> if not given, default used/ True, must give value in cmd
    experiment_time = str(datetime.now()).split('.')[0]
    out_csv = 'outputs/single-intersection/{}_alpha{}_gamma{}_eps{}_decay{}'.format(experiment_time, args.alpha, args.gamma, args.epsilon, args.decay)
    

    # q_table for plot
    
    discounted_return = []
    avg_delay_all_epi = []
    print("flow:",args.flow_ns*3600,args.flow_ew*3600,',runs:', args.runs, ",road_type:", args.road_type, ",is fixed time:", args.fixed_time, ",state:", args.state,
    ",reward:", args.reward, ",min_green:", args.min_green,",Generate_cars_step:", args.Generate_cars_step, ",episode_time:", args.episode_time,",decay:",args.decay)
    
    avg_delay_all_epi = []

    # lst_path = ["500.0_900.0_ucb_2.0","600.0_900.0_e-greedy","600.0_900.0_ucb_2.0","700.0_700.0_e-greedy","700.0_700.0_ucb_2.0"]

    floder_path = "./outputs/constant/flow/ql/state_6ele_den_tslg_flow_{}/q_table/".format(args.test_path)
    print("floder_path:",floder_path)
    q_table= read_table(floder_path)
    if args.norm:
        q_table = norm_q_table(q_table, 75)
    # mean = load_file(floder_path, "mean")
    # std = load_file(floder_path, "std")   

    actions = {'0': 0} # action initialization
    # num_in_q = 0
    for epi in range(args.runs):
        num_in_q = 0
        print("epi:", epi)
        #every episode, generate cars, save dict of cars, ini env, ini ts(use different veh dict), ini agent
        NS_veh_lst = []
        EW_veh_lst = []
        d, vehNr = generate_routefile(args.flow_ns,args.flow_ew,epi,args.Generate_cars_step, args.road_type,args.seeds)
        
        file_name = str(Path('nets/single-intersection/'+args.road_type+'/single-intersection_{}.rou.xml'.format(epi)).absolute())
        if args.tune_q:
            episode_output_path = floder_path+"test/tune_q"
        elif args.cluster:
            episode_output_path = floder_path+"test/cluster"
        else:
            episode_output_path = floder_path+"test/original_q"

        env = SumoEnvironment(net_file=str(Path('nets/single-intersection/'+args.road_type+'/single-intersection.net.xml').absolute()),
                        route_file=file_name,
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
                        test = args.test,
                        episode_output_path = episode_output_path,
                        new_state = args.new_state)
    
        initial_states = env.reset()
        ql_agents = {ts: QLAgent(starting_state=env.encode(initial_states[ts], ts),
                                state_space=env.observation_space,
                                action_space=env.action_space,
                                alpha=args.alpha,
                                gamma=args.gamma,
                                path = q_table,
                                exploration_strategy=EpsilonGreedy(initial_epsilon=args.epsilon, min_epsilon=args.min_epsilon, decay=args.decay),
                                fixed_time=args.fixed_time) for ts in env.ts_ids}
        done = {'__all__': False}
        if args.fixed:
            while not done['__all__']:
                _, _, done, _ = env.step({})
        else:
            while not done['__all__']:
                
            # while traci.simulation.getMinExpectedNumber() > 0:
                s, r, selected_action, done, _ = env.step(action=actions) # take action, get reward, get next state:s
                
                for agent_id in ql_agents.keys():
                    next_state = env.encode(s[agent_id], agent_id) 
                    if args.cluster:
                        if next_state in q_table and freq_q_table[next_state] >100:
                            num_in_q += 1
                            actions = {ts: np.argmax(q_table[next_state]) for ts in ql_agents.keys()}
                        else:
                            close_state = find_close_center(next_state, q_center, mean,std)
                            actions = {ts: q_center[close_state] for ts in ql_agents.keys()}
                    elif args.tune_q: 
                        # close_state = find_close(next_state, tune_q)
                        if next_state in tune_q:
                            actions = {ts: np.argmax(tune_q[next_state]) for ts in ql_agents.keys()}
                        else:
                            # print("--------not in q----------")
                            close_state = find_close_center(next_state,q_norm_tuned, mean,std)
                            actions = {ts:  q_norm_tuned[close_state] for ts in ql_agents.keys()}
                            # print("current time:",env.sumo.simulation.getTime())
                            # print("state from environment:",next_state)
                            # print("close state in tuned q table:", close_state,"q values",q_norm_tuned[close_state])
                    else: # original q table
                        if args.norm:
                            next_state = norm_state(next_state,75)
                        close_state,num_in_q = find_close(next_state, q_table,num_in_q)
                        actions = {ts: np.argmax(q_table[close_state]) for ts in ql_agents.keys()}
                        # print("time:",env.sumo.simulation.getTime())
                        # print("state:",next_state)
                        # print("close state:",close_state)
                        # print("q values:", q_table[close_state])
                        # print("freq:", freq_q_table[close_state])
                        if not close_state == next_state:
                            print("time:",env.sumo.simulation.getTime())
                            print("state:",next_state)
                            print("close state:",close_state)
                            print("q values:", q_table[close_state])
                            # print("freq:", freq_q_table[close_state])
                            # breakpoint()
                if num_in_q > 100:
                    break

                # if env.sumo.simulation.getTime() > 1400:
                #     breakpoint()
        print('num_not_in_q:',num_in_q)
        env.close()
        


                
                




""" mg 13s, 0.7gamma
    q_table = {'[0, 0, 0, 0, 0]': [-47.85968141398271, -41.762995006253846], '[0, 1, 1, 0, 0]': [-82.45489057614901, -73.10439681888397], '[1, 1, 1, 0, 0]': [-74.7994642660451, -72.29413559052983], '[1, 2, 1, 0, 0]': [-96.95151554861737, -96.26557914654829], '[0, 2, 1, 0, 0]': [-92.9177029657853, -85.88263230476456], '[0, 2, 2, 0, 1]': [-132.8424223608281, -130.87468535002867], '[0, 1, 2, 0, 1]': [-133.62465376847996, -123.98710484446796], '[1, 1, 2, 0, 1]': [-116.5173180643074, -95.46877064214597], '[1, 2, 1, 0, 1]': [-111.6175391674081, -104.6500724408906], 
    '[0, 2, 2, 0, 0]': [-112.01088182014735, -101.33774670798681], '[0, 1, 2, 0, 0]': [-109.45110306335094, -99.12039228624775], '[1, 2, 2, 0, 1]': [-129.8132514669586, -123.54869434225675], '[0, 2, 1, 0, 1]': [-109.14982284358203, -103.52238878444282], '[1, 1, 1, 0, 1]': [-91.32428885501909, -79.322280820394], '[1, 2, 2, 0, 2]': [-174.58169561923637, -144.26082824746828], '[0, 1, 2, 0, 2]': [-161.570884222231, -134.96907531748562], '[1, 1, 2, 0, 2]': [-53.73020453009839, -138.1334964035626], '[0, 2, 2, 0, 2]': [-185.28463730587657, -167.33035211695508], '[1, 0, 2, 0, 1]': [-88.68331410114394, -87.61572620460984], '[0, 0, 2, 0, 1]': [-120.97783017400545, -114.25759127995386], 
    '[1, 2, 2, 1, 1]': [-122.76055591454084, -120.15783487814439], '[0, 2, 2, 1, 1]': [-144.89738474971512, -89.99531025475434], '[0, 1, 1, 0, 1]': [-111.04261247670732, -94.83987587759772], '[1, 2, 1, 1, 0]': [-50.32837560008811, -77.89294912450197], '[0, 2, 2, 1, 0]': [-117.24319099368577, 0], '[1, 2, 2, 0, 0]': [-98.45933236046585, -89.79469224877263], '[1, 2, 2, 1, 0]': [-66.9206354522948, -76.17456197455854], '[1, 2, 1, 1, 1]': [-73.98343258699225, -95.91299189658923], '[0, 2, 1, 1, 1]': [-118.37070110884753, 0], '[1, 1, 2, 0, 0]': [-85.35774349555894, -72.19402993258991], '[1, 1, 2, 1, 1]': [-67.0700344224526, -102.99545065766131], '[0, 2, 1, 1, 0]': [-94.4323387313, 0], 
    '[0, 0, 1, 0, 1]': [-103.61386449561911, -93.74551960082601], '[0, 0, 0, 0, 1]': [-91.85465162988993, -92.51921946520348], '[1, 0, 1, 0, 1]': [-60.66928330226061, -62.356989432606746], '[1, 0, 2, 0, 2]': [0, -110.29373527426189], '[0, 1, 2, 1, 1]': [-140.65204082946292, -115.19355506838835]}
    """
    # """ mg 6s, 0,7 original
    # q_table = {"[0, 0, 0, 0, 0]": [-39.01465331765534, -38.37265139826965], "[1, 0, 1, 0, 0]": [-52.98913798528017, -50.04219850688811], "[1, 1, 1, 0, 0]": [-76.17502808415945, -70.89490576575125], "[0, 1, 1, 0, 0]": [-77.94681399563225, -73.73128598937757], "[0, 1, 2, 0, 0]": [-96.46855013182562, -94.28152468467644], "[0, 1, 2, 0, 1]": [-123.27590053778492, -116.9263006788186], "[0, 2, 2, 0, 1]": [-130.07423714731735, -127.71402878543465], "[1, 1, 2, 0, 1]": [-121.96129301854069, -90.26569853210162], "[1, 2, 2, 0, 1]": [-129.99012423866463, -110.63506890821135], "[0, 2, 2, 0, 0]": [-104.00471938443633, -95.96646210128928], "[1, 2, 2, 0, 0]": [-99.00368101923554, -82.41362915437539], "[1, 2, 1, 0, 0]": [-89.72202536470732, -89.77355010160315], "[0, 2, 1, 0, 1]": [-100.5187311187643, -99.13042031869557], "[0, 1, 1, 0, 1]": [-103.69106469248788, -95.86801257971267], "[0, 0, 2, 0, 1]": [-123.4261487859359, -118.24866294572773], "[1, 2, 2, 0, 2]": [-219.35681222558424, -140.0173413705707], "[0, 2, 2, 0, 2]": [-197.59271783368655, -183.85412906089644], "[0, 1, 2, 0, 2]": [-208.16567988906075, -158.5283406738765], "[0, 0, 2, 0, 2]": [-125.1710951632497, -129.01114929952203], "[1, 1, 2, 0, 2]": [-170.50639947899683, -131.737283464448], "[0, 1, 2, 0, 3]": [-359.19250380830294, -336.0786061000137], "[0, 2, 2, 0, 3]": [-365.74341197747987, -292.3322866360585], "[1, 2, 2, 0, 3]": [-329.0368642748542, -326.4476886197854], "[0, 0, 2, 0, 3]": [-73.52368912101437, -71.81646846040242], "[1, 1, 2, 0, 3]": [-254.8417149558273, -300.47756533593895], "[1, 1, 2, 0, 4]": [-127.37947887685365, -363.9665988006036], "[1, 2, 2, 0, 4]": [-268.0735694727857, -391.284182398197], "[0, 1, 2, 0, 4]": [-337.4838913676284, -308.9876840211113], "[0, 2, 2, 0, 4]": [-367.972594579044, -261.55723861123425], "[1, 1, 2, 0, 5]": [0, -36.99263787557202], "[1, 2, 2, 1, 4]": [-24.323899669350716, 0], "[0, 2, 2, 1, 4]": [-71.46862038675019, 0], "[1, 2, 2, 1, 3]": [-17.661673008387947, -29.200602901118447], "[0, 2, 2, 1, 3]": [-97.88483761154859, 0], "[1, 2, 2, 1, 2]": [-101.32899183637876, -152.9131185916593], "[0, 2, 2, 1, 2]": [-195.9848406940211, -91.27362968509105], "[1, 0, 2, 0, 4]": [0, -61.79387477320054], "[1, 2, 2, 1, 1]": [-153.66737056772115, -126.9556243930968], "[0, 2, 2, 1, 1]": [-140.913562271136, -153.00694109300449], "[1, 2, 1, 0, 1]": [-110.46521676585155, -103.9195390649513], "[0, 2, 1, 0, 0]": [-93.17493206229979, -79.61726454131146], "[0, 0, 2, 0, 0]": [-92.25050914882542, -89.93313112944115], "[0, 2, 1, 1, 1]": [-132.01162075454383, -110.5750399240704], "[0, 2, 1, 1, 0]": [-91.83185770978884, 0], "[1, 0, 2, 0, 1]": [-100.29163903986898, -92.00142985442895], "[1, 0, 2, 0, 2]": [0, -122.33209288929277], "[1, 0, 2, 0, 3]": [0, -32.49172439943909], "[0, 0, 0, 0, 1]": [-87.6923134387176, -78.56877375949023], "[0, 0, 0, 0, 2]": [-9.050550606578064, -9.838933468981859], "[0, 0, 1, 0, 1]": [-100.24207861144089, -93.58532091740922], "[1, 1, 2, 0, 0]": [-80.70351758362135, -69.82809165448752], "[1, 1, 2, 1, 1]": [-138.4601718629637, -124.84037898875269], "[1, 1, 1, 0, 1]": [-94.1145242503973, -82.1369251488169], "[1, 2, 2, 1, 0]": [-78.38914965081374, -93.95939868426842], "[1, 2, 1, 1, 0]": [-70.16334892968086, -95.97346072304401], "[0, 2, 2, 1, 0]": [-117.84883788337537, -35.30930409104719], "[1, 2, 1, 1, 1]": [-134.3372477909652, -148.67108597606986], "[0, 1, 2, 1, 1]": [-151.50251155215526, -148.38623113897566], "[1, 2, 1, 0, 2]": [-114.00147780740889, -116.11590096089539], "[0, 2, 1, 0, 2]": [-165.20601891625205, -128.97715812026075], "[1, 1, 1, 1, 1]": [-58.213727149841496, -63.71582999945174], "[0, 1, 2, 1, 0]": [-50.502712840583, -39.910679254461414], "[1, 0, 0, 0, 1]": [-58.67877347454043, -63.860488910686314], "[1, 0, 0, 0, 0]": [0, -17.58996864034597], "[1, 0, 2, 0, 0]": [-47.09659538269658, -53.475302035389404], "[0, 0, 1, 0, 0]": [-65.99387522281833, -64.36050608193754], "[1, 0, 1, 0, 1]": [-71.6818879441343, -63.23002203143387], "[1, 2, 0, 0, 0]": [-41.600727196825815, -45.04042144915688], "[0, 1, 1, 1, 1]": [-81.48392941040049, -24.80375086203371], "[1, 1, 1, 0, 2]": [0, -34.36825331345399], "[0, 2, 0, 0, 0]": [-60.16834358079337, 0], "[1, 1, 2, 1, 0]": [-23.728913028474892, -40.187566152726966], "[1, 0, 0, 0, 2]": [0, -48.608203535726474], "[0, 1, 0, 0, 0]": [-40.51073054579377, -6.93899189327101], "[1, 1, 1, 1, 0]": [-37.64899009948202, -43.21785271977713], "[0, 2, 0, 0, 1]": [-64.11590678266491, -5.10034196834214], "[0, 1, 1, 0, 2]": [-87.8146935279871, -82.33356508566135], "[1, 2, 0, 1, 0]": [-9.526286657470997, -5.469535758669922], "[0, 2, 0, 1, 0]": [-7.840103058572847, 0], "[0, 1, 0, 0, 1]": [-7.096955280584945, 0], "[1, 1, 2, 1, 2]": [0, -30.731055457498666], "[1, 2, 1, 1, 2]": [-19.442038817561624, -6.948712555571989], "[0, 2, 1, 1, 2]": [-56.20592587555595, 0], "[1, 2, 1, 2, 0]": [-9.87435299349729, 0], "[0, 2, 1, 2, 0]": [-9.722786327284394, 0], "[1, 1, 0, 0, 0]": [-0.7482103259957095, 0], "[0, 2, 2, 2, 1]": [-25.21898557800359, 0]}
    # """
    # delete mg 6s
    # q_table = {'[0, 0, 0, 0, 0]': [-39.01465331765534, -38.37265139826965], '[1, 1, 1, 0, 0]': [-76.17502808415945, -70.89490576575125], '[0, 1, 1, 0, 0]': [-77.94681399563225, -73.73128598937757], '[0, 1, 2, 0, 0]': [-96.46855013182562, -94.28152468467644], '[0, 1, 2, 0, 1]': [-123.27590053778492, -116.9263006788186], '[0, 2, 2, 0, 1]': [-130.07423714731735, -127.71402878543465], '[1, 1, 2, 0, 1]': [-121.96129301854069, -90.26569853210162], '[1, 2, 2, 0, 1]': [-129.99012423866463, -110.63506890821135], '[0, 2, 2, 0, 0]': [-104.00471938443633, -95.96646210128928], '[1, 2, 2, 0, 0]': [-99.00368101923554, -82.41362915437539], '[1, 2, 1, 0, 0]': [-89.72202536470732, -89.77355010160315], '[0, 2, 1, 0, 1]': [-100.5187311187643, -99.13042031869557], '[0, 1, 1, 0, 1]': [-103.69106469248788, -95.86801257971267], '[0, 0, 2, 0, 1]': [-123.4261487859359, -118.24866294572773], '[1, 2, 2, 0, 2]': [-219.35681222558424, -140.0173413705707], '[0, 2, 2, 0, 2]': [-197.59271783368655, -183.85412906089644], '[0, 1, 2, 0, 2]': [-208.16567988906075, -158.5283406738765], '[1, 1, 2, 0, 2]': [-170.50639947899683, -131.737283464448], '[0, 2, 2, 0, 3]': [-365.74341197747987, -292.3322866360585], '[1, 2, 2, 0, 3]': [-329.0368642748542, -326.4476886197854], '[1, 2, 2, 1, 1]': [-153.66737056772115, -126.9556243930968], '[0, 2, 2, 1, 1]': [-140.913562271136, -153.00694109300449], '[1, 2, 1, 0, 1]': [-110.46521676585155, -103.9195390649513], '[0, 2, 1, 0, 0]': [-93.17493206229979, -79.61726454131146], '[0, 2, 1, 1, 1]': [-132.01162075454383, -110.5750399240704], '[0, 2, 1, 1, 0]': [-91.83185770978884, 0], '[1, 0, 2, 0, 1]': [-100.29163903986898, -92.00142985442895], '[1, 0, 2, 0, 2]': [0, -122.33209288929277], '[0, 0, 1, 0, 1]': [-100.24207861144089, -93.58532091740922], '[1, 1, 2, 0, 0]': [-80.70351758362135, -69.82809165448752], '[1, 1, 2, 1, 1]': [-138.4601718629637, -124.84037898875269], '[1, 1, 1, 0, 1]': [-94.1145242503973, -82.1369251488169], '[1, 2, 2, 1, 0]': [-78.38914965081374, -93.95939868426842], '[1, 2, 1, 1, 0]': [-70.16334892968086, -95.97346072304401], '[0, 2, 2, 1, 0]': [-117.84883788337537, -35.30930409104719], '[1, 2, 1, 1, 1]': [-134.3372477909652, -148.67108597606986], '[0, 1, 2, 1, 1]': [-151.50251155215526, -148.38623113897566], '[1, 0, 2, 0, 0]': [-47.09659538269658, -53.475302035389404], '[1, 0, 1, 0, 1]': [-71.6818879441343, -63.23002203143387]}
   
    #13mg
    # q_center = {'[-1.2800871473928055e-13, 0.9618034524529278, 2.0000000000000453, 0.009728952150230785, 1.0527331920048804]': 1, '[1.0000000000001152, 2.000000000000063, 1.0000000000000617, -3.95516952522712e-15, 1.0010841526786314]': 1, '[0.7500508026823631, 1.9999999999999767, 2.000000000000018, 5.773159728050814e-15, 2.0014224751067324]': 1, '[0.9999999999999478, 1.9999999999999485, 2.0000000000000573, 2.3175905639050143e-15, 1.0000000000000369]': 1, '[1.000000000000091, 0.9533554411379007, 2.000000000000045, 0.009904878261625644, 1.0000000000000173]': 1, '[7.904787935331115e-14, 1.9999999999999942, 0.9968276708964819, 0.20180106426525257, 3.985700658404312e-14]': 1, '[1.000000000000129, 1.92063118228926, 0.9984926990108927, 0.08407913330195055, 5.1736392947532295e-14]': 1, '[-2.5934809855243657e-13, 1.9999999999999292, 2.000000000000061, 2.058075931898884e-14, 1.000000000000043]': 1, '[1.0000000000000926, 2.000000000000049, 2.000000000000047, 0.06611333714939086, 5.1514348342607263e-14]': 1, '[1.149080830487037e-13, 2.0000000000000244, 0.9976818633403809, -5.509481759702339e-15, 1.002318136659685]': 1, '[-1.587618925213974e-14, 1.999999999999981, 1.9999999999999858, 1.0000000000000004, 0.7981447124304254]': 1, '[5.828670879282072e-14, 1.9999999999999762, 2.000000000000022, 6.897260540483785e-15, 4.163336342344337e-14]': 1, '[-8.548717289613705e-15, 0.7987385321100674, 0.8749999999999994, 8.326672684688674e-17, 0.6510894495412791]': 1, '[0.999999999999988, 0.9508584961515659, 1.9999999999999896, 0.0017761989342787998, 2.001480165778604]': 0, '[-1.4765966227514582e-14, 1.9979971387696498, 1.0000000000000315, 0.9999999999999659, 1.0042918454935548]': 1, '[-8.215650382226158e-15, 0.9523551705468113, 1.9999999999999976, 0.0016242555495410704, 4.440892098500626e-16]': 1, '[0.9999999999999984, 1.999999999999985, 1.9999999999999938, 1.000000000000024, 1.0304847576211933]': 1, '[1.000000000000003, 0.9608197709463311, 1.9999999999999996, 0.003616636528030834, -2.9976021664879227e-15]': 1, '[0.999999999999995, 0.9205512768544567, 0.9679773003648234, 0.008917713822455778, 1.004458856911232]': 1, '[0.9999999999999873, 1.9999999999999793, 1.0000000000000284, 0.9999999999999752, 1.003400309119007]': 0}
    
    # 6 mg cluster
"""
    q_center = {'[-1.865174681370263e-13, 1.9999999999999627, 0.9987529098770783, -1.199040866595169e-14, 1.0042401064183952]': 1, 
    '[0.9999999999999997, 1.9999999999999738, 1.9999999999999425, 0.07744191856106913, 1.4976908602193362e-13]': 1, '[0.9999999999998019, 2.000000000000039, 2.0000000000001323, 0.05383655089391011, 1.0000000000001092]': 1,
     '[0.9999999999999866, 0.9542076141713959, 1.9999999999999543, 0.008712527871820572, 1.0000000000000284]': 1, '[0.9999999999997468, 2.000000000000091, 1.0000000000002487, -1.952604744559494e-14, 1.0021283873666356]': 1, 
     '[-1.9906298831529057e-13, 2.0000000000000004, 2.0000000000000506, -1.317002062961592e-14, 1.0000000000000646]': 1, '[-1.0236256287043943e-13, 1.9999999999999916, 0.996397453083164, 0.15440683646112502, 1.2423395645555502e-13]': 1, 
     '[0.9999999999998934, 1.9999999999999536, 0.998238176512845, 0.07950228486481863, 1.5087930904655877e-13]': 0, '[1.0000000000000122, 0.9171038824763712, 0.979538300104883, 0.00944386149003136, 1.0034102833158358]': 1, 
     '[-2.3581137043038325e-13, 0.9612482646226057, 2.0000000000000595, 0.009959558157766515, 1.0000000000000773]': 1, '[1.1546319456101628e-14, 1.9999999999999925, 2.0, 1.0004263483266056, 0.7183969302920403]': 0, 
     '[1.0000000000000113, 1.999999999999989, 0.9999999999999618, 1.0003229974160563, 1.000645994832033]': 0, '[1.2878587085651816e-14, 1.9999999999999893, 1.9999999999999942, 3.608224830031759e-16, 3.2529534621517087e-14]': 1, 
     '[0.7396465312582553, 1.9999999999999838, 1.9999999999999827, 0.029807438670536014, 2.1276707992615247]': 1, 
     '[1.3211653993039363e-14, 1.9918391139609306, 0.9999999999999563, 1.0002914602157316, 1.000874380647033]': 1, '[1.0103029524088925e-14, 0.8514672686230409, 0.9074492099322626, -4.163336342344337e-17, 0.6532731376975094]': 1, 
     '[1.00000000000001, 0.9664694280079156, 0.9956607495068793, 0.010256410256409249, -9.769962616701378e-15]': 1, 
     '[0.7093333333333343, 0.9516666666666879, 1.9953333333333405, 0.0006666666666648169, 2.1410000000000804]': 1, '[1.00000000000001, 0.950437317784278, 2.000000000000012, 0.005414410662223326, -9.992007221626409e-15]': 1, 
     '[1.0658141036401503e-14, 0.9563206577595119, 2.0000000000000093, 0.007194244604317876, -8.992806499463768e-15]': 1}
    """


        