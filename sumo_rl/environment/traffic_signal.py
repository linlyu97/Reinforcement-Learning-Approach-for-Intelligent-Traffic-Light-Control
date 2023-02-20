import os
import sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import traceback
import numpy as np
from gym import spaces

import json
import math



class TrafficSignal:
    """
    This class represents a Traffic Signal of an intersection
    It is responsible for retrieving information and changing the traffic phase using Traci API

    IMPORTANT: It assumes that the traffic phases defined in the .net file are of the form:
        [green_phase, yellow_phase, green_phase, yellow_phase, ...]
    Currently it is not supporting all-red phases (but should be easy to implement it).

    Default observation space is a vector R^(#greenPhases + 2 * #lanes)
    s = [current phase one-hot encoded, density for each lane, queue for each lane]
    You can change this by modifing self.observation_space and the method _compute_observations()

    Action space is which green phase is going to be open for the next delta_time seconds
    """
    def __init__(self, env, ts_id, delta_time, yellow_time, min_green, max_green, begin_time, sumo, epi, road_type, veh_info,gamma):
        self.id = ts_id
        self.is_init_phase = True
        self.env = env
        self.delta_time = delta_time
        self.yellow_time = yellow_time
        self.all_red_time = 2
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.is_yellow = False
        self.is_all_red = False
        self.time_since_last_phase_change = 0
        self.next_action_time = begin_time
        self.last_measure = 0.0
        self.last_reward = None
        self.delay = None
        self.sumo = sumo
        self.epi = epi
        self.step_ = 0
        self.discounted_r = 0
        self.veh_delay = {}
        self.veh_delay_NS = {}
        self.veh_delay_EW = {}
        self.road_type = road_type
        self.EW_veh = 0
        self.NS_veh = 0
        self.geh_delay = {}
        self.delay_within_delta_time = []
        self.delay_within_delta_time_ew = []
        self.delay_within_delta_time_ns = []
        self.acc_pressure_with_delta_time = []
        self.acc_pressure_with_delta_time_no_abs = []
        self.std_ew = []
        self.std_ns = []
        self.gamma = gamma
        self.discharge_veh = []

        
        # with open("sumo_rl/environment/veh_info/veh_info_{}".format(self.epi), "r") as f:
        #     veh_info_dict = json.load(f)
        self.veh_info = veh_info

        self.build_phases()

        self.lanes = list(dict.fromkeys(self.sumo.trafficlight.getControlledLanes(self.id)))  # Remove duplicates and keep order
        self.out_lanes = [link[0][1] for link in self.sumo.trafficlight.getControlledLinks(self.id) if link]
        self.out_lanes = list(set(self.out_lanes))
        self.lanes_lenght = {lane: self.sumo.lane.getLength(lane) for lane in self.lanes}

        self.observation_space = spaces.Box(low=np.zeros(self.num_green_phases+2*len(self.lanes), dtype=np.float32), high=np.ones(self.num_green_phases+2*len(self.lanes), dtype=np.float32))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_green_phases),                       # Green Phase
            spaces.Discrete(2),                                           # Binary variable active if min_green seconds already elapsed
            *(spaces.Discrete(10) for _ in range(2*len(self.lanes)))      # Density and stopped-density for each lane
        ))
        self.action_space = spaces.Discrete(self.num_green_phases)

    def build_phases(self):
        # breakpoint()
        phases = self.sumo.trafficlight.getCompleteRedYellowGreenDefinition(self.id)[0].phases
        # print('phases:', phases)
        # Returns a list of Logic objects
        if self.env.fixed_ts:
            # fixed_ts: (bool) If true, 
            # it will follow the phase configuration in the route_file and ignore the actions.
            self.num_green_phases = len(phases)//2  # Number of green phases == number of phases (green+yellow) divided by 2
            return

        self.green_phases = []
        self.yellow_dict = {}
        for phase in phases:
            state = phase.state
            # print("phase.state:", phase.state)
            # print("state.count:", state.count)
            # print("state.count(r):",state.count('r') )
            if 'y' not in state and (state.count('r') + state.count('s') != len(state)):
                self.green_phases.append(self.sumo.trafficlight.Phase(60, state))
        self.num_green_phases = len(self.green_phases)
        self.all_phases = self.green_phases.copy()

        for i, p1 in enumerate(self.green_phases):
            # print("self.green_phase:", self.green_phases)
            for j, p2 in enumerate(self.green_phases):
                # print("i, p1, j, p2:", i,p1,j,p2)
                if i == j: continue
                yellow_state = ''
                for s in range(len(p1.state)):
                    # print("p1.state:", p1.state)
                    if (p1.state[s] == 'G' or p1.state[s] == 'g') and (p2.state[s] == 'r' or p2.state[s] == 's'):
                        yellow_state += 'y'
                    else:
                        yellow_state += p1.state[s]
                self.yellow_dict[(i,j)] = len(self.all_phases)
                # print('self.yellow_dict:', self.yellow_dict)
                # breakpoint()
                self.all_phases.append(self.sumo.trafficlight.Phase(self.yellow_time, yellow_state))
                # Phase, class, _init__(self, duration, state, minDur=-1, maxDur=-1, next=(), name='')
                # Initialize self.  See help(type(self)) for accurate signature.

        self.all_phases.append(self.sumo.trafficlight.Phase(2, 'rrrr'))
        programs = self.sumo.trafficlight.getAllProgramLogics(self.id)
        logic = programs[0]
        logic.type = 0
        logic.phases = self.all_phases
        self.sumo.trafficlight.setProgramLogic(self.id, logic)
        self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[0].state)
        # print("self.all_phases:", self.all_phases)
        # breakpoint()

    @property
    def time_to_act(self):
        return self.next_action_time == self.env.sim_step
    
    def update(self):
        # traceback.print_stack()
        self.time_since_last_phase_change += 1
        if self.is_yellow and self.time_since_last_phase_change == self.yellow_time:
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[4].state) #set red
            # print('yellow time:', self.all_phases[4].state)
            # print('current time red:', self.env.sim_step)
            self.is_yellow = False
            self.is_all_red = True
            # print("red",self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[4].state))
        
        if self.is_all_red and self.time_since_last_phase_change == (self.all_red_time + self.yellow_time):
            #self.sumo.trafficlight.setPhase(self.id, self.green_phase)
            
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state) #set next green
            # print('yellow time:',self.id, self.all_phases[self.green_phase].state)
            # print('current time green start:', self.env.sim_step)
            self.is_all_red = False
            # print("all red is done")
            # print("tiem step:",self.env.sim_step)
            # breakpoint()
            # print("yellow",self.all_phases[self.green_phase].state)
            
    def set_next_phase(self, new_phase):
        # new_phase = actions: a chosed action from main function
        
        # Sets what will be the next green phase and sets yellow phase if the next phase is different than the current

        # :param new_phase: (int) Number between [0..num_green_phases] 
        
        new_phase = int(new_phase)
        # print("__________________enter function__________________")
        # print('action, new_phase:', new_phase)
        # print('current_phase:', self.green_phase)
        # self.green_phase = 0 in initialization, new_phase:0, initial, random choose, argmax, all 0 at first
        # self.min_green = 10
        # 'do not change phase or phase duration is less than yellow + min green'
        # if self.green_phase == new_phase or self.time_since_last_phase_change < self.yellow_time + self.min_green + self.all_red_time:
        if self.green_phase == new_phase and self.time_since_last_phase_change < self.max_green+5: # max green is self.max_green
            # print("set phase, self.time_since_last_phase_change:",self.time_since_last_phase_change)
            # if self.time_since_last_phase_change == 0:
            # print('--------if-------')
            # breakpoint()
            #     print("self.time_since_last_phase_change:",self.time_since_last_phase_change)
            
            # print('self.time_since_last_phase_change:', self.time_since_last_phase_change)
            """print("green_phase:",self.green_phase)
            print("new_phase:",new_phase)
            print("tiem step:",self.env.sim_step)
            breakpoint()"""
          
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.green_phase].state)
            """
            time_since_last_phase_change contains red and yellow
            self.green_phase: current phase index in all phases
            [Phase(duration=60, state='GrGr', minDur=-1, maxDur=-1, next=()), NS
            Phase(duration=60, state='rGrG', minDur=-1, maxDur=-1, next=()), EW
            Phase(duration=3, state='yryr', minDur=-1, maxDur=-1, next=()), 
            Phase(duration=3, state='ryry', minDur=-1, maxDur=-1, next=()), 
            Phase(duration=2, state='rrrr', minDur=-1, maxDur=-1, next=())]
            """
            # print('self.all_phases[self.green_phase]:', self.all_phases[self.green_phase])
            # print('self.all_phases[self.green_phase].state:', self.all_phases[self.green_phase].state)
            # self.next_action_time = self.env.sim_step + 5
            self.next_action_time = self.env.sim_step + self.delta_time
            indicator = False # not change
            current_phase = self.green_phase

            offset = 0 if self.is_init_phase else self.yellow_time + 2
            phase_duration = self.time_since_last_phase_change - offset

            # print('current time:', self.env.sim_step)
            # print('phase_duration', phase_duration)
            # print('----------end------')
            return self.green_phase, indicator, phase_duration, current_phase
        
        
        else:
            # 'change phase'
            #self.sumo.trafficlight.setPhase(self.id, self.yellow_dict[(self.green_phase, new_phase)])  # turns yellow
             # next: next line; list: current location
            
            if self.green_phase == 1 and new_phase == 1:
                new_phase = 0
                print("force to change to 1 0")
            elif self.green_phase == 0 and new_phase == 0:
                new_phase=1
                print("force to change to 0 1")
            """print("green_phase:",self.green_phase)
            print("new_phase:",new_phase)
            print("tiem step:",self.env.sim_step)
            breakpoint()"""
            self.sumo.trafficlight.setRedYellowGreenState(self.id, self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            """
            self.yellow_dict: {(0, 1): 2, (1, 0): 3}
            set yellow time, according to current phase and next phase index
            """
            # print('--------else--------')
            # print('self.time_since_last_phase_change:', self.time_since_last_phase_change)
            # print('self.time_since_last_phase_change:', self.time_since_last_phase_change)
            # print('self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]]:', self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]])
            # print('self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state', self.all_phases[self.yellow_dict[(self.green_phase, new_phase)]].state)
            # print('current time:', self.env.sim_step)
            
            """ 3s yellow time, 2s all red time"""
            offset = 0 if self.is_init_phase else self.yellow_time + 2
            phase_duration = self.time_since_last_phase_change - offset
            current_phase = self.green_phase
            self.green_phase = new_phase
            self.next_action_time = self.env.sim_step + self.delta_time + 5
            # breakpoint()
            # self.env.sim_step: get current simulation second on SUMO
            self.is_init_phase = False
            self.is_yellow = True
            self.time_since_last_phase_change = 0
            indicator = True # change phase
            # print('current time:', self.env.sim_step)
            # if phase_duration > 5:
            # print('phase_duration', phase_duration)
            return new_phase, indicator, phase_duration, current_phase


    def compute_observation(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        density = self.get_lanes_density()
        # print('density:', density)
        queue = self.get_lanes_queue()
        # print('queue:', queue)
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        # print('observation:', observation)
        return observation

    def compute_observation_paper(self, phase_duration):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        phase_duration = phase_duration
        avg_delay = self.get_avg_delay()
        incoming_veh, outcoming_veh = self.get_state_in_out()
        density = self.get_lanes_density()
        # print('density:', density)
        queue = self.get_lanes_queue()
        # print('queue:', queue)
        observation = np.array(phase_id + [phase_duration, incoming_veh, outcoming_veh, avg_delay], dtype=np.float32)
        # print('observation:', observation)
        return observation
    
    def compute_state_num_veh(self):
        upstream = [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes]
        downstream = [self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes]
        
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        observation = np.array(phase_id + upstream + downstream, dtype=np.float32)
        return observation

        
    def compute_observation_9(self):
        
        # "no min green"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        density = self.get_lanes_density()
        # print('density:', density)
        queue = self.get_lanes_queue()
        # print('queue:', queue)
        observation = np.array(phase_id + density + queue, dtype=np.float32)
        # print("self.time_since_last_phase_change:",self.time_since_last_phase_change)
        # print('observation:', observation)
        # breakpoint()
        return observation
    
    def compute_observation_queue(self):
        # "no min green"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        queue = self.get_lanes_queue()
        # print('queue:', queue)
        observation = np.array(phase_id + queue, dtype=np.float32)
        # print("self.time_since_last_phase_change:",self.time_since_last_phase_change)
        # print('observation:', observation)
        # breakpoint()
        return observation
        
    def compute_observation_queue_tslg(self):
        # "no min green"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        queue = self.get_lanes_queue()
        # print('queue:', queue)
        
        offset = 0 if self.is_init_phase else self.yellow_time + 2
        phase_duration = self.time_since_last_phase_change - offset
        norm_phase_duration = phase_duration/(self.max_green)
        observation = np.array(phase_id + queue + [norm_phase_duration], dtype=np.float32)
        # print("self.time_since_last_phase_change:",self.time_since_last_phase_change)
        # print('observation:', observation)
        # breakpoint()
        return observation

    def compute_observation_9_tslg(self):
        # "no min green"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        density = self.get_lanes_density()
        # print('density:', density)
        queue = self.get_lanes_queue()
        # print('queue:', queue)
        offset = 0 if self.is_init_phase else self.yellow_time + 2
        phase_duration = self.time_since_last_phase_change - offset
        norm_phase_duration = phase_duration/(self.max_green)
        observation = np.array(phase_id + density + queue + [norm_phase_duration], dtype=np.float32)
        # print('observation:',observation )
        # breakpoint()
        return observation

    def compute_observation_6ele_den_tslg(self):
        # "no min green"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)

        N_den = 0
        E_den = 0
        S_den = 0
        W_den = 0

        # print("self.env.:density_lst_separate", self.env.density_lst_separate)
        # breakpoint()
        for i in range(1, len(self.env.density_lst_separate)):
            N_den += self.env.density_lst_separate[i][0]
            E_den += self.env.density_lst_separate[i][1]
            S_den += self.env.density_lst_separate[i][2]
            W_den += self.env.density_lst_separate[i][3]
        if len(self.env.density_lst_separate) == 0:
            density = [0,0,0,0]
        else:
            density = [min(1,N_den/(len(self.env.density_lst_separate)-1)),min(1,E_den/(len(self.env.density_lst_separate)-1)),
            min(1,S_den/(len(self.env.density_lst_separate)-1)), min(1,W_den/(len(self.env.density_lst_separate)-1))]
        self.env.density_lst_separate = []
        
        observation = np.array(phase_id + density + [self.time_since_last_phase_change], dtype=np.float32)
        # print('observation:', observation)

        return observation

    def compute_observation_5(self):
        "no min green; combined NS and EW"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        density = self.get_lanes_density_combine()
        # print('density:', density)
        queue = self.get_lanes_queue_combine()
        # print('queue:', queue)
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        # print('observation:', observation)
        return observation

    def compute_observation_3_comb_queue(self):
        "no min green; combined NS and EW queue"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        queue = self.get_lanes_queue_combine()
        # print('queue:', queue)
        observation = np.array(phase_id + min_green + queue, dtype=np.float32)
        # print('observation:', observation)
        return observation

    def compute_observation_3_comb_density(self):
        "no min green; combined NS and EW density"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        density = self.get_lanes_density_combine()
        # print('density:', density)
        observation = np.array(phase_id + min_green + density, dtype=np.float32)
        # print('observation:', observation)
        return observation
    def compute_observation_4ele_den_tslg(self):
        "no min green; combined NS and EW density"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time + 2 else 1]
        assert min_green == [0] or min_green == [1]
        # print('min_green:', min_green)
        # breakpoint()
        # density = self.get_lanes_density_combine()
        NS_den = 0
        EW_den = 0
    
        for i in range(1, len(self.env.density_lst)):
            NS_den += self.env.density_lst[i][0]
            EW_den += self.env.density_lst[i][1]
        if len(self.env.density_lst) == 0:
            density = [0,0]
        else:
            density = [min(1,NS_den/(len(self.env.density_lst)-1)),min(1,EW_den/(len(self.env.density_lst)-1))]
        self.env.density_dict = {}
        self.env.density_lst = []
        # print('density:', density)
        observation = np.array(phase_id + min_green + density + [self.time_since_last_phase_change], dtype=np.float32)
        # print('observation:', observation)
        return observation
        
    def compute_observation_4_comb_avg_density(self):
        "no min green; combined NS and EW density"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time + 2 else 1]
        assert min_green == [0] or min_green == [1]
        # print('min_green:', min_green)
        # breakpoint()
        # density = self.get_lanes_density_combine()
        NS_den = 0
        EW_den = 0
        NS = []
        EW = []
        # print("self.env.density_dict:", self.env.density_dict)
        # breakpoint()
        for i in range(1, len(self.env.density_lst)):
            NS_den += self.env.density_lst[i][0]
            NS.append(self.env.density_lst[i][0])

            EW_den += self.env.density_lst[i][1]
            EW.append(self.env.density_lst[i][1])
        if len(self.env.density_lst) == 0:
            density = [0,0]
        else:
            density = [min(1,NS_den/(len(self.env.density_lst)-1)),min(1,EW_den/(len(self.env.density_dict)-1))]
        self.std_ew.append(np.std(np.array(EW)))
        self.std_ns.append(np.std(np.array(NS)))
        self.env.density_dict = {}
        self.env.density_lst = []
        # print('density:', density)
        observation = np.array(phase_id + min_green + density, dtype=np.float32)
        # print('observation:', observation)
        return observation

    def compute_observation_6_comb_avg_time_density(self):
        "no min green; combined NS and EW density"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time + 2 else 1]
        assert min_green == [0] or min_green == [1]
        # print('min_green:', min_green)
        # breakpoint()
        # density = self.get_lanes_density_combine()
        NS_den = 0
        EW_den = 0
        NS_time = 0
        EW_time = 0
        # print("self.env.density_dict:", self.env.density_dict)
        # print("self.env.cum_time:", self.env.cum_time)
        # breakpoint()
        for i in range(1, len(self.env.density_lst)):
            NS_den += self.env.density_lst[i][0]
            EW_den += self.env.density_lst[i][1]
            NS_time += self.env.cum_time[i][0]
            EW_time += self.env.cum_time[i][1]
        if len(self.env.density_lst) == 0:
            density = [0,0]
        elif len(self.env.density_lst) == 1:
            density = self.env.density_lst[0]
        else:
            density = [min(1,NS_den/(len(self.env.density_lst)-1)),min(1,EW_den/(len(self.env.density_dict)-1))]
        
        if len(self.env.cum_time) == 0:
            time = [0,0]
        elif len(self.env.cum_time) == 1:
            time = self.env.cum_time[0]
        else:
            time = [NS_time/(len(self.env.cum_time)-1),EW_time/(len(self.env.cum_time)-1)]
        
        self.env.density_dict = {}
        self.env.density_lst = []
        self.env.cum_time = []
        # print('density:', density)
        observation = np.array(phase_id + min_green + time + density, dtype=np.float32)
        # print('observation:', observation)
        # breakpoint()
        return observation  
    def compute_observation_3_comb_avg_density(self):
        "no min green; combined NS and EW density"
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        # print('phase_id:', phase_id)
        min_green = [0 if self.time_since_last_phase_change < self.min_green + self.yellow_time else 1]
        # print('min_green:', min_green)
        # density = self.get_lanes_density_combine()
        NS_den = 0
        EW_den = 0
        NS = []
        EW = []
        # print("self.env.density_dict:", self.env.density_dict)
        # print("self.env.density_lst:", self.env.density_lst)
        # breakpoint()
        # for key in self.env.density_dict:
        #     NS_den += self.env.density_dict[key][0] #NS
        #     EW_den += self.env.density_dict[key][1] #NS
        for i in range(1, len(self.env.density_lst)):
            NS_den += self.env.density_lst[i][0]
            NS.append(self.env.density_lst[i][0])

            EW_den += self.env.density_lst[i][1]
            EW.append(self.env.density_lst[i][1])
        if len(self.env.density_lst) == 0:
            density = [0,0]
        else:
            density = [min(1,NS_den/(len(self.env.density_lst)-1)),min(1,EW_den/(len(self.env.density_dict)-1))]
        self.std_ew.append(np.std(np.array(EW)))
        self.std_ns.append(np.std(np.array(NS)))
        self.env.density_dict = {}
        self.env.density_lst = []
        # print('density:', density)
        observation = np.array(phase_id + min_green + density, dtype=np.float32)
        # print('observation:', observation)
        return observation

    def compute_observation_3(self):
        S1, S2, cur_phase = self.get_state()
        S_index = self.get_state_index(S1, S2, cur_phase)
        # print('S_index:', S_index)
        return np.array([S_index], dtype=np.float32)
    
    def compute_observation_3_num_veh(self):
        phase_id = [1 if self.green_phase == i else 0 for i in range(self.num_green_phases)]  # one-hot encoding
        
        # number of vehicles
        num_veh_NS, num_veh_EW= self.get_state()
        # print('S_index:', S_index)
        return np.array(phase_id + [num_veh_NS, num_veh_EW], dtype=np.float32)
    
    def get_lanes_density_combine(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        NS_veh = 0
        EW_veh = 0
        NS_max_veh = 0
        EW_max_veh = 0
        NS_d = 0
        EW_d = 0
        # print("lanes:", self.lanes)
        for i in range(len(self.lanes)):
            # self.lanes: ['n_t_0', 'n_t_1', 'w_t_0', 'w_t_1']
            # print('lane length:', self.lanes_lenght[self.lanes[i]])
            if self.road_type == "one_way":
                assert self.lanes == ['n_t_0', 'n_t_1', 'w_t_0', 'w_t_1']
                if i < 2:
                    NS_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    NS_veh += self.sumo.lane.getLastStepVehicleNumber(self.lanes[i])
                    
                else:
                    EW_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    EW_veh += self.sumo.lane.getLastStepVehicleNumber(self.lanes[i])
            else: # two way: upstream laneIDs: ['-4i_0', '-2i_0', '-3i_0', '1i_0'] nesw
                assert self.lanes == ['-4i_0', '-2i_0', '-3i_0', '1i_0']
                if i in [0,2]:
                    NS_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    NS_veh += self.sumo.lane.getLastStepVehicleNumber(self.lanes[i])
                    
                if i in [1,3]:
                    EW_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    EW_veh += self.sumo.lane.getLastStepVehicleNumber(self.lanes[i])
        self.NS_veh = NS_veh
        self.EW_veh = EW_veh
        NS_d = NS_veh/NS_max_veh
        EW_d = EW_veh/EW_max_veh
        return [min(1, NS_d), min(1, EW_d)]

    def get_lanes_queue_combine(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        NS_veh = 0
        EW_veh = 0
        NS_max_veh = 0
        EW_max_veh = 0
        NS_q = 0
        EW_q = 0
        for i in range(len(self.lanes)):
            if self.road_type == "one_way":
            # self.lanes: ['n_t_0', 'n_t_1', 'w_t_0', 'w_t_1']
                if i < 2:
                    NS_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    NS_veh += self.sumo.lane.getLastStepHaltingNumber(self.lanes[i])
                    
                else:
                    EW_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    EW_veh += self.sumo.lane.getLastStepHaltingNumber(self.lanes[i])
            else: # two way: upstream laneIDs: ['-4i_0', '-2i_0', '-3i_0', '1i_0'] nesw
                if i in [0,2]:
                    NS_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    NS_veh += self.sumo.lane.getLastStepHaltingNumber(self.lanes[i])
                    
                if i in [1,3]:
                    EW_max_veh = 2*self.lanes_lenght[self.lanes[i]] / vehicle_size_min_gap
                    EW_veh += self.sumo.lane.getLastStepHaltingNumber(self.lanes[i])
        self.NS_veh = NS_veh
        self.EW_veh = EW_veh
        NS_q = NS_veh/NS_max_veh
        EW_q = EW_veh/EW_max_veh
        # if EW_q > 0.7:
        #     breakpoint()
        return [min(1, NS_q), min(1, EW_q)]

        #use
    def compute_reward(self):
        self.last_reward = self._waiting_time_reward()
        dis = math.pow(self.gamma,self.step_)
        self.discounted_r += self.last_reward*dis
        self.step_ += 1
        return self.last_reward

    def compute_reward_pressure(self):
        self.last_reward = self.get_pressure()
        dis = math.pow(self.gamma,self.step_)
        self.discounted_r += self.last_reward*dis
        self.step_ += 1
        # breakpoint()
        # print("reward:",self.last_reward)
        # print("self.discounted_r:",self.discounted_r)
        # if self.step_ % 20 == 0:
        #    breakpoint()
        return self.last_reward

    
    def _pressure_reward(self):
        return -self.get_pressure()

    def compute_reward_paper(self):
        self.last_reward = self._waiting_time_reward()
        return self.last_reward
    # use
    def _waiting_time_reward(self):
        ts_wait = sum(self.get_waiting_time_per_lane()) / 100.0 # four lanes 
        reward = self.last_measure - ts_wait
        self.last_measure = ts_wait
        return reward

    def _waiting_time_reward2(self):
        ts_wait = sum(self.get_waiting_time())
        self.last_measure = ts_wait
        if ts_wait == 0:
            reward = 1.0
        else:
            reward = 1.0/ts_wait
        return reward

    def _waiting_time_reward3(self):
        ts_wait = sum(self.get_waiting_time())
        reward = -ts_wait
        self.last_measure = ts_wait
        return reward

    def get_reward_two(self, indicator):
        sum_queue = 0
        laneIDs = ['w_t_0', "w_t_1" , 'n_t_0', 'n_t_0']
        for lane in laneIDs:
            sum_queue += self.sumo.lane.getLastStepHaltingNumber(lane)
        if indicator == True: # phase change
            l_indicator = 1
        else: # phase not change
            l_indicator = 0
        return -0.5*sum_queue-0.5*l_indicator

    def get_waiting_time_per_lane(self):
        wait_time_per_lane = []
        veh_num = 0
        for lane in self.lanes: #last step veh incoming lane
            veh_list = self.sumo.lane.getLastStepVehicleIDs(lane)
            veh_num += len(veh_list) # total number of vehs on incoming lanes, last step
            wait_time = 0.0
            for veh in veh_list:
                veh_lane = self.sumo.vehicle.getLaneID(veh) # current lane, might be different from 
                acc = self.sumo.vehicle.getAccumulatedWaitingTime(veh)
                # print('acc:', acc)
                # The accumulated waiting time of a vehicle collects the vehicle's waiting time 
                # over a certain time interval (interval length is set per option '--waiting-time-memory') 100s
                if veh not in self.env.vehicles:
                    self.env.vehicles[veh] = {veh_lane: acc}
                else:
                    self.env.vehicles[veh][veh_lane] = acc - sum([self.env.vehicles[veh][lane] for lane in self.env.vehicles[veh].keys() if lane != veh_lane])
                    # veh first on lane 1, and then on lane 2, lane(previous lane) != veh_lane (current lane that veh is on), eg, acc = 30s, previous acc on lane 1 is 20s, difference = 30-20=10s
                    # id veh is on the outcoming lane, waiting time is 0, so waiting time is only considering vehs on the incoming lane.
                wait_time += self.env.vehicles[veh][veh_lane] # all veh in one lane
            wait_time_per_lane.append(wait_time) # wait time change for one lane
        # print('wait_time_per_lane:', wait_time_per_lane)
        total_wait_time = sum(wait_time_per_lane) # sum of each lane, each veh waiting time past 100s
        if veh_num == 0:
            avg_wait_time = 0
        else:
            avg_wait_time = total_wait_time/veh_num # 50s, 60s, 50+60 = 110 110/2 = 55
            #after taking action, get last step vehicle ID and calculate the acc waiting time over the past 100s
        # print('avg_wait_time:', avg_wait_time)
        # print('current time:', self.env.sim_step)
        # traceback.print_stack()

        return wait_time_per_lane

    def get_avg_cumulative_time(self):
        """
        Return current simulation second on SUMO
        """
        cur_time = self.sumo.simulation.getTime()
        num_veh_ns = 0
        num_veh_ew = 0
        tot_time_ns = 0
        tot_time_ew = 0
        laneIDs = self.lanes # just using upstream laneIDs: ['-4i_0', '-2i_0', '-3i_0', '1i_0'] nesw
        # print("laneIDs:",laneIDs)
        for i in range(len(laneIDs)):# incoming
            vehicleIDs=self.sumo.lane.getLastStepVehicleIDs(laneIDs[i]) # vehs ids of one lane
            vehicleNum=self.sumo.lane.getLastStepVehicleNumber(laneIDs[i]) # total number of vehs of one lane
            if laneIDs[i] == '-2i_0' or laneIDs[i] == '1i_0': #ew
                    num_veh_ew += vehicleNum
            else:
                    num_veh_ns += vehicleNum  
            if vehicleNum !=0:
                for v in vehicleIDs: 
                    enter_time = int(self.veh_info[v])
                    d_ = cur_time - enter_time
                    assert d_ >= 0
                    if laneIDs[i] == '-2i_0' or laneIDs[i] == '1i_0': #ew
                        tot_time_ew += d_
                    else:
                        tot_time_ns += d_
        if  num_veh_ns == 0 or num_veh_ew == 0:
            avg_time = [0,0]
        else:      
            avg_time = [tot_time_ns/num_veh_ns, tot_time_ew/num_veh_ew]
        return avg_time 
    
    def get_delay_lane_based(self,norm_delay):
        delay_lane = []
        for laneID in self.lanes:# incoming ['-4i_0', '-2i_0', '-3i_0', '1i_0'] nesw
            mean_speed = self.sumo.lane.getLastStepMeanSpeed(laneID)
            # print("mean speed:",mean_speed)
            # print("max speed:",self.sumo.lane.getMaxSpeed(laneID))
            numVeh = self.sumo.lane.getLastStepVehicleNumber(laneID)
            actual_TT = numVeh*1 # actual travel time of all veh with last step, 1s
            if mean_speed > self.sumo.lane.getMaxSpeed(laneID):
                fftt = 1
            else:
                fftt = mean_speed*1*numVeh/self.sumo.lane.getMaxSpeed(laneID)
            if norm_delay:
                if numVeh ==0:
                    delay = 0
                else:
                    delay = (actual_TT - fftt)/80
                    
            else:
                delay = actual_TT - fftt
            # print("actual_TT:",actual_TT)
            # print("fftt:", fftt)
            # print("mean_speed:", mean_speed)
            # print("self.sumo.lane.getMaxSpeed(laneID):", self.sumo.lane.getMaxSpeed(laneID))
            assert delay >= 0
            
            delay_lane.append(delay)
        # print("delay:",delay_lane)
        # breakpoint()
        self.delay_within_delta_time.append(-sum(delay_lane))

    def get_dischaarge_rate(self):
        # (veh_id, veh_length, entry_time, exit_time, vType)
        # [('eastbound_22', 5.0, 40.46961920592065, 40.79729176376016, 'typeWE')]
        ID = traci.inductionloop.getIDList()
        # print("ID:",ID)
        # breakpoint()
        d_r = []
        for id in ID:
            # print("info:",traci.inductionloop.getVehicleData(id))
            veh_ids = traci.inductionloop.getVehicleData(id) #vehs id in a specific loop
            if len(veh_ids) > 0:
                for veh in veh_ids:
                    if veh[3] != -1: # ensure the veh has left the loop detector
                        d_r.append(veh[0])
                    # print('d_r:',d_r)
        self.discharge_veh.append(len(d_r))
        # print("discharge_veh:",self.discharge_veh)
        

    def get_delay_lane_based_ew(self):
        delay_lane = []
        lane_ids = ['-2i_0','1i_0']
        for laneID in lane_ids:# incoming
            mean_speed = self.sumo.lane.getLastStepMeanSpeed(laneID)
            numVeh = self.sumo.lane.getLastStepVehicleNumber(laneID)
            actual_TT = numVeh*1 # actual travel time of all veh with last step, 1s
            if mean_speed > self.sumo.lane.getMaxSpeed(laneID):
                fftt = 1
            else:
                fftt = mean_speed*1*numVeh/self.sumo.lane.getMaxSpeed(laneID)
            delay = actual_TT - fftt
            # print("actual_TT:",actual_TT)
            # print("fftt:", fftt)
            # print("mean_speed:", mean_speed)
            # print("self.sumo.lane.getMaxSpeed(laneID):", self.sumo.lane.getMaxSpeed(laneID))
            assert delay >= 0
            
            delay_lane.append(delay)
        self.delay_within_delta_time_ew.append(-sum(delay_lane))

    def get_delay_lane_based_ns(self):
        delay_lane = []
        lane_ids = ['-4i_0', '-3i_0']
        for laneID in lane_ids:# incoming
            mean_speed = self.sumo.lane.getLastStepMeanSpeed(laneID)
            numVeh = self.sumo.lane.getLastStepVehicleNumber(laneID)
            actual_TT = numVeh*1 # actual travel time of all veh with last step, 1s
            if mean_speed > self.sumo.lane.getMaxSpeed(laneID):
                fftt = 1
            else:
                fftt = mean_speed*1*numVeh/self.sumo.lane.getMaxSpeed(laneID)
            delay = actual_TT - fftt
            # print("actual_TT:",actual_TT)
            # print("fftt:", fftt)
            # print("mean_speed:", mean_speed)
            # print("self.sumo.lane.getMaxSpeed(laneID):", self.sumo.lane.getMaxSpeed(laneID))
            assert delay >= 0
            
            delay_lane.append(delay)
        self.delay_within_delta_time_ns.append(-sum(delay_lane))

    def reward_pressure(self):
        in_pressure = 0
        out_pressure = 0
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        for lane in self.lanes:# incoming
            # print("incoming:", lane)
            in_pressure += self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)
        for lane in self.out_lanes:# outcoming
            # print("outcoming:", lane)
            out_pressure += self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)

        self.acc_pressure_with_delta_time.append(-abs(in_pressure - out_pressure))
        # print("acc:", self.acc_pressure_with_delta_time )
        # return out_pressure - in_pressure

    def get_pressure_no_abs(self):
        in_pressure = 0
        out_pressure = 0
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        for lane in self.lanes:# incoming
            # print("incoming:", lane)
            in_pressure += self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)
        for lane in self.out_lanes:# outcoming
            # print("outcoming:", lane)
            out_pressure += self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)

        self.acc_pressure_with_delta_time_no_abs.append(-in_pressure + out_pressure)
        # print("reward:",self.acc_pressure_with_delta_time_no_abs)
        
        # return -sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.lanes) + sum(self.sumo.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)

    def get_out_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.sumo.lane.getLength(lane) / vehicle_size_min_gap)) for lane in self.out_lanes]

    def get_lanes_density(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        # print('self.lanes:', self.lanes) # self.lanes: ['-4i_0', '-2i_0', '-3i_0', '1i_0']
        
        return [min(1, self.sumo.lane.getLastStepVehicleNumber(lane) / (self.lanes_lenght[lane] / vehicle_size_min_gap)) for lane in self.lanes]

    def get_lanes_queue(self):
        vehicle_size_min_gap = 7.5  # 5(vehSize) + 2.5(minGap)
        return [min(1, self.sumo.lane.getLastStepHaltingNumber(lane) / (self.lanes_lenght[lane] / vehicle_size_min_gap)) for lane in self.lanes]
    
    def get_total_queued(self):
        return sum([self.sumo.lane.getLastStepHaltingNumber(lane) for lane in self.lanes])

    def _get_veh_list(self):
        veh_list = []
        for lane in self.lanes:
            veh_list += self.sumo.lane.getLastStepVehicleIDs(lane)
        return veh_list
