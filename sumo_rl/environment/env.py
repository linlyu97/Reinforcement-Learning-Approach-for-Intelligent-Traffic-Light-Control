import os
import sys
from pathlib import Path
from typing import Optional, Union
import sumo_rl
import math
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd
import json

from .traffic_signal import TrafficSignal

from gym.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def env(**kwargs):
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param begin_time: (int) The time step (in seconds) the simulation starts
    :param num_seconds: (int) Number of simulated seconds on SUMO. The time in seconds the simulation must end.
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    :sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed.
    :fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions.
    :sumo_warnings: (bool) If False, remove SUMO warnings in the terminal
    """
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self, 
        net_file: str, 
        route_file: str, 
        add_file: str,
        out_csv_name: Optional[str] = None, 
        use_gui: bool = True, 
        begin_time: int = 0, 
        num_seconds: int = 100000, 
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1, 
        delta_time: int = 5, 
        yellow_time: int = 3, 
        min_green: int = 5, 
        max_green: int = 50, 
        single_agent: bool = False, 
        sumo_seed: Union[str,int] = 'random', 
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        epi: int = 0,
        road_type: str = "one_way",
        state: str = "",
        reward: str = "",
        sim_max_time: int = 0,
        veh_info: int = 0,
        gamma: float = 0.99,
        test_flow: bool = False,
        flow_ns = 300,
        flow_ew = 700,
        strategy = None,
        c_ucb = None,
        episode_output_path = None,
        test = None,
        new_state = None,
        seed = None
    ):
        
        self.new_state = new_state
        self.seed = seed
        self.gamma = gamma
        self.test = test
        self.episode_output_path = episode_output_path
        self.c_ucb= c_ucb
        self.strategy = strategy
        self.flow_ns = flow_ns
        self.flow_ew = flow_ew
        self.test_flow = test_flow
        self.road_type = road_type
        self.reward = reward
        self.state = state
        self._net = net_file
        self._route = route_file
        self._add = add_file
        self.use_gui = use_gui
        self.discounted_return = 0
        self.step_action = 0
        self.last_action_reward = None

        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time

        self.sim_max_time = sim_max_time

        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        self.epi = epi
        # with open("sumo_rl/environment/veh_info/veh_info_{}".format(self.epi), "r") as f:
        #     veh_info_dict = json.load(f)
        self.veh_info = veh_info
        self.veh_speed = {}
        self.veh_speed_lst = []
        self.density_dict = {}
        self.density_lst = []
        self.cum_time = []
        self.delay_container = []
        self.density_lst_separate = []


        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)


        self.ts_ids = list(conn.trafficlight.getIDList())
        # Returns a list of ids of all traffic lights within the scenario (the given Traffic Lights ID is ignored)
        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  conn,
                                                  self.epi,
                                                  self.road_type,
                                                  self.veh_info,
                                                  self.gamma) for ts in self.ts_ids}
        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
    
    def _start_simulation(self):
        newpath = self.episode_output_path 
        if self.test:
            test_path = "test"
            newpath = os.path.join(newpath, test_path)
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        # os.path.join(p1, p2)
        # "--tripinfo-output", os.path.join(newpath, "tripinfo.xml"),
        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '-a', self._add,
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', '10000',
                     '--time-to-teleport', str(self.time_to_teleport),
                    "--tripinfo-output", os.path.join(newpath, "tripinfo.xml"),
                     "--statistic-output", os.path.join(newpath, "stat_interval_{}_seed{}.xml".format(self.epi,self.seed))]
        """
        --max-depart-delay:
        How long vehicles wait for departure before being skipped, defaults to -1, here set to 100000
        which means vehicles are never skipped
        --waiting-time-memory: Length of time interval, over which accumulated waiting time is taken into account (default is 100s.)
        --time-to-teleport TIME: Specify how long a vehicle may wait until being teleported, defaults to 300,
            If the value is not positive, teleporting due to grid-lock is disabled. 
            A teleported vehicle is removed from the network. It is then moved along its route, but no longer being on the street. 
            It is reinserted into the network as soon as this becomes possible. While being teleported, 
            the vehicle is moved along its route with the average speed of the edge it was removed from or - later - it is currently "passing". 
            The vehicle is reinserted into the network if there is enough place to be placed on a lane which allows to continue its drive.
            Consequently a vehicle may teleport multiple times within one simulation.
            The threshold value can be configure using the option --time-to-teleport <INT> which sets the time in seconds. 
            If the value is not positive, teleporting due to grid-lock is disabled. Note that for vehicles which have a stop as part of their route, 
            the time spent stopping is not counted towards their waiting time.
        """
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
        

    def reset(self):
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self._start_simulation()
        # self.sumo = traci, the same as the conn in def ini

        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  self.sumo,
                                                  self.epi,
                                                  self.road_type,
                                                  self.veh_info,
                                                  self.gamma) for ts in self.ts_ids}
        self.vehicles = dict()

        if self.single_agent:
            if self.state == "3ele_queue":
                return self._compute_observation_3_comb_queue()[self.ts_ids[0]]
            if self.state == "3ele_density":
                return self._compute_observation_3_comb_density()[self.ts_ids[0]]
            if self.state == None:
                return self._compute_observations()[self.ts_ids[0]]
            if self.state == "9ele":
                return self._compute_observations_9()[self.ts_ids[0]]
            if self.state == "5ele":
                return self._compute_observations_5()[self.ts_ids[0]]
            if self.state == "3ele_density_avg":
                return self._compute_observations_3_comb_avg_density()[self.ts_ids[0]]
            if self.state == "4ele_density_avg":
                return self._compute_observations_4_comb_avg_density()[self.ts_ids[0]]
            if self.state == "6ele_mg_time_density_avg" or self.state == "5ele_mg_time_density_avg":
                return self._compute_observations_6_comb_avg_time_density()[self.ts_ids[0]]
            if self.state == "6ele_den_tslg":
                return self._compute_observations_6ele_den_tslg()[self.ts_ids[0]]
            if self.state == "4ele_den_tslg":
                return self._compute_observations_4ele_den_tslg()[self.ts_ids[0]]
            if self.state == "num_veh":
                return self._compute_observations_num_veh()[self.ts_ids[0]]
            if self.state == "9ele_tslg":
                return self._compute_observations_9_tslg()[self.ts_ids[0]]
            if self.state == "6ele_queue":
                return self._compute_observations_6ele_queue()[self.ts_ids[0]]
            if self.state == "6ele_queue_tslg":
                return self._compute_observations_6ele_queue_tslg()[self.ts_ids[0]]
            

        else:
            if self.state == "3ele_queue":
                return self._compute_observation_3_comb_queue()
            if self.state == "3ele_density":
                return self._compute_observation_3_comb_density()
            if self.state == None:
                return self._compute_observations()
            if self.state == "9ele":
                return self._compute_observations_9()
            if self.state == "5ele":
                return self._compute_observations_5()
            if self.state == "3ele_density_avg":
                return self._compute_observations_3_comb_avg_density()
            if self.state == "4ele_density_avg":
                return self._compute_observations_4_comb_avg_density()
            if self.state == "6ele_mg_time_density_avg" or self.state == "5ele_mg_time_density_avg":
                return self._compute_observations_6_comb_avg_time_density()
            if self.state == "6ele_den_tslg":
                return self._compute_observations_6ele_den_tslg()
            if self.state == "4ele_den_tslg":
                return self._compute_observations_4ele_den_tslg()
            if self.state == "num_veh":
                return self._compute_observations_num_veh()
            if self.state == "9ele_tslg":
                return self._compute_observations_9_tslg()
            if self.state == "6ele_queue":
                return self._compute_observations_6ele_queue()
            if self.state == "6ele_queue_tslg":
                return self._compute_observations_6ele_queue_tslg()


    @property
    # 
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(self, action,random_, predict_q, current_state,norm_delay,consider_yellow_red):
        # print('step action', action)
        # breakpoint()
        # No action, follow fixed TL defined in self.phases
        # action is dict from main function
        # print('action outer:', action)
        if action is None or action == {}:
            # print('action:', action)
            for _ in range(self.delta_time):

                self._sumo_step()
                # self.sumo.simulationStep()
        else:
            selected_action, indicator, phase_duration,current_phase = self._apply_actions(action)
            # print("run untill to next action time")
            # Set the next green phase for the traffic signals
            self._run_steps(norm_delay) 
            """
            loop delta time then get reward and next state, 
            could get acc reward during this delta time in run steps
            loop time_to_act = True
            """
        # breakpoint()
        # observations = self._compute_observations_3()
        if self.state == "3ele_queue":
            observations = self._compute_observation_3_comb_queue()
        if self.state == "3ele_density":
            observations = self._compute_observation_3_comb_density()
        if self.state == None:
            observations = self._compute_observations()
        if self.state == "9ele":
            observations = self._compute_observations_9()
            # breakpoint()
        if self.state == "5ele":
            observations = self._compute_observations_5()
        if self.state == "3ele_density_avg":
            observations = self._compute_observations_3_comb_avg_density()
        if self.state == "4ele_density_avg":
            observations = self._compute_observations_4_comb_avg_density()
        if self.state == "6ele_mg_time_density_avg" or self.state == "5ele_mg_time_density_avg":
            observations = self._compute_observations_6_comb_avg_time_density()
        if self.state == "6ele_den_tslg":
            observations = self._compute_observations_6ele_den_tslg()
        if self.state == "4ele_den_tslg":
            observations = self._compute_observations_4ele_den_tslg()
        if self.state == "num_veh":
            observations = self._compute_observations_num_veh()
        if self.state == "9ele_tslg":
            observations = self._compute_observations_9_tslg()
        if self.state == "6ele_queue":
            observations = self._compute_observations_6ele_queue()
        if self.state == "6ele_queue_tslg":
            observations = self._compute_observations_6ele_queue_tslg()
                
            
        
        # observations = self._compute_observation_3_num_veh()
        # observations = self._compute_observations_paper(phase_duration)

        # rewards = self._compute_rewards_3() #old reward
        if self.reward == "r_waiting":
            rewards = self._compute_rewards()
        if self.reward == "pressure":
            # r: {'0': -0.8146124953041614}
            rewards = self._compute_rewards_pressure()
        if self.reward == "lane_based_delay":
            for ts in self.ts_ids:
                # print("delay:", self.traffic_signals[ts].delay_within_delta_time)
                r = sum(self.traffic_signals[ts].delay_within_delta_time[int(-self.delta_time):])/len(self.traffic_signals[ts].delay_within_delta_time[int(-self.delta_time):])
                """for discounted return"""
                self.last_action_reward = r
                rewards = {ts: r}
                dis = math.pow(self.gamma,self.step_action)
                self.discounted_return += self.last_action_reward*dis
                self.step_action += 1
           
        if self.reward == "acc_pressure":
            for ts in self.ts_ids:
                r = sum(self.traffic_signals[ts].acc_pressure_with_delta_time[int(-self.delta_time):])
                # print("r:", r)
                """for discounted return"""
                self.last_action_reward = r
                rewards = {ts: r}
                dis = math.pow(self.gamma,self.step_action)
                self.discounted_return += self.last_action_reward*dis
                self.step_action += 1

        if self.reward == "acc_pressure_no_abs":
            for ts in self.ts_ids:
                r = sum(self.traffic_signals[ts].acc_pressure_with_delta_time_no_abs[int(-self.delta_time):])
                # print("r:", r)
                """for discounted return"""
                self.last_action_reward = r
                rewards = {ts: r}
                dis = math.pow(self.gamma,self.step_action)
                self.discounted_return += self.last_action_reward*dis
                self.step_action += 1
        if self.reward == "discharge_rate":
            for ts in self.ts_ids:
                if consider_yellow_red: # consider the reward of yellow+red time 
                    l=len(self.traffic_signals[ts].discharge_veh)
                    r = sum(self.traffic_signals[ts].discharge_veh)/l
                else: # consider only green time
                    r = sum(self.traffic_signals[ts].discharge_veh[int(-self.delta_time):])/5
                # print('r:',r)
                # print("r:", r)
                """for discounted return"""
                self.last_action_reward = r
                rewards = {ts: r}
                dis = math.pow(self.gamma,self.step_action)
                self.discounted_return += self.last_action_reward*dis
                self.step_action += 1

        obs = observations["0"]
        dones = self._compute_dones()

        """reset attribute to 0 in compute info before next action"""
        # breakpoint()
        self._compute_info(phase_duration,current_phase,selected_action,random_, predict_q, current_state,obs,consider_yellow_red)
        # for ts in self.ts_ids:
        #     print("avg delay:",self.traffic_signals[ts].get_avg_delay() )

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], selected_action, dones['__all__'], {}
        else:
            return observations, rewards, selected_action, dones, {}

    def _run_steps(self,norm_delay):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            # self.sumo.simulationStep()

            for ts in self.ts_ids:
                # self.ts_ids is list of traffic lights
                self.traffic_signals[ts].get_delay_lane_based(norm_delay) # get total delay with last step of all lanes
                self.traffic_signals[ts].get_delay_lane_based_ew()
                self.traffic_signals[ts].get_delay_lane_based_ns()
                self.traffic_signals[ts].reward_pressure() # get acc pressure within delta time
                self.traffic_signals[ts].get_pressure_no_abs() # get acc pressure within delta time, upstream-downstream
                self.traffic_signals[ts].get_dischaarge_rate()
                self.traffic_signals[ts].update() # self.time_since_last_phase_change + 1
                # get combiedn density
                """density_lst = self.traffic_signals[ts].get_lanes_density_combine() #combine
                assert self.sim_step-1 not in self.density_dict
                self.density_dict[self.sim_step-1] = density_lst
                self.density_lst.append(density_lst)"""
                # print("in",self.density_lst)
                #get combined cumulative time
                """avg_time = self.traffic_signals[ts].get_avg_cumulative_time()
                self.cum_time.append(avg_time)"""
                # get density nsew every second
                """den_lst_sep = self.traffic_signals[ts].get_lanes_density() #four values
                self.density_lst_separate.append(den_lst_sep)"""
                # breakpoint()
                # print("in:",self.density_lst_separate)
                if self.traffic_signals[ts].time_to_act: # self.next_action_time == self.env.sim_step
                    time_to_act = True

    def _apply_actions(self, actions):
        # actions: a chosed action from main function
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                selected_action, indicator, phase_duration,current_phase = self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
                return selected_action, indicator, phase_duration,current_phase
        else:
            #"not single agent"
            # breakpoint()
            for ts, action in actions.items():
                # acrions: {'t': 0}
                # ts is the id of traffic light
                # action is 0, initialization
                # print('actions.items():', actions.items())
                '''actions.items(): dict_items([('t', 1)])'''
                '''actions.items(): dict_items([('t', 0)])'''
                if self.traffic_signals[ts].time_to_act: # self.next_action_time == self.env.sim_step
                    selected_action, indicator, phase_duration,current_phase = self.traffic_signals[ts].set_next_phase(action)
                    return selected_action, indicator, phase_duration,current_phase


    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}

        dones['__all__'] = self.sim_step >= self.sim_max_time

        return dones
    
    def _compute_info(self,phase_duration,current_phase,selected_action,random_, predict_q, current_state,obs,consider_yellow_red):
        info = self._compute_step_info(phase_duration,current_phase,selected_action,random_, predict_q, current_state,obs,consider_yellow_red)
        self.metrics.append(info)
    
    def _compute_observations_paper(self,phase_duration):
        
        # 5 observations paper
        # {agent ID: observation}
        # state: [cur_phase, min_green or not, queue for NSEW, density for NSEW]
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_paper(phase_duration) for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}


    def _compute_observations(self):
        # ten observations
        # {agent ID: observation}
        # state: [cur_phase, min_green or not, queue for NSEW, density for NSEW]
        self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_observations_num_veh(self):
        # ten observations
        # {agent ID: observation}
        # state: [cur_phase, min_green or not, queue for NSEW, density for NSEW]
        self.observations.update({ts: self.traffic_signals[ts].compute_state_num_veh() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_observations_9(self):
        # {agent ID: observation}
        # delete min green
        # state: [cur_phase, queue for NSEW, density for NSEW]
        # update values for key, if key did not exist, add a new key
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_9() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observations_9_tslg(self):
        # {agent ID: observation}
        # delete min green
        # state: [cur_phase, queue for NSEW, density for NSEW]
        # update values for key, if key did not exist, add a new key
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_9_tslg() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observations_6ele_queue(self):
        # {agent ID: observation}
        # delete min green
        # state: [cur_phase, queue for NSEW, density for NSEW]
        # update values for key, if key did not exist, add a new key
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_queue() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observations_6ele_queue_tslg(self):
        # {agent ID: observation}
        # delete min green
        # state: [cur_phase, queue for NSEW, density for NSEW]
        # update values for key, if key did not exist, add a new key
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_queue_tslg() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observations_5(self):
        # {agent ID: observation}
        # combined state: [cur_phase, combine queue for NS&EW, combined queue for NS&EW]
        # update values for key, if key did not exist, add a new key
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_5() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_observation_3_comb_queue(self):
        # {agent ID: observation}
        # old_state: [cur_phase, #veh for NS, #veh for EW]
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_3_comb_queue() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observation_3_comb_density(self):
        # {agent ID: observation}
        # old_state: [cur_phase, #veh for NS, #veh for EW]
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_3_comb_density() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observations_3_comb_avg_density(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_3_comb_avg_density() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    def _compute_observations_4_comb_avg_density(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_4_comb_avg_density() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    def _compute_observations_6_comb_avg_time_density(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_6_comb_avg_time_density() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    def _compute_observations_6ele_den_tslg(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_6ele_den_tslg() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    def _compute_observations_4ele_den_tslg(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_4ele_den_tslg() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_observation_3_num_veh(self):
        # {agent ID: observation}
        # old_state: [cur_phase, #veh for NS, #veh for EW]
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_3_num_veh() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}
    

    def _compute_observations_3(self):
        # {agent ID: observation}
        # old_state: [cur_phase, #veh for NS, #veh for EW]
        self.observations.update({ts: self.traffic_signals[ts].compute_observation_3() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self):
        #github reward
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards_2(self, indicator):
        # paper reward: -0.5*sum_queue-0.5*l_indicator
        self.rewards.update({ts: self.traffic_signals[ts].get_reward_two(indicator) for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}
    
    def _compute_rewards_pressure(self):
        # paper reward: pressure
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward_pressure() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}


    @property
    def observation_space(self):
        return self.traffic_signals[self.ts_ids[0]].observation_space
    
    @property
    def action_space(self):
        return self.traffic_signals[self.ts_ids[0]].action_space
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        
        # for ts in self.ts_ids:
        #     self.traffic_signals[ts].get_avg_delay() 

        """
        # print()
        # every seconds, call this function, to store and update delay for each veh untill they leave upstream
        
        if self.road_type == "one_way":
            laneIDs =  ['n_t_0', 'n_t_1', 'w_t_0', 'w_t_1']
        else:
            laneIDs = ['-4i_0', '-2i_0', '-3i_0', '1i_0']
        for lane in laneIDs:
            vehIDs = traci.lane.getLastStepVehicleIDs(lane)
            for veh in vehIDs:
                traci.vehicle.setLaneChangeMode(veh, 0b001000000000)
                self.veh_speed_lst.append(int(traci.vehicle.getSpeed(veh)))
                if veh not in self.veh_speed:
                    self.veh_speed[veh] = []
                    self.veh_speed[veh].append(traci.vehicle.getSpeed(veh))
                else:
                    self.veh_speed[veh].append(traci.vehicle.getSpeed(veh))
                
                # print('getLaneChangeMode(self, vehID):', bin(traci.vehicle.getLaneChangeMode(veh)),self.sumo.simulation.getTime())
        """
        
        self.sumo.simulationStep()
    """
    def _compute_step_info(self):
        # for ts in self.ts_ids:
        #     print('self.traffic_signals[ts].get_waiting_time_per_lane():', self.traffic_signals[ts].get_waiting_time_per_lane())
            
        return {
            'step_time': self.sim_step,
            'reward': self.traffic_signals[self.ts_ids[0]].last_reward,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids),
            'avg_wait_time': sum(sum(self.traffic_signals[ts].get_avg_waiting_time_per_lane()) for ts in self.ts_ids),
            'waiting_number': sum(sum(self.traffic_signals[ts].get_waiting_time_number_per_lane()) for ts in self.ts_ids),
            'Avg_delay':self.traffic_signals[self.ts_ids[0]].delay,
        }
    """
    def _compute_step_info(self,phase_duration,current_phase,selected_action,random_, predict_q, current_state,obs,consider_yellow_red):
        # for ts in self.ts_ids:
        #     print('self.traffic_signals[ts].get_waiting_time_per_lane():', self.traffic_signals[ts].get_waiting_time_per_lane())
        for ts in self.ts_ids:
            l=len(self.traffic_signals[ts].discharge_veh)
            # print(len(self.traffic_signals[ts].discharge_veh))
            # print(len(self.traffic_signals[ts].delay_within_delta_time))
            # print(len(self.traffic_signals[ts].acc_pressure_with_delta_time))
            # print(len(self.traffic_signals[ts].delay_within_delta_time_ns))
            # print(len(self.traffic_signals[ts].delay_within_delta_time_ew))
            # breakpoint()
            if not consider_yellow_red: # only green
                delay_in_delta_time_avg = sum(self.traffic_signals[ts].delay_within_delta_time[int(-self.delta_time):])/len(self.traffic_signals[ts].delay_within_delta_time[int(-self.delta_time):])
                delay_in_delta_time_avg_ew = sum(self.traffic_signals[ts].delay_within_delta_time_ew[int(-self.delta_time):])/len(self.traffic_signals[ts].delay_within_delta_time_ew[int(-self.delta_time):])
                delay_in_delta_time_avg_ns = sum(self.traffic_signals[ts].delay_within_delta_time_ns[int(-self.delta_time):])/len(self.traffic_signals[ts].delay_within_delta_time_ns[int(-self.delta_time):])
                pressure = sum(self.traffic_signals[ts].acc_pressure_with_delta_time[int(-self.delta_time):])
                pressure_no_abs = sum(self.traffic_signals[ts].acc_pressure_with_delta_time_no_abs[int(-self.delta_time):])
                discharge_rate = sum(self.traffic_signals[ts].discharge_veh[int(-self.delta_time):])/len(self.traffic_signals[ts].discharge_veh[int(-self.delta_time):])
            else: # consider yellow+red
                delay_in_delta_time_avg = sum(self.traffic_signals[ts].delay_within_delta_time)/l
                delay_in_delta_time_avg_ew = sum(self.traffic_signals[ts].delay_within_delta_time_ew)/l
                delay_in_delta_time_avg_ns = sum(self.traffic_signals[ts].delay_within_delta_time_ns)/l
                pressure = sum(self.traffic_signals[ts].acc_pressure_with_delta_time)
                pressure_no_abs = sum(self.traffic_signals[ts].acc_pressure_with_delta_time_no_abs)
                discharge_rate = sum(self.traffic_signals[ts].discharge_veh)/l
            self.traffic_signals[ts].discharge_veh = []
            self.traffic_signals[ts].delay_within_delta_time = []
            self.traffic_signals[ts].acc_pressure_with_delta_time_no_abs = []
            self.traffic_signals[ts].acc_pressure_with_delta_time = []
            self.traffic_signals[ts].delay_within_delta_time_ns = []
            self.traffic_signals[ts].delay_within_delta_time_ew = []
            """self.delay_container.append(self.traffic_signals[ts].get_avg_delay())"""
        return {
            'step_time_after_performing_selected_action': self.sim_step,
            'total_delay_per_second_in_delta_time': delay_in_delta_time_avg,
            'total_pressure_in_delta_time': pressure,
            'phase_duration_at_current_time': phase_duration,
            'phase_before_performing_selected_phase': current_phase,
            'selected_phase':selected_action,
            'total_ew_delay_per_second_in_delta_time': delay_in_delta_time_avg_ew,
            'total_ns_delay_per_secondin_delta_time': delay_in_delta_time_avg_ns,
            "random_choice_or_not": random_,
            "predict_Q_when_selecting_action": predict_q,
            "current_state_after_performing_action":obs,
            "discharge_rate": discharge_rate
        }

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        self.sumo = None
    
    def __del__(self):
        self.close()
    
    def render(self, mode=None):
        pass
    
    def save_csv(self, out_csv_name, run, csv_folders):
        # breakpoint()
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(Path(os.path.join(csv_folders,out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv'), index=False))

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        # state: s: observations[self.ts_ids[0]] ts_ids is agent, 11 elements
        # ts_id: agent id
        # breakpoint()
        """
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])"""
        # import pdb; pdb.set_trace()
        # 1，0 True False -> 1; 0,1 False True -> 1
        # if ==0, first green phase; else second green phase
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        # phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        # if self.state == "4ele_density_avg" or self.state == "6ele_mg_time_density_avg":
        min_green = state[self.traffic_signals[ts_id].num_green_phases]  # observation 里的 第三位

        'for 9ele_d_q, 5ele_d_q, 4ele_mg_d, 3ele_d, 3ele_avg_d, 3ele_avg_q'
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        'for 9/5/3combined q or density elements'
        """large section"""
        # if self.state == "3ele_density": # no mg
        #     density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases :]]
        if self.state == "6ele_mg_time_density_avg" or self.state == "5ele_mg_time_density_avg": #[phase0,1, mg, t1,t2, den1,den2]
            time =  [self._discretize_time(t) for t in state[self.traffic_signals[ts_id].num_green_phases+1 : 5]]
            density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases+3 :]]
        if self.state == "6ele_den_tslg" or self.state == "4ele_den_tslg": #[phase_id01 + min_green + density + queue, self.time_since_last_phase_change]
            density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases+1 : -1]]
            # tslg = [self._discretize_tslg(state[-1])]
            tslg = [state[-1]]
            #new state
            if self.new_state == True:
                if tslg[0] >= 60:
                    print("tslg:",tslg[0])
                    tslg = [60]

        # d_q = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases :]]
        # r1 = (density_queue[0] + 1) / (density_queue[1] + 1)
        # r1 = np.log(r1)
        # r1 = int(r1*10)

        # r2 = density_queue[0] - density_queue[1]
        # r2 = int(r2*10)
        # density_queue = [r1, r2, d_q[0],d_q[1]]


        # density_queue is the last 8 elements, the queue and density
        # density_queue = [self._discretize_density_small(d) for d in state[self.traffic_signals[ts_id].num_green_phases :]]

        "not discretize"
        # density_queue = [self._discretize_density_remain(d) for d in state[self.traffic_signals[ts_id].num_green_phases :]]
        # tuples are hashable and can be used as key in python dictionary

        'for 10 elements'
        # return tuple([phase, min_green] + density_queue)
        'for 9/5/3 combined q or density elements'
        # print("observations:", str([phase] + time + density_queue))
        if self.state == "5ele_mg_time_density_avg":
            return str([phase] + time + density_queue)
        elif self.state == "6ele_mg_time_density_avg":
            return str([phase, min_green] + time + density_queue)
        elif self.state == "4ele_density_avg":
            return str([phase, min_green] + density_queue)
        elif self.state == "6ele_den_tslg" or self.state == "4ele_den_tslg":
            # print(str([phase] + density_queue + tslg))
            # breakpoint()
            return str([phase] + density_queue + tslg)
        else: 
            return str([phase] + density_queue)
            """for 9ele_d_q, 5ele_d_q, 3ele_d, 3ele_avg_d"""
       

    def _discretize_density(self, density):
        return min(int(density*10), 9)
    
    def _discretize_time(self, time):
        if time < 10:
            return 0
        elif 10 <= time < 20:
            return 1
        else:
            return 2
    def _discretize_tslg(self,tslg):
        if tslg <= 5:
            return 0
        elif tslg == 10:
            return 1
        elif tslg == 15:
            return 2
        else:
            return 3

    def _discretize_density_small(self, density):
        if density < 0.3:
        # return min(int(density*10), 9)
            return round(density*10*2)/2
        else:
            return min(int(density*10), 9)

    def _discretize_density_ratio(self, density):
        return 

    def _discretize_density_remain(self, density):
        return density
    

    def encode_paper(self, state, ts_id):
        # state: s: observations[self.ts_ids[0]] ts_ids is agent, 11 elements
        # ts_id: agent id
        # breakpoint()
        """
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])"""
        # import pdb; pdb.set_trace()
        # 1，0 True False -> 1; 0,1 False True -> 1
        # if ==0, first green phase; else second green phase
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        # breakpoint()
        return tuple([phase] + list(state[self.traffic_signals[ts_id].num_green_phases :]))

    
    def encode_new(self, state, ts_id):
        # breakpoint()
        # print('state:', state)
        observations = state[0]
        # print('observations in encode:', observations)
        return tuple([observations,])


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {'render.modes': [], 'name': "sumo_rl_v0"}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self):
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent):
        obs = self.env.observations[agent].copy()
        return obs

    def state(self):
        raise NotImplementedError('Method state() currently not implemented.')

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    def save_csv(self, out_csv_name, run):
        self.env.save_csv(out_csv_name, run)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.env._compute_info()
        else:
            self._clear_rewards()
        
        done = self.env._compute_dones()['__all__']
        self.dones = {a : done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
