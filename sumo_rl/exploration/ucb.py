import numpy as np
from gym import spaces
import random

class UCB:
    def __init__(self, c=1.0):
        self.c = c
        self.epsilon = 0.05

    def choose(self, q_table, state, action_space, state_action_total,action_freq_dict):
        # print("here")
        # action_freq_dict: {action: freq}, total: total freq
        u = [0] * len(q_table[state]) # for saving u term of each action
        # if np.random.rand() < self.epsilon or state not in action_freq_dict:
        if len(action_freq_dict) == 0:
            action = int(action_space.sample())
            # print(action_space.sample())
            return action
        for action_id in range(len(q_table[state])):
            u[action_id] = self.get_delta(action_freq_dict.get(str(action_id), 0), state_action_total)
        value = np.array(q_table[state]) + np.array(u)
        
        action = np.argmax(value)
        # print(self.epsilon)
        return action

    def get_delta(self, t, total):
        # t: frequency of the action
        # total: total number of actions
        if t == 0:
            # return self.c
            return float('inf') 
        else:
            return self.c * np.sqrt(np.log(float(total)) / t)


class dql_UCB:
    def __init__(self, c=1.0):
        self.c = c
        self.epsilon = 0.05
        self.uncertainty1 = []
        self.uncertainty2 = []
    
    def choose_max_action(self, q_table, state):
        action = np.argmax(q_table[state])
        return action

    def choose(self, q_table1, q_table2, state, action_space, state_action_total,action_freq_dict):
        # print("ucb")
        # action_freq_dict: {action: freq}, total: total freq
        u = [0] * len(q_table1[state]) # for saving u term of each action
        # if np.random.rand() < self.epsilon or state not in action_freq_dict:
        if len(action_freq_dict) == 0:
            action = int(action_space.sample())
            # print(action_space.sample())
            return action
        for action_id in range(len(q_table1[state])):
            u[action_id] = self.get_delta(action_freq_dict.get(str(action_id), 0), state_action_total)
        # breakpoint()
        value = np.array(q_table1[state]) + np.array(q_table2[state]) +np.array(u)
        
        action = np.argmax(value)
        self.uncertainty1.append(u[0])
        self.uncertainty2.append(u[1])
        # print(self.epsilon)
        return action

    def get_delta(self, t, total):
        # t: frequency of the action
        # total: total number of states visited 
        if t == 0:
            # return self.c
            # return float('inf') 
            return 10000
        else:
            return self.c * np.sqrt(np.log(float(total)) / t)
# command + shift + 7

class ddqn_UCB:
    def __init__(self, c=1.0):
        self.c = c
        self.epsilon = 0.05
        self.uncertainty1 = []
        self.uncertainty2 = []

    def choose(self, q_predict, state_action_total,action_freq_dict, action_code_lst):
        # print("ucb")
        # action_freq_dict: {action: freq}, total: total freq
        u = [0] * len(np.squeeze(q_predict)) # for saving u term of each action
        # if np.random.rand() < self.epsilon or state not in action_freq_dict:
        if len(action_freq_dict) == 0: # make sure every state visited at least once
            # action = int(action_space.sample())
            action = random.choice(action_code_lst)
            # action = np.argmax(q_predict)
            i = True
            # print(action_space.sample())
            return action, i
        # print(len(np.squeeze(q_predict)))
        for action_id in range(len(np.squeeze(q_predict))):
            # print("action_id:", action_id)
            u[action_id] = self.get_delta(action_freq_dict.get(str(action_id), 0), state_action_total)
        # breakpoint()
        value = np.squeeze(q_predict) + np.array(u)
        if u[0]!=u[1]:
            print("u",np.array(u))
        #     print("state_action_total:", state_action_total)
        #     print(action_freq_dict.get(str(0), 0))
        #     print(action_freq_dict.get(str(1), 0))
        #     breakpoint()
        # print("q_predict:",np.squeeze(q_predict))
        # print("value",value)
        
        
        action = np.argmax(value)
        # print("action:", action)
        # breakpoint()
        self.uncertainty1.append(u[0])
        self.uncertainty2.append(u[1])
        # print("u:",u[0],u[1])
        return action, None

    def get_delta(self, t, total):
        # t: frequency of the action
        # total: total number of states visited 
        if t == 0:
            # return self.c
            # return float('inf') 
            return 10000
        else:
            # print(self.c * np.sqrt(np.log(float(total)) / t))
            return self.c * np.sqrt(np.log(float(total)) / t)
# command + shift + 7

"""if first time visiting state s, random;
then if # action visited at state a is 0, it will select this action at the second time; (making sure every action in this state will be implemented at least once)
then if will follow the u term + q term making actions
"""
