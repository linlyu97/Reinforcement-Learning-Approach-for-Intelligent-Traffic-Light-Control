import numpy as np

import sumo_rl

# if 'SUMO_HOME' in os.environ:
#     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
#     sys.path.append(tools)
# else:
#     sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
# from sumolib import checkBinary  # noqa
# import traci  # noqa
from sumo_rl.exploration.ucb import UCB
from sumo_rl.exploration.ucb import dql_UCB


class DoubleQLAgent:

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, 
    exploration_strategy=EpsilonGreedy(), path_1 = None, path_2 = None, action_dict = None, 
    action_dict_1 = None, action_dict_2 = None,alpha_change=None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.alpha_change = alpha_change
        self.freq_by_state_by_action = action_dict
        self.freq_by_state_by_action_q1 = action_dict_1
        self.freq_by_state_by_action_q2 = action_dict_2

        """if action_dict == None:
            # self.freq_by_state_by_action = {} # {state: {action: count}}
            self.freq_by_state_by_action_q1 = {} # {state: {action: count}}
            self.freq_by_state_by_action_q2 = {} # {state: {action: count}}
        else:
            self.freq_by_state_by_action= action_dict
            self.freq_by_state_by_action_q1 = action_dict_1
            self.freq_by_state_by_action_q2 = action_dict_2"""
        self.reward_by_state_by_action = {}
            
        if path_1 == None:
            self.q_table_1 = {self.state: [0 for _ in range(action_space.n)]}
            self.q_table_2 = {self.state: [0 for _ in range(action_space.n)]}
        else:
            self.q_table_1 = path_1
            self.q_table_2 = path_2
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        action_freq = self.freq_by_state_by_action.get(self.state, {}) # {action: count}, action dict of a specific state
        state_action_total = 0 # for calculating the total number that state has been visited
        for action_id in action_freq:
            # action_space: Set([0, 1])
            # action_id: 0 or 1
            state_action_total += action_freq.get(str(action_id), 0)

        if isinstance(self.exploration, dql_UCB):
            # self.freq_by_state_by_action = {} # {state: {action: count}}
            action_freq = self.freq_by_state_by_action.get(self.state, {}) # {action: count}, action dict of a specific state
            # dict.get(key, value to be returned if no key found)
            self.action = self.exploration.choose(self.q_table_1,self.q_table_2, self.state, self.action_space, state_action_total,action_freq_dict=action_freq)
            # print("ucb")
        else:
            self.action = self.exploration.double_ql_choose(self.q_table_1, self.q_table_2,self.state, self.action_space,state_action_total)
        # print('self.action:', self.action)
        
        # update frequency
        self.freq_by_state_by_action.setdefault(self.state, {}) # setdefault, if no key, set a {}, else, did not change anything
        self.freq_by_state_by_action[self.state].setdefault(str(self.action), 0)
        self.freq_by_state_by_action[self.state][str(self.action)] += 1
        # print("self.freq_by_state_by_action:",self.freq_by_state_by_action)
        return self.action

    def count_freq(self,fre_action_q,a):
        fre_action_q.setdefault(self.state, {})
        fre_action_q[self.state].setdefault(str(a), 0)
        fre_action_q[self.state][str(a)] += 1

    def learn(self, next_state, selected_action, reward, done=False):
        if next_state not in self.q_table_1 or next_state not in self.q_table_2:
            self.q_table_1[next_state] = [0 for _ in range(self.action_space.n)]
            self.q_table_2[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state # current state
        s1 = next_state #next state, after applying the action in main function, call this function and give next state
        # a = self.action
        a = selected_action
        if np.random.random() < 0.5:
            #update q 1
            self.count_freq(self.freq_by_state_by_action_q1,a)
            # print(self.freq_by_state_by_action_q1)
            if self.alpha_change == True:
                self.alpha = 1/(self.freq_by_state_by_action_q1[s][str(a)])
            a_star = self.exploration.choose_max_action(self.q_table_1, s1)
            self.q_table_1[s][a] = self.q_table_1[s][a] + self.alpha*(reward + self.gamma*self.q_table_2[s1][a_star] - self.q_table_1[s][a])
        else:
            #update q 2
            self.count_freq(self.freq_by_state_by_action_q2,a)
            if self.alpha_change == True:
                self.alpha = 1/(self.freq_by_state_by_action_q2[s][str(a)])
            a_star = self.exploration.choose_max_action(self.q_table_2, s1)
            self.q_table_2[s][a] = self.q_table_2[s][a] + self.alpha*(reward + self.gamma*self.q_table_1[s1][a_star] - self.q_table_2[s][a]) 

        self.state = s1
        self.acc_reward += reward
        return self.q_table_1, self.q_table_2
"""
    def learn_ucb(self, next_state, selected_action, reward, done=False):
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]
        if next_state not in self.dict_freq:
            self.dict_freq[next_state] = 1
        else:
            self.dict_freq[next_state] += 1

        s = self.state
        s1 = next_state
        # a = self.action
        a = selected_action
        # breakpoint()
        H = 2880
        total = self.total_freq
        t = self.freq_by_state_by_action.get(s, {}).get(a, 0)
        bt = 1.0 * np.sqrt(2 * np.log(total) / t)
        self.q_table[s][a] = self.q_table[s][a] + self.alpha*(reward + self.gamma*max(self.q_table[s1]) - self.q_table[s][a])
        self.state = s1
        self.acc_reward += reward
        return self.q_table
"""


