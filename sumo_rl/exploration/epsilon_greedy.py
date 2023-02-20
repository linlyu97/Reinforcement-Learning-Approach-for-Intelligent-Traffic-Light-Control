import numpy as np
from gym import spaces
import math
import random


class EpsilonGreedy:

    def __init__(self, initial_epsilon=1.0, min_epsilon=0.0, decay=0.99,epsilon_change = None):
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay
        self.epsilon_change = epsilon_change
        self.exploration_list = []
        
    def choose(self, q_table, state, action_space,state_action_total):
        if self.epsilon_change == True:
            if state_action_total == 0:
                self.epsilon = self.epsilon
            else:
                self.epsilon = 1/(math.sqrt(state_action_total))
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())

        else:
            action = np.argmax(q_table[state])

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        breakpoint()
        # print(self.epsilon)
        return action
    
    def double_ql_choose(self,q_table_1, q_table_2,state,action_space,state_action_total):
        """exploration_dict here: exploration_dict[epi]"""
        if self.epsilon_change == True:
            if state_action_total == 0:
                self.epsilon = self.epsilon
            else:
                self.epsilon = 1/(math.sqrt(state_action_total))
        if np.random.rand() < self.epsilon:
            action = int(action_space.sample())

        else:
            q1_action_values = np.array(q_table_1[state])
            q2_action_values = np.array(q_table_2[state])
            action = np.argmax(q1_action_values+q2_action_values)

        self.epsilon = max(self.epsilon*self.decay, self.min_epsilon)
        self.exploration_list.append(self.epsilon)
        # print(self.epsilon)
        return action

    def choose_max_action(self, q_table, state):
        action = np.argmax(q_table[state])
        return action

    def new_choose(self, q_table, state, action_space):# for fixed time
        cur_phase = state[1]
        # print('cur_phase:', cur_phase)
        if int(cur_phase) == 0:
            return 1
        else:
            return 0

    def reset(self):
        self.epsilon = self.initial_epsilon

    def ddqn_choose(self,q_predict,state_action_total, action_code_lst):
        # state_action_total: # of times of state visited
        """exploration_dict here: exploration_dict[epi]"""
        if self.epsilon_change == "adaptive":
            if state_action_total == 0:
                self.epsilon = 0.05
                print("epsilon:",self.epsilon)
            else:
                self.epsilon = 1/(math.sqrt(state_action_total))
                print("not first time:",self.epsilon )
        if random.random() < self.epsilon:
            # action = int(action_space.sample())
            action = random.choice(action_code_lst)
            i = True

        else:
            action = np.argmax(q_predict)
            i = False
        self.exploration_list.append(self.epsilon)
        # print(self.epsilon)
        return action, i

# command + shift + 7
