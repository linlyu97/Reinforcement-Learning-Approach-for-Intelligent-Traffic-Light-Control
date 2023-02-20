import random
import numpy as np
import sumo_rl
from sumo_rl.agents.deep_q.memory import Memory
from sumo_rl.agents.deep_q.model import TrainModel
from tensorflow.keras.models import clone_model
from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy
# from sumolib import checkBinary  # noqa
# import traci  # noqa
from sumo_rl.exploration.ucb import UCB
from sumo_rl.exploration.ucb import ddqn_UCB

class Adaptor:
    def __init__(self, state_space, action_space) -> None:
        self.state_space = state_space
        self.action_space = action_space
    
    def _state_from_original_to_deep(self, original_state: str) -> np.ndarray:
        return original_state
    
    def _state_from_deep_to_original(self, deep_state: np.ndarray):
        return deep_state
    
    def _action_from_original_to_deep(self, original_action: int) -> int:
        return original_action
    
    def _action_from_deep_to_original(self, deep_action: int) -> int:
        return deep_action

    def _reward_from_original_to_deep(self, original_reward: float) -> float:
        return original_reward

class DeepQAgent:
    def __init__(self, starting_state, state_space, action_space, train, model: TrainModel,
     memory: Memory, epsilon=0.05, gamma=0.8, exploration_strategy=EpsilonGreedy(), state_action_dict = None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action_code_lst = []
        for i in range(self.action_space):
            self.action_code_lst.append(i)
        self.action = None
        self._Model = model
        if train:
            self._Model_target = model.clone()
        self._Memory = memory
        self.adaptor = Adaptor(state_space, action_space)
        self.epsilon = epsilon
        self._num_states = len(starting_state) # 理论上应该是 state_space.shape[0]
        self._num_actions = action_space
        self._gamma = gamma
        self.freq_by_state_by_action = state_action_dict
        self.exploration = exploration_strategy

    def act(self, train=True):
        deep_state = self.adaptor._state_from_original_to_deep(self.state)
        prediction: np.ndarray = self._Model.predict_one(deep_state)
        action_freq = self.freq_by_state_by_action.get(str(self.state), {}) # {action: count}, action dict of a specific state
        state_action_total = 0 # for calculating the total number that state has been visited
        for action_id in action_freq:
            # action_space: Set([0, 1])
            # action_id: 0 or 1
            state_action_total += action_freq.get(str(action_id), 0)
        if isinstance(self.exploration, ddqn_UCB): # UCB
            # self.freq_by_state_by_action = {} # {state: {action: count}}
            # dict.get(key, value to be returned if no key found)
            self.action, random_or_not = self.exploration.choose(prediction, state_action_total,action_freq, self.action_code_lst)
            # print("ucb")
        else: # egreedy or adaptive one
            self.action, random_or_not = self.exploration.ddqn_choose(prediction,state_action_total, self.action_code_lst)
            # print("e-greedy")
        # print('self.action:', self.action)
        # update frequency
        self.freq_by_state_by_action.setdefault(str(self.state), {}) # setdefault, if no key, set a {}, else, did not change anything
        self.freq_by_state_by_action[str(self.state)].setdefault(str(self.action), 0)
        self.freq_by_state_by_action[str(self.state)][str(self.action)] += 1
        # print("self.freq_by_state_by_action:",self.freq_by_state_by_action)

        return self.action, random_or_not, prediction, self.state

        """if random.random() < self.epsilon and train == True:
            # breakpoint()
            # self.action = int(self.action_space.sample())
            self.action = random.choice(self.action_code_lst)
            # print("action:", self.action)
            return self.action, True, None, self.state
        else:
            deep_state = self.adaptor._state_from_original_to_deep(self.state)
            prediction: np.ndarray = self._Model.predict_one(deep_state)
            # print("prediction",prediction)
            
            action_idx = np.argmax(prediction)
            self.action = self.adaptor._action_from_deep_to_original(action_idx)
            return self.action,False,prediction,self.state"""
    

    def set_next_state(self, next_state):
        self.state = next_state

    def get_data(self, next_state, selected_action, reward):
        state = self.adaptor._state_from_original_to_deep(self.state)
        action = self.adaptor._action_from_original_to_deep(selected_action)
        next_state = self.adaptor._state_from_original_to_deep(next_state)
        deep_reward = self.adaptor._reward_from_original_to_deep(reward)
        self._Memory.add_sample((state, action, deep_reward, next_state))
        self.state = next_state
        return None
    
    def replay(self, update=False, model="DQN"):
        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch])  # extract states from the batch
            next_states = np.array([val[3] for val in batch])  # extract next states from the batch
            # print("data:",batch)
            q_s_a = self._Model.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_prime = self._Model.predict_batch(next_states)  # predict Q(next_state), for every sample
            q_s_a_d = self._Model_target.predict_batch(next_states)  # predict target_Q(next_state), for every sample
            # setup training arrays
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                if model == "DQN":
                    current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action); amax: Maximum of the flattened array
                elif model == "DDQN": 
                    a_max = np.argmax(q_s_a_prime[i]) 
                    current_q[action] = reward + self._gamma * q_s_a_d[i][a_max]
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
                # breakpoint()
            self._Model.train_batch(x, y)  # train the NN; 
            # print("y:",y)
            # x is still a batch of states, predict y here is q_s_a(a1,a2), and the true y is calculated using q_s_a_d(a1,a2); calculated MSE for the only action in memory above
            if update:
                self._Model_target = self._Model.clone()

