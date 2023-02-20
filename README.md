# Reinforcement-Learning-Approach-for-Intelligent-Traffic-Light-Control

## MDP - Observations, Actions and Rewards

### Observation
The default observation for each traffic signal agent is a vector:
```python
    obs = [phase_one_hot, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue, phase_duration]
```
- ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
- ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
- ```lane_i_queue```is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane
- ```phase_duration```is the phase duration of the current phase

You can define your own observation changing the method 'compute_observation' of [TrafficSignal](https://github.com/linlyu97/Reinforcement-Learning-Approach-for-Intelligent-Traffic-Light-Control/blob/main/sumo_rl/environment/traffic_signal.py).

### Actions
The action space is discrete.
Every 'delta_time' seconds, each traffic signal agent can choose the next green phase configuration.

    
Important: every time a phase change occurs, the next phase is preeceded by a 3s yellow phase lasting ```yellow_time``` seconds and 2s all red time implemented.

### Rewards
For double deep q model, the default reward function is the discharge rate, which is, the number of vehicles that could be discharged per second between two consecutive actions.

For double q learning model, the reward is "lane_based_delay" between two consecutive actions.

You can choose a different reward function (see the ones implemented in [TrafficSignal](https://github.com/linlyu97/Reinforcement-Learning-Approach-for-Intelligent-Traffic-Light-Control/blob/main/sumo_rl/environment/traffic_signal.py)) 

It is also possible to implement your own reward function:

```python
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

env = SumoEnvironment(..., reward_fn=my_reward_fn)
```

### Single Agent Environment

If your network only has ONE traffic light, then you can instantiate a standard Gymnasium env (see [Gymnasium API](https://gymnasium.farama.org/api/env/)):
```python
import gymnasium as gym
import sumo_rl
env = gym.make('sumo-rl-v0',
                net_file='path_to_your_network.net.xml',
                route_file='path_to_your_routefile.rou.xml',
                out_csv_name='path_to_output.csv',
                use_gui=True,
                num_seconds=100000)
obs, info = env.reset()
done = False
while not done:
    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    done = terminated or truncated
```

### PettingZoo Multi-Agent API
See [Petting Zoo API](https://pettingzoo.farama.org/content/basic_usage/) for more details.

```python
import sumo_rl
env = sumo_rl.env(net_file='sumo_net_file.net.xml',
                  route_file='sumo_route_file.rou.xml',
                  use_gui=True,
                  num_seconds=3600)  
env.reset()
for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    action = policy(observation)
    env.step(action)
```

### Experiments

WARNING: Gym 0.26 had many breaking changes, stable-baselines3 and RLlib still do not support it, but will be updated soon. See [Stable Baselines 3 PR](https://github.com/DLR-RM/stable-baselines3/pull/780) and [RLib PR](https://github.com/ray-project/ray/pull/28369).
Hence, only the tabular Q-learning experiment is running without erros for now.

Check [experiments](https://github.com/LucasAlegre/sumo-rl/tree/master/experiments) for examples on how to instantiate an environment and train your RL agent.

### [Double Q-learning](https://github.com/linlyu97/Reinforcement-Learning-Approach-for-Intelligent-Traffic-Light-Control/blob/main/experiments/double_ql_single-intersection.py) in a one-way single intersection:
```bash
python experiments/double_ql_single-intersection.py 
```

### [Double Deep Q-network](https://github.com/linlyu97/Reinforcement-Learning-Approach-for-Intelligent-Traffic-Light-Control/blob/main/experiments/deep_q_learning.py) in a one-way single intersection:
```bash
python experiments/deep_q_learning.py 
```

