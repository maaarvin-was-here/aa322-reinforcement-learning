"""
import gymnasium as gym
env = gym.make("FrozenLake-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
"""


import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
# make environment
env = gym.make('FrozenLake-v1', desc=generate_random_map(size=6), is_slippery=False, render_mode="human")

observation, info = env.reset()

# as the environment is continues there cannot be finite number of states
states = env.observation_space.n  # used if discrete environment

# check number of actions that can be
actions = env.action_space.n
# initialize value table randomly
value_table = np.zeros((states, 1))


def value_iterations(env, n_iterations, gamma=0.99, threshold=1e-30):
    for i in range(n_iterations):
        new_valuetable = np.copy(value_table)
        for state in range(states):
            q_value = []
            for action in range(actions):
                next_state_reward = []
                for next_state_parameters in env.env.P[state][action]:
                    transition_prob, next_state, reward_prob, _ = next_state_parameters
                    print(next_state_parameters)

                    reward = transition_prob * (reward_prob + gamma * new_valuetable[next_state])
                    next_state_reward.append(reward)

                q_value.append((np.sum(next_state_reward)))
            value_table[state] = max(q_value)
    return value_table


def extract_policy(value_table, gamma=1.0):
    policy = np.zeros(env.observation_space.n)
    for state in range(env.observation_space.n):
        Q_table = np.zeros(env.action_space.n)
        for action in range(env.action_space.n):
            for next_sr in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (transition_prob * (reward_prob + gamma * value_table[next_state]))
        policy[state] = np.argmax(Q_table)
        #print(Q_table)
    return policy


value_table = value_iterations(env, 10)
print(value_table)
policy = extract_policy(value_table)
print(policy)

observation = 0

for i in range(10000):
    action = policy[observation]  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(int(action))
    if terminated or truncated:
        observation, info = env.reset()
        break
