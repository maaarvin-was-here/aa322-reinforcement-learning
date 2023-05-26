import numpy as np
import gymnasium as gym

env = gym.make("CartPole-v0")
env.reset()



max_number_of_steps = 200  # 每一场游戏的最高得分
# the max reward for every episode
# 获胜的条件是最近100场平均得分高于195
# if the reward of recent 100 episodes are higher than 195
# means the agent win
goal_average_steps = 100
num_consecutive_iterations = 100

# we will have 200 episode in total
# it should be higher but colab will crash if I do that much
num_episodes = 600  # 共进行600场游戏
last_time_steps = np.zeros(num_consecutive_iterations)
# Only store the reward of the most recent 100 episodes

q_table = np.random.uniform(low=-1, high=1, size=(4 ** 4, env.action_space.n))


def bins(clip_min, clip_max, num):
    return np.linspace(clip_min, clip_max, num + 1)[1:-1]


def digitize_state(observation):
    cart_pos = observation[0]
    cart_v = observation[1]
    pole_angle = observation[2]
    pole_v = observation[3]
    digitized = [np.digitize(cart_pos, bins=bins(-2.4, 2.4, 4)),
                 np.digitize(cart_v, bins=bins(-3.0, 3.0, 4)),
                 np.digitize(pole_angle, bins=bins(-0.5, 0.5, 4)),
                 np.digitize(pole_v, bins=bins(-2.0, 2.0, 4))]
    return sum([x * (4 ** i) for i, x in enumerate(digitized)])


def get_action(state, action, observation, reward, episode):
    next_state = digitize_state(observation)
    # set up epsilon 'ε' in greedy strategy
    epsilon = 0.5 * (0.99 ** episode)  # 0.5 initial
    # 随着episode增大，epsilon会减小
    if epsilon <= np.random.uniform(0, 1):
        next_action = np.argmax(q_table[next_state])  # With probability (1-epsilon), we take greedy action
    else:
        next_action = np.random.choice([0, 1])  # else, take random action
    # 小于探索率就继续 大于就利用
    # learning and update Q-table
    alpha = 0.2
    gamma = 0.99
    q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (
                reward + gamma * q_table[next_state, next_action])

    return next_action, next_state


for episode in range(num_episodes):
    observation = env.reset()
    state = digitize_state(observation[0])  # Discretize the initial state
    action = np.argmax(q_table[state])  # Get the optimal action from the Q-table
    episode_reward = 0

    for t in range(max_number_of_steps):
        env.render()
        # frames4.append(env.render(mode='rgb_array'))
        observation, reward, done, info, _ = env.step(action)
        # 对致命错误行动进行极大力度的惩罚，让模型恨恨地吸取教训
        # set up punishment on the error(wrong action)
        # make the model learn
        if done:
            reward = -100  # If failed, negative reward
        action, state = get_action(state, action, observation, reward, episode)  # 作出下一次行动的决策
        episode_reward += reward
        if done:
            print('%d Episode finished after %f time steps / mean %f' % (episode, t + 1, last_time_steps.mean()))
            last_time_steps = np.hstack((last_time_steps[1:], [t+1]))
            break

        if (last_time_steps.mean() >= goal_average_steps):
            print('Episode %d train agent successfuly!' % episode)
            break


