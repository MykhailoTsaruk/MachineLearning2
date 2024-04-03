import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

PRELOAD = True

env = gym.make('Pendulum-v1', render_mode="human", max_episode_steps=600); env.metadata['render_fps'] = 60
# env = gym.make('Pendulum-v1', max_episode_steps=600)

learning_rate = 0.1
discount_rate = 0.95

if not PRELOAD:
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.999
else:
    epsilon = 0.15
    epsilon_min = 0.05
    epsilon_decay = 0.999

episodes = 150000
total_reward = 0

action_space_size = 41
""" count of action [-2 : 2] with 0.1 step, 0->19 = [-2 : -0.1], 20 = 0, 21->40 = [0.1 : 2] """
observation_space_size = [63, 161]
"""
First is count of pendulum state [-pi : pi] with 0.1 step, 0->31 = [-pi : -pi/31], 32 = 0, 33->62 = [pi/32 : pi]
Second is count of pendulum angular velocity [-8 : 8] with 0.1 step, 0->79 = [-8 : -0.1], 80 = 0, 81->160 = [0.1 : 8]
"""

action_space = np.linspace(-2, 2, num=action_space_size)
observation_space = [np.linspace(-np.pi, np.pi, num=observation_space_size[0]),
                     np.linspace(-8.0, 8.0, num=observation_space_size[1])]

if not PRELOAD:
    q_table = np.random.uniform(low=-2, high=-0, size=(observation_space_size + [action_space_size]))
else:
    q_table = np.load("D:/Programming/MachineLearning2/Zadanie1/TD_q_table_pendulum.npy")

def get_discrete_action(state):
    theta = np.arctan2(state[0], state[1])
    index_state = np.digitize(theta, observation_space[0]) - 1
    index_velocity = np.digitize(state[2], observation_space[1]) - 1
    action = q_table[index_state, index_velocity]
    return action

def get_indexes(state):
    theta = np.arctan2(state[0], state[1])
    index_state = np.digitize(theta, observation_space[0]) - 1
    index_velocity = np.digitize(state[2], observation_space[1]) - 1
    return [index_state, index_velocity]

total_reward_for_episode = list()

for episode in range(episodes):
    state = env.reset()
    state = state[0]
    done = False
    episode_reward = 0

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(get_discrete_action(state))
        else:
            action = np.random.randint(0, action_space_size)

        step = [action_space[action]]

        observation, reward, done, truncated, *info = env.step(step)

        total_reward += reward
        episode_reward += reward

        # Change q table
        index_state, index_velocity = get_indexes(state)
        new_index_state, new_index_velocity = get_indexes(observation)
        new_action = np.argmax(get_discrete_action(observation))

        current_q = q_table[index_state, index_velocity, action]
        new_step = np.max(q_table[new_index_state, new_index_velocity])
        new_q = current_q + learning_rate * (reward + discount_rate * new_step - current_q)
        q_table[index_state, index_velocity, action] = new_q
        state = observation

        # Change epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # Truncated
        if truncated:
            state = env.reset()
            break

    if episode % 1000 == 0:
        print("Episode: {}".format(episode))
        print("Total reward = {}".format(total_reward))
        print("Average reward = {}".format(total_reward/episode))

plt.plot(total_reward_for_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()


np.save("D:/Programming/MachineLearning2/Zadanie1/q_table_pendulum", q_table)