import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

PRELOAD = False

# env = gym.make('Pendulum-v1', render_mode="human", max_episode_steps=600); env.metadata['render_fps'] = 60
env = gym.make('Pendulum-v1', max_episode_steps=600)

episodes = 15000
discount_rate = 0.95

if not PRELOAD:
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99999
else:
    epsilon = 0.1
    epsilon_min = 0.01
    epsilon_decay = 0.99999

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
    q_table = np.random.uniform(low=-1, high=-0, size=(observation_space_size + [action_space_size]))
    # q_table = np.zeros(observation_space_size + [action_space_size])
    returns_table = np.zeros(observation_space_size + [action_space_size])
    returns_table_count = np.zeros(observation_space_size + [action_space_size])
else:
    q_table = np.load("D:/Programming/MachineLearning2/Zadanie1/MC_q_table_pendulum.npy")
    returns_tables = np.load("D:/Programming/MachineLearning2/Zadanie1/returns_tables.npy")
    returns_table = returns_tables[0]
    returns_table_count = returns_tables[1]

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

for episode in range(1, episodes+1):
    state = env.reset()
    state = state[0]
    episode_history = list()
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

        index_state, index_velocity = get_indexes(state)
        new_index_state, new_index_velocity = get_indexes(observation)

        episode_history.append((index_state, index_velocity, action, reward))

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if truncated:
            state = env.reset()
            break

    G = 0
    for new_index_state, new_index_velocity, action, reward in reversed(episode_history):
        G = discount_rate * G + reward
        # If the state-action pair is detected for the first time in this episode, we update the Q-value.
        if (new_index_state, new_index_velocity, action) not in episode_history:
            returns_table[new_index_state, new_index_velocity, action] += G
            returns_table_count[new_index_state, new_index_velocity, action] += 1
            q_table[new_index_state, new_index_velocity, action] = \
                (returns_table[new_index_state, new_index_velocity, action] /
                 returns_table_count[new_index_state, new_index_velocity, action])

    if episode % 500 == 0:
        print("Episode: {}".format(episode))
        # print("Total reward = {}".format(total_reward))
        print("Average reward = {}\n".format(total_reward/episode))

    total_reward_for_episode.append(episode_reward)

plt.plot(total_reward_for_episode)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

returns_tables = [returns_table, returns_table_count]
np.save("D:/Programming/MachineLearning2/Zadanie1/MC_q_table_pendulum", q_table)
np.save("D:/Programming/MachineLearning2/Zadanie1/returns_tables", returns_tables)
