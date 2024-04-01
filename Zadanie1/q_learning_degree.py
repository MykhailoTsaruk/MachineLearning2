import gymnasium as gym
import keyboard
import numpy as np
import pickle

# env = gym.make('Pendulum-v1', g=9.81, render_mode="human", max_episode_steps=600)
env = gym.make('Pendulum-v1', g=9.81)
env.metadata['render_fps'] = 60


learning_rate = 0.1
discount_rate = 0.9

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999

episodes = 10000
total_reward = 0

action_space_size = 41  # count of action [-2 : 2] with 0.1 step, 0->19 = [-2 : -0.1], 20 = 0, 21->40 = [0.1 : 2]
observation_space_size = [63, 161]
"""
First is count of pendulum state [-pi : pi] with 0.1 step, 0->31 = [-pi : -pi/31], 32 = 0, 33->62 = [pi/32 : pi]
Second is count of pendulum angular velocity [-8 : 8] with 0.1 step, 0->79 = [-8 : -0.1], 80 = 0, 81->160 = [0.1 : 8]
"""

action_space = np.linspace(-2, 2, num=action_space_size)
observation_space = [np.linspace(-np.pi, np.pi, num=observation_space_size[0]),
                     np.linspace(-8.0, 8.0, num=observation_space_size[1])]
# print(action_space)
# print(observation_space)

q_table = np.random.uniform(low=-0.25, high=0.25, size=(observation_space_size + [action_space_size]))

def get_discrete_action(state):
    theta = np.arctan2(state[0], state[1])
    index_state = np.digitize(theta, observation_space[0])
    index_velocity = np.digitize(state[0], observation_space[1])
    action = q_table[index_state, index_velocity]
    return action

def get_indexes(state):
    index_state = np.digitize(np.arctan2(state[0], state[1]), observation_space[0])
    index_velocity = np.digitize(q_table[index_state], observation_space[1])
    return [index_state, index_velocity]

for episode in range(episodes):
    state = env.reset()
    state = state[0]
    done = False
    episode_reward = 0

    print("Episode: {}".format(episode))
    print("Total reward = {}".format(total_reward))

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(get_discrete_action(state))
            # print("action={}".format(action))
        else:
            action = np.random.randint(0, action_space_size)
            # print("\nRANDOM!!action={}\n".format(action))

        step = [action_space[action]]

        observation, reward, done, truncated, *info = env.step(step)
        total_reward += reward
        episode_reward += reward

        index_state, index_velocity = get_indexes(state)
        max_future_q = np.max(q_table[index_state, index_velocity])
        current_q = q_table[index_state, index_velocity, action]
        # new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_rate * max_future_q)
        new_step = action_space[np.argmax(get_discrete_action(observation))]
        new_q = current_q + learning_rate * (reward + discount_rate * new_step - current_q)
        q_table[index_state, index_velocity, action] = new_q
        state = observation

        if (state[0] == 0 and state[1] == 1) or done:
            q_table[index_state, index_velocity, action] = 0
            print(f"Acheived in {episode}")
            np.save("D:/Programming/MachineLearning2/Zadanie1/q_table_pendulum", q_table)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        # if truncated:
        #     state = env.reset()
        #     break
    print("Episode reward = {}\n".format(episode_reward))

