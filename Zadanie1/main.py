import gymnasium as gym
import keyboard
import numpy as np

env = gym.make('Pendulum-v1', g=9.81, render_mode="human")
# env = gym.make('Pendulum-v1', g=9.81)
env.metadata['render_fps'] = 60

learning_rate = 0.15
discount_rate = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.95
episodes = 10000
total_reward = 0

action_space_size = 41
observation_space = [21, 21, 161]
observation_space_size = np.prod(observation_space)

discrete_action_space_win_size = (env.action_space.high - env.action_space.low) / (action_space_size - 1)
action_space = {}
for i in range(action_space_size):
    action_space[i] = [env.action_space.low[0] + (i * discrete_action_space_win_size[0])]

q_table = np.random.uniform(low=-2, high=-0, size=(observation_space + [action_space_size]))

# q_table_list = []
def get_discrete_state(state):
    discrete_state = []
    for i in range(len(state)):
        # Scale the continuous state value to the range [0, number of bins - 1]
        scaling = (state[i] - env.observation_space.low[i]) / (env.observation_space.high[i] - env.observation_space.low[i])
        new_state = int(scaling * (observation_space[i] - 1))
        discrete_state.append(new_state)
    return tuple(discrete_state)

for episode in range(episodes):
    state = env.reset()
    discrete_state = get_discrete_state(state[0])
    done = False
    episode_reward = 0

    print("Episode: {}".format(episode))
    print("Total reward = {}".format(total_reward))

    while not done:
        # q_table_list.append(q_table)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
            # print(action_space)
            # print(action_space[action])
            # print("action={}".format(action))
        else:
            action = np.random.randint(0, action_space_size)
            # print("\nRANDOM!!\naction={}\n".format(action))

        action_continuous = action_space[action]
        print(action_continuous)
        # action_continuous = [action / (action_space_size - 1) * 4 - 2]
        observation, reward, done, truncated, *info = env.step(action_continuous)
        total_reward += reward
        episode_reward += reward

        new_discrete_state = get_discrete_state(observation)
        # print(new_discrete_state)
        # print(action)
        # print(q_table[new_discrete_state + (action, )])
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]
        new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount_rate * max_future_q)
        # print(new_q)
        q_table[discrete_state + (action, )] = new_q
        discrete_state = new_discrete_state
        # print("new state = {}".format(new_discrete_state))
        # print("action = {}".format(action))
        # print("current = {}".format(current_q))
        # print("new = {}".format(new_q))

        # print("x={} y={} velocity={}".format(observation[0], observation[1], observation[2]))
        # print(observation)
        # print("reward={}".format(reward))
        # keyboard.wait('f')

        # keyboard.wait('f')

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if observation[0] == 0 and observation[1] == 1:
            q_table[discrete_state + (action,)] = 0
            print(f"Acheived in {episode}")

        # if truncated:
        #     observation, info = env.reset()
        #     break

    print("Episode reward = {}\n".format(episode_reward))

env.close()

