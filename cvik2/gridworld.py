import numpy as np

class Gridworld:
    def __init__(self, height, width, goal_position=None, traps=[]):
        self.height = height
        self.width = width
        self.goal_position = goal_position
        self.traps = traps

        while self.goal_position is None:
            goal_x = np.random.randint(1, self.width - 1)
            goal_y = np.random.randint(1, self.height - 1)
            self.goal_position = (goal_y, goal_x)
            if self.goal_position in self.traps:
                self.goal_position = None

        self.agent_position = None

    def reset(self):

        while self.agent_position is None:
            goal_x = np.random.randint(1, self.width - 1)
            goal_y = np.random.randint(1, self.height - 1)
            self.agent_position = (goal_y, goal_x)
            if self.agent_position in self.traps or self.agent_position == self.goal_position:
                self.agent_position = None

        return self.agent_position

    def calculate_reward(self, new_state):
        if new_state in self.traps:
            return -10
        if new_state == self.goal_position:
            return 10
        return -1


    def is_done(self):
        if self.agent_position == self.goal_position:
            return True
        if self.agent_position in self.traps:
            return True

        return False

    def step(self, action):
        agent_y, agent_x = self.agent_position
        if action == 0:
            agent_y -= 1
        elif action == 1:
            agent_x += 1
        elif action == 2:
            agent_y += 1
        elif action == 3:
            agent_x -= 1
        else:
            raise ValueError("Unknown action" + str(action))

        if agent_x != 0 and agent_x != self.width - 1 and agent_y != 0 and agent_y != self.height - 1:
            self.agent_position = (agent_y, agent_x)

        reward = self.calculate_reward(self.agent_position)
        done = self.is_done()
        truncated = False
        info = dict()
        return self.agent_position, reward, done, truncated, info


    def render():
        pass

if __name__ == '__main__':
    world = Gridworld(5, 5, goal_position=(3, 3), traps=[(2, 1)])
    state = world.reset()
    done = False
    points = 0
    while not done:
        action = np.random.randint(0, 4)
        new_state, reward, done, truncate, info = world.step(action)
        print("{} - {} -> {}; reward {}; done {}".format(state, action, new_state, reward, done))
        state = new_state
        points += reward
    print("points: {}".format(points))