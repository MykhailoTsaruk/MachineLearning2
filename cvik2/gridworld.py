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
        self.walls = []

    def generate_traps(self):
        height = self.height - 2
        width = self.width - 2

        count = int((height * width) / 15)
        if count == 0:
            return []

        for _ in range(count):
            trap = None
            while trap is None:
                trap_y = np.random.randint(1, self.height - 1)
                trap_x = np.random.randint(1, self.width - 1)
                trap = (trap_y, trap_x)
                if trap in self.traps or trap == self.goal_position or trap in self.walls:
                    trap = None

            self.traps.append(trap)

        return self.traps


    def generate_wall(self):
        height = self.height - 2
        width = self.width - 2

        count = int((height * width) / 9)
        if count == 0:
            return []

        for _ in range(count):
            wall = None
            while wall is None:
                wall_y = np.random.randint(1, self.height - 1)
                wall_x = np.random.randint(1, self.width - 1)
                wall = (wall_y, wall_x)
                if wall in self.traps or wall == self.goal_position or wall in self.walls:
                    wall = None

            self.walls.append(wall)

        return self.walls

    def reset(self):

        while self.agent_position is None:
            goal_x = np.random.randint(1, self.width - 1)
            goal_y = np.random.randint(1, self.height - 1)
            self.agent_position = (goal_y, goal_x)
            if self.agent_position in self.traps or self.agent_position == self.goal_position or self.agent_position in self.walls:
                self.agent_position = None

        return self.agent_position

    def calculate_reward(self, new_state):
        if self.agent_position in self.walls:
            return 0

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

        isWall = False
        if agent_x != 0 and agent_x != self.width - 1 and agent_y != 0 and agent_y != self.height - 1 and (agent_x, agent_y) not in self.walls:
            self.agent_position = (agent_y, agent_x)
        else:
            isWall = True

        reward = self.calculate_reward(self.agent_position)
        done = self.is_done()
        truncated = False
        info = dict()
        return self.agent_position, reward, done, truncated, info, isWall


    def render(self, agent_state):
        height = self.height
        width = self.width

        field = [['#' for i in range(0, width)] for _ in range(0, height)]
        for i in range(1, height-1):
            for j in range(1, width-1):
                field[i][j] = ' '

            for trap in self.traps:
                y, x = trap
                field[y][x] = 'O'

            for wall in self.walls:
                y, x = wall
                field[y][x] = '#'

        y_goal, x_goal = self.goal_position
        field[y_goal][x_goal] = 'X'

        y_agent, x_agent = agent_state
        field[y_agent][x_agent] = '.'

        for _ in range(0, height):
            a = str()
            for i in range(0, width):
                a += field[_][i]
            print(a)
        print()


if __name__ == '__main__':
    world = Gridworld(5, 9, goal_position=(3, 3))
    walls = world.generate_wall()
    traps = world.generate_traps()
    state = world.reset()
    done = False
    points = np.abs(world.goal_position[0] - state[0]) + np.abs(world.goal_position[1] - state[1])
    print('Distance to goal: {}'.format(points))
    while not done:
        world.render(state)
        action = np.random.randint(0, 4)
        new_state, reward, done, truncate, info, iswall = world.step(action)
        direction = str()
        match action:
            case 0:
                direction = 'N'

            case 1:
                direction = 'E'

            case 2:
                direction = 'S'

            case 3:
                direction = 'W'

        print("{} - {} -> {}; reward {}; score {}; done {}".format(state, direction, new_state, reward, (points + reward), done))
        if iswall:
            print("Wall!")
        state = new_state
        points += reward

    world.render(state)
    print("points scored: {}".format(points))
