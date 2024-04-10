import numpy as np


class CarRentalEnvironment:
    def __init__(self, lambda_rent=[3, 4], lambda_return=[3, 2], max_cars=20, max_transfer=5, rent_cost=10,
                 transfer_cost=2):
        self.max_cars = max_cars
        self.max_transfer = max_transfer
        self.rent_cost = rent_cost
        self.transfer_cost = transfer_cost
        self.lambda_rent = lambda_rent
        self.lambda_return = lambda_return
        self.state = [10, 10]  # начальное состояние: 10 автомобилей в каждом филиале
        self.history = list()

    def reset(self):
        self.state = [10, 10]
        return self.state

    def step(self, action):
        # Проверяем, что действие в допустимом диапазоне
        action = max(-self.max_transfer, min(self.max_transfer, action))

        # Выполняем перенос автомобилей между филиалами, учитывая стоимость переноса
        transferred_cars = min(self.state[0] if action > 0 else self.state[1], abs(action))
        self.state[0] -= transferred_cars if action > 0 else 0
        self.state[1] += transferred_cars if action > 0 else -transferred_cars
        transfer_cost = abs(transferred_cars) * self.transfer_cost

        rent_rewards = 0
        for i in range(2):
            # Случайное число арендованных автомобилей, основанное на распределении Пуассона
            rent_requests = np.random.poisson(self.lambda_rent[i])
            rent_completed = min(self.state[i], rent_requests)
            self.state[i] -= rent_completed

            # Подсчет вознаграждения за аренду
            rent_rewards += rent_completed * self.rent_cost

            # Случайное число возвращенных автомобилей, также основанное на распределении Пуассона
            returned_cars = np.random.poisson(self.lambda_return[i])
            self.state[i] = min(self.max_cars, self.state[i] + returned_cars)

        reward = rent_rewards - transfer_cost
        return self.state, reward

    def render(self):
        print(f"Location 1 Cars: {self.state[0]}, Location 2 Cars: {self.state[1]}")

def simple_policy(state):
    """
    Простая политика, где агент всегда пытается перевезти 1 автомобиль из первого филиала во второй,
    если в первом филиале больше автомобилей.
    """
    return 1 if state[0] > state[1] else -1

def train_agent(environment, episodes=1000, policy=simple_policy):
    total_rewards = 0
    for episode in range(episodes):
        state = environment.reset()
        done = False
        while not done:
            action = policy(state)
            next_state, reward = environment.step(action)
            print("Old state: {},\tnew state: {}".format(state, next_state))
            total_rewards += reward
            if next_state == state:  # Простая проверка на завершение эпизода
                done = True
            state = next_state
    average_reward = total_rewards / episodes
    return average_reward

if __name__ == "__main__":
    env = CarRentalEnvironment()
    average_reward = train_agent(env)
    print(f"Среднее вознаграждение после обучения: {average_reward:.2f}")