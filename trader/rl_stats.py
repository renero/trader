import time


class RLStats:

    def __init__(self):
        self.avg_rewards = []
        self.avg_loss = []
        self.avg_mae = []
        self.avg_profit = []
        self.last_avg: float = 0.0
        self.sum_rewards = 0
        self.sum_loss = 0
        self.sum_mae = 0
        self.start = time.time()

    def reset(self):
        self.sum_rewards = 0
        self.sum_loss = 0
        self.sum_mae = 0

    def step(self, loss, mae, reward):
        # Update states and metrics
        self.sum_loss += loss
        self.sum_mae += mae
        self.sum_rewards += reward

    def update(self, num_episodes, last_profit):
        self.avg_rewards.append(self.sum_rewards / num_episodes)
        self.avg_loss.append(self.sum_loss / num_episodes)
        self.avg_mae.append(self.sum_mae / num_episodes)
        self.avg_profit.append(last_profit)