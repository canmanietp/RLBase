import numpy as np
from collections import deque


class SingleStateAgent:
    bandit_count = 0

    def __init__(self, action_space, alpha, discount):
        self.action_space = action_space
        self.Q_table = np.zeros(action_space)
        self.action_visits = np.zeros(action_space)
        SingleStateAgent.bandit_count += 1
        self.ALPHA = alpha
        self.DISCOUNT = discount
        self.last_n_rewards = [deque(maxlen=10) for a in range(self.action_space)]

    def random_action(self):
        return np.random.randint(self.action_space)

    def e_greedy_action(self, eps):
        if np.random.uniform(0, 1) < eps:
            return self.random_action()
        return self.greedy_action()

    def greedy_action(self):
        qv = self.Q_table
        return np.random.choice(np.flatnonzero(qv == max(qv)))

    def e_greedy_average_action(self, eps):
        if np.random.uniform(0, 1) < eps:
            return self.random_action()

        qv = [np.average(o) for o in self.last_n_rewards]
        if any([math.isnan(v) for v in qv]):
            return self.e_greedy_action(eps)
        return np.random.choice(np.flatnonzero(qv == max(qv)))

    def ucb1_action(self):
        action = self.random_action()
        if np.sum(self.action_visits) > 1:
            qv = self.Q_table + np.sqrt(np.divide(2 * math.log(np.sum(self.action_visits)), self.action_visits))
            action = np.random.choice(np.flatnonzero(qv == max(qv)))
        return action

    def decay(self, decay_rate):
        self.ALPHA *= decay_rate

    def update(self, action, reward):
        self.Q_table[action] += self.ALPHA * (reward - self.Q_table[action])
        self.action_visits[action] += 1
        self.last_n_rewards[action].append(reward)