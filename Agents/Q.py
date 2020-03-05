from Agents.tabular import TabularAgent
import numpy as np
import random


class QAgent(TabularAgent):
    def __init__(self, env, alpha, alpha_min, epsilon, epsilon_min, discount):
        super().__init__(env)
        self.name = 'Q'
        self.ALPHA = alpha
        self.ALPHA_MIN = alpha_min
        self.EPSILON = epsilon
        self.EPSILON_MIN = epsilon_min
        self.DISCOUNT = discount
        self.Q_table = np.zeros([self.observation_space, self.action_space])

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.EPSILON:
            return self.random_action()

        return np.argmax(self.Q_table[state])

    def decay(self, decay_rate):
        if self.ALPHA > self.ALPHA_MIN:
            self.ALPHA *= decay_rate
        if self.EPSILON > self.EPSILON_MIN:
            self.EPSILON *= decay_rate

    def update(self, state, action, reward, next_state, done):
        if done:
            self.Q_table[state][action] += self.ALPHA * (reward - self.Q_table[state][action])
        else:
            self.Q_table[state][action] += self.ALPHA * (
                reward + self.DISCOUNT * max(self.Q_table[next_state]) - self.Q_table[state][action])

    def run_episode(self):
        state = self.current_state
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        self.update(state, action, reward, next_state, done)
        return reward, done


class QMiniAgent(QAgent):
    def __init__(self, env, alpha, alpha_min, epsilon, epsilon_min, discount, mini_observation_space, mini_action_space):
        super().__init__(env, alpha, alpha_min, epsilon, epsilon_min, discount)
        self.name = 'Mini'
        self.observation_space = mini_observation_space
        self.action_space = mini_action_space
        self.Q_table = np.zeros([self.observation_space, self.action_space])
