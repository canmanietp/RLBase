from agents.base import BaseAgent
import numpy as np
import random, copy
from helpers import ranges



class QAgent(BaseAgent):
    def __init__(self, env, params):
        super().__init__(env)
        self.name = 'Q'
        self.params = copy.copy(params)
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.state_variables = list(range(len(self.params.size_state_vars)))
        self.ranges = ranges.get_var_ranges(self, [np.zeros(self.observation_space), self.params.size_state_vars], self.state_variables)

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()
        return np.argmax(self.Q_table[state])

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def update(self, state, action, reward, next_state, done):
        if done:
            td_error = reward - self.Q_table[state][action]
        else:
            td_error = reward + self.params.DISCOUNT * max(self.Q_table[next_state]) - self.Q_table[state][action]

        self.Q_table[state][action] += self.params.ALPHA * td_error
        return td_error

    def do_step(self):
        state = self.current_state
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        self.update(state, action, reward, next_state, done)
        self.current_state = next_state
        return reward, done


class QMiniAgent(QAgent):
    def __init__(self, env, params, mini_observation_space, mini_action_space):
        super().__init__(env, params)
        self.name = 'QMini'
        self.params = copy.copy(params)
        self.observation_space = mini_observation_space
        self.action_space = mini_action_space
        self.Q_table = np.zeros([self.observation_space, self.action_space])
