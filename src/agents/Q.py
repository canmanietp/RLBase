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
        self.sa_next_state = [[[] for a in range(self.action_space)] for os in range(self.observation_space)]
        self.sa_reward = [[[] for a in range(self.action_space)] for os in range(self.observation_space)]

        self.max_steps = 1
        self.steps = 0

    def greedy_action(self, state):
        if self.inadmissible_actions[state]:
            admissible_actions = [x for x in range(self.action_space) if x not in self.inadmissible_actions[state]]
        else:
            admissible_actions = None
        if admissible_actions:
            qv = self.Q_table[state][admissible_actions]
            val_choice = np.random.choice(np.flatnonzero(qv == qv.max()))
            return admissible_actions[val_choice]
        else:
            qv = self.Q_table[state]
        return np.random.choice(np.flatnonzero(qv == qv.max()))

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action(state)
        if all([x == 0 for x in self.Q_table[state]]):
            return self.random_action(state)
        return self.greedy_action(state)

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def update(self, state, action, reward, next_state, done):
        td_error = reward + (not done) * self.params.DISCOUNT * max(self.Q_table[next_state]) - self.Q_table[state][action]
        self.Q_table[state][action] += self.params.ALPHA * td_error
        return td_error

    def sarsa_update(self, state, action, next_action, reward, next_state, done):
        td_error = reward + (not done) * self.params.DISCOUNT * self.Q_table[next_state, next_action] - self.Q_table[state][action]
        self.Q_table[state][action] += self.params.ALPHA * td_error
        return next_action

    def do_step(self):
        state = self.current_state
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        if 'SysAdmin' in str(self.env) and self.steps > self.max_steps:
            done = True
        self.update(state, action, reward, next_state, done)
        self.current_state = next_state
        self.steps += 1
        return reward, done


class QMiniAgent(QAgent):
    def __init__(self, env, params, mini_observation_space, mini_action_space):
        super().__init__(env, params)
        self.name = 'QMini'
        self.params = copy.copy(params)
        self.observation_space = mini_observation_space
        self.action_space = mini_action_space
        self.Q_table = np.zeros([self.observation_space, self.action_space])
