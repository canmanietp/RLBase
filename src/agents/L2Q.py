from agents.base import BaseAgent
from agents.Q import QMiniAgent
import numpy as np
import random, copy


class L2QAgent(BaseAgent):
    def __init__(self, env, params):
        super().__init__(env)
        self.name = 'L2Q'
        self.params = copy.copy(params)
        self.Q_table = np.zeros([self.observation_space, self.action_space])
        self.mini_observation_space = 125
        self.mini_space = [0, 1, 2]
        self.mini_agent = QMiniAgent(self.env, self.params, self.mini_observation_space, self.action_space + 1)

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
        mini_state = self.encode_abs_state(list(self.env.decode(state)), self.mini_space)
        action = self.mini_agent.e_greedy_action(mini_state)
        query = False
        if action == self.action_space:  # A QUERY ACTION
            action = self.e_greedy_action(state)
            query = True
        next_state, reward, done = self.step(action)
        if query:
            self.mini_agent.update(mini_state, self.action_space, reward,
                                   self.encode_abs_state(list(self.env.decode(next_state)), self.mini_space), done)
            self.update(state, action, reward, next_state, done)
        else:
            self.mini_agent.update(mini_state, action, reward,
                                   self.encode_abs_state(list(self.env.decode(next_state)), self.mini_space), done)

        return reward, done

    def encode_abs_state(self, state, abstraction):
        abs_state = [state[k] for k in abstraction]
        var_size = copy.copy([self.params.size_state_vars[k] for k in abstraction])
        var_size.pop(0)
        encoded_state = 0

        for e in range(len(abs_state) - 1):
            encoded_state += abs_state[e] * np.prod(var_size)
            var_size.pop(0)

        encoded_state += abs_state[-1]
        return encoded_state
