import numpy as np


class BaseAgent:
    def __init__(self, env):
        self.name = 'Base'
        self.env = env
        self.observation_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.sa_visits = np.zeros([self.observation_space, self.action_space])
        self.current_state = self.reset()

    def reset(self):
        self.current_state = self.env.reset()
        return self.current_state

    def random_action(self):
        return np.random.randint(self.action_space)

    def get_state(self):
        return self.current_state

    def set_state(self, state):
        if hasattr(self.env, 'env'):
            self.env.env.s = state
        else:
            self.env.s = state
        self.current_state = state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)

        self.sa_visits[self.current_state][action] += 1
        self.current_state = next_state

        return next_state, reward, done

