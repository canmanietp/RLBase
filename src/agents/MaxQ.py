from agents.base import BaseAgent
import numpy as np
import random, copy


class MaxQAgent(BaseAgent):
    def __init__(self, env, params, options):
        super().__init__(env)
        self.name = 'MaxQ'
        self.params = copy.copy(params)
        self.options = options
        self.Q_table = np.zeros([self.observation_space, self.action_space])
