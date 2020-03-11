import numpy as np
import copy
from agents.DQN import DQNAgent


class DQNIiBAgent(DQNAgent):
    def __init__(self, env, params, meta_model):
        super().__init__(env, params)
        self.name = 'IiB'
        self.meta_model = meta_model
