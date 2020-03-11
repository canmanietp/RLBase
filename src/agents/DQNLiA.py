import numpy as np
import copy
from agents.DQN import DQNAgent


class DQNLiAAgent(DQNAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'DQNLiA'
        self.sub_agents = []

        for isx, ss in enumerate(params.sub_spaces):
            sub_params = copy.copy(params)
            sub_params.observation_space = len(ss)
            sub_params.INIT_MODEL = params.sub_models[isx]
            sub_params.EPSILON = params.PHI
            sub_params.EPSILON_MIN = params.PHI_MIN
            self.sub_agents.append(DQNAgent(env, sub_params))

        self.action_space = len(params.sub_spaces)
        self.model = params.META_MODEL
        self.target_model = params.META_MODEL

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        for ab in self.sub_agents:
            ab.decay(decay_rate)

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.remember(np.reshape(self.current_state, [1, self.params.observation_space]), action, reward, np.reshape(next_state, [1, self.params.observation_space]), done)
        for sax, sa in enumerate(self.sub_agents):
            abs_state = self.current_state[self.params.sub_spaces[sax]]
            abs_next_state = next_state[self.params.sub_spaces[sax]]
            sa.remember(np.reshape(abs_state, [1, sa.params.observation_space]), action, reward, np.reshape(abs_next_state, [1, sa.params.observation_space]), done)
        self.current_state = next_state
        return next_state, reward, done

    def e_greedy_LIA_action(self, state):
        ab_index = self.e_greedy_action(state)
        print("ab {}".format(ab_index))
        abs_state = state[self.params.sub_spaces[ab_index]]
        action = self.sub_agents[ab_index].e_greedy_action(abs_state)
        print("action {}".format(action))
        return ab_index, action

    def replay_DQNLIA(self):
        for ia, ab in enumerate(self.sub_agents):
            ab.replay()
        self.replay()

    def run_episode(self):
        state = np.reshape(self.current_state, [1, self.params.observation_space])
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        if len(self.memory) > self.params.BATCH_SIZE:
            self.replay_DQNLIA()
        return reward, done
