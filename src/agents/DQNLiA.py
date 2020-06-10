import random
import numpy as np
import copy
from agents.DQN import DQNAgent, DQNMiniAgent


class DQNLiAAgent(DQNAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'DQNLiA'
        self.sub_agents = []

        for isx, ss in enumerate(params.sub_spaces):
            sub_params = copy.copy(params)
            sub_params.INIT_MODEL = sub_params.sub_models[isx]
            sub_params.EPSILON = sub_params.PHI
            sub_params.EPSILON_MIN = sub_params.PHI_MIN
            sub_params.observation_space = len(ss)
            sub_params.action_space = sub_params.action_space
            self.sub_agents.append(DQNMiniAgent(env, sub_params))

        self.params.action_space = len(params.sub_spaces)
        self.action_space = self.params.action_space
        self.model = self.params.META_MODEL
        self.target_model = copy.copy(self.params.META_MODEL)

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        for ab in self.sub_agents:
            ab.decay(decay_rate)

    def step_LiA(self, abstraction, action):
        next_state, reward, done, next_state_info = self.env.step(action)
        if 'AtariARIWrapper' in str(self.env):
            next_state = self.info_into_state(next_state_info, None)
        temp = copy.copy(self.last_n_states[int(self.params.observation_space/self.params.REPEAT_N_FRAMES):])
        temp = np.append(temp, next_state)
        next_last_n_states = temp

        if len(self.last_n_states) == self.params.REPEAT_N_FRAMES:
            self.remember(np.reshape(self.last_n_states, [1, self.params.observation_space]), action, reward,
                          np.reshape(next_last_n_states, [1, self.params.observation_space]), done)

        for sax, sa in enumerate(self.sub_agents):
            # abs_last_two_states = self.last_two_states[self.params.sub_spaces[sax]]
            abs_last_n_states = self.last_n_states[self.params.sub_spaces[sax]]
            # abs_next_last_two_states = next_last_two_states[self.params.sub_spaces[sax]]
            abs_next_last_n_states = next_last_n_states[self.params.sub_spaces[sax]]
            sa.remember(np.reshape(abs_last_n_states, [1, sa.params.observation_space]), action, reward, np.reshape(abs_next_last_n_states, [1, sa.params.observation_space]), done)
        self.current_state = next_state
        self.last_n_states = next_last_n_states
        return next_state, reward, done

    def e_greedy_LiA_action(self, state):
        ab_index = self.e_greedy_action(state)
        abs_state = list(state[0][self.params.sub_spaces[ab_index]])
        abs_state = np.reshape(abs_state, [1, len(self.params.sub_spaces[ab_index])])
        action = self.sub_agents[ab_index].e_greedy_action(abs_state)
        return ab_index, action

    def replay_DQNLIA(self):
        for ia, ab in enumerate(self.sub_agents):
            ab.replay()
        self.replay()

    def do_step(self):
        state = np.reshape(self.last_n_states, [1, self.params.observation_space])
        abstraction, action = self.e_greedy_LiA_action(state)
        next_state, reward, done = self.step_LiA(abstraction, action)
        if len(self.memory) > self.params.BATCH_SIZE:
            self.replay_DQNLIA()
        if done:
            self.decay(self.params.DECAY_RATE)
        return reward, done

    def pseudo_count(self, state):
        count = [0. for a in list(range(self.action_space))]
        if len(self.memory) > 1000:
            minibatch = random.sample(self.memory, 1000)
            for s, a, _, _, _ in minibatch:
                norm = 0.
                for p, sd in enumerate(state[0]):
                    norm += abs(sd - s[0][p])
                count[a] += max(0., 1. - (norm / 0.1))
        return count
