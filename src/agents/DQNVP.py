import numpy as np
import copy, random
from agents.DQN import DQNAgent, DQNMiniAgent


class DQNVPAgent(DQNAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'QVP'

        self.model = params.INIT_MODEL
        self.target_model = params.INIT_MODEL

        meta_params = copy.copy(params)
        meta_params.INIT_MODEL = meta_params.META_MODEL
        meta_params.EPSILON = meta_params.PHI
        meta_params.EPSILON_MIN = meta_params.PHI_MIN
        meta_params.action_space = len(meta_params.sub_spaces)
        self.meta_agent = DQNMiniAgent(self.env, meta_params)

        self.num_samples = 10
        self.d0 = 1.  # For pseudo count, should be a parameter for QVP

    def step_VP(self, abstraction, action):
        next_state, reward, done, next_state_info = self.env.step(action)
        if 'AtariARIWrapper' in str(self.env):
            next_state = self.info_into_state(next_state_info, None)
        self.remember(np.reshape(self.current_state, [1, self.params.observation_space]), action, reward, np.reshape(next_state, [1, self.params.observation_space]), done)
        self.meta_agent.remember(np.reshape(self.current_state, [1, self.params.observation_space]), abstraction, reward, np.reshape(next_state, [1, self.params.observation_space]), done)
        self.current_state = next_state
        return next_state, reward, done

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        self.meta_agent.decay(decay_rate)

    def e_greedy_VP_action(self, state):
        ab_index = self.meta_agent.e_greedy_action(state)

        if random.uniform(0, 1) < self.params.PHI or len(self.memory) < self.num_samples:
            return ab_index, self.random_action()
        else:
            states = self.sample_states(state, self.params.sub_spaces[ab_index])
            q_values = []
            visits = []
            for s in states:
                q_values.append(self.model.predict(s)[0])
                visits.append(self.pseudo_count(s))

            qs = np.zeros(self.action_space)

            for a in range(self.action_space):
                most_visited = int(np.argmax([item[a] for item in visits]))
                qs[a] = q_values[most_visited][a]

            return ab_index, np.argmax(qs)

    def replay_DQNVP(self):
        self.replay()
        self.meta_agent.replay()

    def do_step(self):
        state = np.reshape(self.current_state, [1, self.params.observation_space])
        abstraction, action = self.e_greedy_VP_action(state)
        next_state, reward, done = self.step_VP(abstraction, action)
        if len(self.memory) > self.params.BATCH_SIZE:
            self.replay_DQNVP()
        return reward, done

    def sample_states(self, state, abs_vars):
        vars = [v for v in list(range(self.params.observation_space))]
        ranges = self.get_ranges()
        states = []
        for n in range(self.num_samples):
            add_state = []
            for iv, var in enumerate(vars):
                if iv in abs_vars:
                    add_state.append(state[0][iv])
                else:
                    add_state.append(random.random() * (ranges[iv][1] - ranges[iv][0]) + ranges[iv][0])
            states.append(np.reshape(add_state, [1, self.params.observation_space]))
        return states

    def get_ranges(self):
        min_max = [[float("inf"), float("-inf")] for v in list(range(self.params.observation_space))]
        for s, _, _, _, _ in self.memory:
            for iv, var in enumerate(s[0]):
                if var < min_max[iv][0]:
                    min_max[iv][0] = var
                if var > min_max[iv][1]:
                    min_max[iv][1] = var
        return min_max

    def pseudo_count(self, state):
        count = [a for a in list(range(self.action_space))]
        for s, a, _, _, _ in self.memory:
            norm = 0.
            for p, sd in enumerate(state[0]):
                norm += abs(sd - s[0][p])
            count[a] += max(0., 1. - (norm / self.d0))
        return count
