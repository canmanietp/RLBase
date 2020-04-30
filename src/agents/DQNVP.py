import numpy as np
import copy, random
from agents.DQN import DQNAgent, DQNMiniAgent


class DQNVPAgent(DQNAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'DQNVP'

        self.model = params.INIT_MODEL
        self.target_model = params.INIT_MODEL

        meta_params = copy.copy(params)
        meta_params.INIT_MODEL = meta_params.META_MODEL
        meta_params.EPSILON = meta_params.PHI
        meta_params.EPSILON_MIN = meta_params.PHI_MIN
        meta_params.action_space = len(meta_params.sub_spaces)
        self.meta_agent = DQNMiniAgent(self.env, meta_params)

        self.num_samples = 10
        self.d0 = 100  # For pseudo count, should be a parameter for QVP
        self.hist = None
        self.binedges = None

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
            # self.calculate_pseudo_counts()
        if done:
            self.decay(self.params.DECAY_RATE)
        return reward, done

    def sample_states(self, state, abs_vars):
        vars = [v for v in list(range(self.params.observation_space))]
        ranges = self.get_ranges()
        states = []
        for n in range(self.num_samples):
            add_state = []
            for iv, var in enumerate(vars):
                range_bin_width = abs(ranges[iv][0] - ranges[iv][1]) / self.num_samples
                if iv in abs_vars:
                    add_state.append(state[0][iv])
                else:
                    add_state.append(ranges[iv][0] + range_bin_width * n)
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

    def calculate_pseudo_counts(self):
        sa = [np.append(item[0][0], item[1]) for item in self.memory]
        sa_data_array = np.array([np.array(xi) for xi in sa])
        hist, binedges = np.histogramdd(sa_data_array)
        # states = [s[0] for s, a, _, _, _ in self.memory]
        # hist, binedges = np.histogramdd(np.array(states), normed=False)
        self.hist = np.array(hist)
        self.binedges = binedges

    def pseudo_count(self, state):
        # if self.hist is not None:
        #     pseudo_count = [[] for a in range(self.action_space)]
        #     for action in range(self.action_space):
        #         bins = []
        #         for i, d in enumerate(np.append(state[0], action)):
        #             for j in range(len(self.binedges[i])):
        #                 if j + 1 < len(self.binedges[i]):
        #                     if self.binedges[i][j] <= d < self.binedges[i][j + 1]:
        #                         bins.append(j)
        #                         continue
        #                     elif j == len(self.binedges[i]) - 2 and d == self.binedges[i][j + 1]:
        #                         bins.append(j)
        #                         continue
        #         if len(bins) == 3:
        #             # print(3)
        #             pseudo_count[action] = self.hist[bins[0], bins[1], bins[2]]
        #         elif len(bins) == 4:
        #             # print(4)
        #             pseudo_count[action] = self.hist[bins[0], bins[1], bins[2], bins[3]]
        #         elif len(bins) == 5:
        #             # print(5)
        #             pseudo_count[action] = self.hist[bins[0], bins[1], bins[2], bins[3], bins[4]]
        #     return pseudo_count
        # else:
        #     return np.zeros(self.action_space)
        count = [0. for a in list(range(self.action_space))]
        if len(self.memory) > 1000:
            minibatch = random.sample(self.memory, 1000)
            for s, a, _, _, _ in minibatch:
                norm = 0.
                for p, sd in enumerate(state[0]):
                    norm += abs(sd - s[0][p])
                count[a] += max(0., 1. - (norm / self.d0))
        return count
