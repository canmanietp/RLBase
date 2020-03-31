from agents.base import BaseAgent
import helpers.unnest
import numpy as np
import copy


class MaxQAgent(BaseAgent):
    def __init__(self, env, params):
        super().__init__(env)
        self.name = 'MaxQ'
        self.params = copy.copy(params)
        self.options = self.params.options
        num_options = len(self.options)
        self.root = num_options - 1
        self.V = np.zeros((num_options, self.observation_space))
        self.C = np.zeros((num_options, self.observation_space, num_options))
        self.C_2 = np.zeros((num_options, self.observation_space, num_options))

        self.done = False
        self.r_sum = 0
        self.new_s = self.current_state

    def reset(self):
        if 'AtariARIWrapper' in str(self.env):
            self.env.reset()
            _, _, _, info = self.env.step(self.random_action())
            self.current_state = self.info_into_state(info, None)
        elif 'PLE' in str(self.env):
            self.env.init()
            obs = self.env.getGameState()
            self.current_state = self.pygame_obs_into_state(obs, None)
        else:
            self.current_state = self.env.reset()

        self.done = False
        self.r_sum = 0
        self.new_s = self.current_state
        return self.current_state

    def is_primitive(self, action):
        return action < self.action_space

    def is_terminal(self, a, done, state=None):
        if done or self.is_primitive(a):
            return True
        if a == self.root:
            return done

        if state is None:
            state = self.current_state

        decoded_state = list(self.env.decode(state))

        if 'TaxiEnv' in str(self.env):
            if a == 9:
                return decoded_state[2] < 4
            elif a == 8:
                return decoded_state[2] >= 4
            elif a == 7:
                return decoded_state[2] >= 4 and (decoded_state[0], decoded_state[1]) == self.env.locs[decoded_state[3]]
            elif a == 6:
                return decoded_state[2] < 4 and (decoded_state[0], decoded_state[1]) == self.env.locs[decoded_state[2]]
        elif 'TaxiFuelEnv' in str(self.env):
            if a == 12:
                return decoded_state[4] >= 13
            elif a == 11:
                return decoded_state[2] < 4
            elif a == 10:
                return decoded_state[2] >= 4
            elif a == 9:
                return (decoded_state[0], decoded_state[1]) == (3, 2)
            elif a == 8:
                return decoded_state[2] >= 4 and (decoded_state[0], decoded_state[1]) == self.env.locs[decoded_state[3]]
            elif a == 7:
                return decoded_state[2] < 4 and (decoded_state[0], decoded_state[1]) == self.env.locs[decoded_state[2]]
        else:
            print("Error: Must write MaxQ termination function for new environment")
            quit()

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def e_greedy_action(self, state, action):
        Q = np.arange(0)
        possible_a = np.arange(0)
        for act2 in self.options[action]:
            if self.is_primitive(act2) or (not self.is_terminal(act2, self.done)):
                Q = np.concatenate((Q, [self.V[act2, state] + self.C_2[action, state, act2]]))
                possible_a = np.concatenate((possible_a, [act2]))
        max_arg = np.argmax(Q)
        if np.random.rand(1) < self.params.EPSILON:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]

    def pseudo_reward(self, state, i):
        if 'TaxiFuelEnv' in str(self.env):
            decoded = list(self.env.decode(state))
            if decoded[4] == 0:
                return (i == self.root) * -1
        return self.is_terminal(i, self.done, state)

    def MAXQQ(self, i, s):  # i is action number
        if self.done:
            i = self.root + 1  # to end recursion
        self.done = False
        seq = []
        if self.is_primitive(i):
            self.new_s, r, self.done, _ = self.env.step(i)
            self.sa_visits[self.current_state][i] += 1
            self.r_sum += r
            self.V[i, s] += self.params.ALPHA * (r - self.V[i, s])
            return [s]
        elif i <= self.root:
            while not self.is_terminal(i, self.done):  # a is new action num
                a = self.e_greedy_action(s, i)
                child_seq = self.MAXQQ(a, s)
                child_seq = list(helpers.unnest.flatten(child_seq))
                q2 = []
                for k in self.options[i]:
                    q2.append((self.V[k, self.new_s] + self.C_2[i][self.new_s][k]))
                poss = list(self.options[i])
                a_star = poss[int(np.argmax(q2))]
                for N, sq in enumerate(child_seq):
                    r = (self.params.DISCOUNT ** (N+1)) * (self.C[i][self.new_s][a_star] + self.V[a_star, self.new_s])
                    r2 = (self.params.DISCOUNT ** (N+1)) * (self.pseudo_reward(self.new_s, i) + self.C_2[i][self.new_s][a_star] + self.V[a_star, sq])
                    self.C_2[i, sq, a] = (1 - self.params.ALPHA) * self.C_2[i, sq, a] + self.params.ALPHA * r2
                    self.C[i, sq, a] = (1 - self.params.ALPHA) * self.C[i, sq, a] + self.params.ALPHA * r
                seq.insert(0, child_seq)
                s = self.new_s
                self.current_state = s
        return seq

    def do_episode(self):
        self.MAXQQ(self.root, self.current_state)
        return self.r_sum

