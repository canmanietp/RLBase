from agents.Q import QAgent
from agents.Q import QMiniAgent
from learning_parameters import DiscreteParameters
import numpy as np
import copy


class QLiA_TAgent(QAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'LiA_T'
        self.sub_agents = []
        self.params = params
        self.fixed_discount = params.DISCOUNT

        for ab in params.sub_spaces:
            ss = 1
            for var in ab:
                ss *= params.size_state_vars[var]

            ab_params = copy.copy(self.params)
            ab_params.EPSILON = params.PHI
            ab_params.EPSILON_MIN = params.PHI_MIN
            self.sub_agents.append(QMiniAgent(self.env, ab_params, ss, self.env.action_space.n))

        self.action_space = len(params.sub_spaces) + 1  # Terminate
        self.Q_table = np.zeros([self.observation_space, self.action_space])

        self.active_sub_agent = None
        self.active_reward = [0 for s in self.sub_agents]
        self.active_steps = [0 for s in self.sub_agents]
        self.inactive_reward = [0 for s in self.sub_agents]
        self.entry_state = None

        self.state_decodings = self.sweep_state_decodings()

    def sweep_state_decodings(self):
        st_vars_lookup = []
        for s in range(self.observation_space):
            st_vars_lookup.append(list(self.env.decode(s)))
        return st_vars_lookup

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

    def decay(self, decay_rate):
        if self.params.ALPHA > self.params.ALPHA_MIN:
            self.params.ALPHA *= decay_rate
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

        for ab in self.sub_agents:
            ab.decay(decay_rate)

    def update_LIAT(self, state, action, reward, next_state, done):
        state_vars = self.state_decodings[state]
        next_state_vars = self.state_decodings[next_state]

        for ia, ab in enumerate(self.sub_agents):
            abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ia])
            abs_next_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[ia])
            ab.update(abs_state, action, reward, abs_next_state, done)

        self.update(state, self.active_sub_agent, reward, next_state, done)

    def do_step(self):
        terminate_or_active = self.e_greedy_action(self.current_state)
        terminate = terminate_or_active == len(self.sub_agents)

        if not terminate:
            self.active_sub_agent = terminate_or_active
            state_vars = self.state_decodings[self.current_state]
            abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[self.active_sub_agent])
            action = self.sub_agents[self.active_sub_agent].e_greedy_action(abs_state)

            next_state, reward, done = self.step(action)
            self.active_reward[self.active_sub_agent] += reward
            self.active_steps[self.active_sub_agent] += 1
            next_state_vars = self.state_decodings[next_state]

            for ix, x in enumerate(self.sub_agents):
                abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ix])
                abs_next_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[ix])
                x.update(abs_state, action, reward, abs_next_state, done)

                if ix != self.active_sub_agent:
                    self.inactive_reward[ix] += reward

                if done:
                    self.inactive_reward[ix] = 0
                    # self.active_reward[ix] = 0
                    # self.active_steps[ix] = 0

            if done:
                if self.entry_state is not None:
                    self.params.DISCOUNT = self.fixed_discount**self.active_steps[self.active_sub_agent]
                    self.update(self.entry_state, self.active_sub_agent, self.active_reward[self.active_sub_agent],
                                self.current_state, True)
                self.entry_state = None
                self.active_reward[self.active_sub_agent] = 0
                self.active_steps[self.active_sub_agent] = 0
                self.inactive_reward[self.active_sub_agent] = 0
                self.active_sub_agent = None

            return reward, done
        else:
            if self.entry_state is not None and self.active_sub_agent is not None:
                self.params.DISCOUNT = self.fixed_discount ** self.active_steps[self.active_sub_agent]
                self.update(self.entry_state, self.active_sub_agent, self.active_reward[self.active_sub_agent], self.current_state, False)
                self.active_reward[self.active_sub_agent] = 0
                self.active_steps[self.active_sub_agent] = 0
                self.inactive_reward[self.active_sub_agent] = 0
            self.entry_state = self.current_state
            return 0, False

    # def do_step(self):
    #     state_vars = self.state_decodings[self.current_state]
    #
    #     if self.active_sub_agent is None:
    #         self.active_sub_agent = self.e_greedy_action(self.current_state)
    #         self.entry_state = self.current_state
    #     abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[self.active_sub_agent])
    #     action = self.sub_agents[self.active_sub_agent].e_greedy_action(abs_state)
    #     if action == self.sub_agents[0].action_space - 1:
    #         if self.entry_state is not None:
    #             self.update(self.entry_state, self.active_sub_agent, self.active_reward[self.active_sub_agent], self.current_state, False)
    #         self.termination_state[self.active_sub_agent] = abs_state
    #         self.active_reward[self.active_sub_agent] = 0
    #         self.inactive_reward[self.active_sub_agent] = 0
    #         self.active_sub_agent = self.e_greedy_action(self.current_state)
    #         self.entry_state = self.current_state
    #         if self.termination_state[self.active_sub_agent] is not None:
    #             new_abs_state = self.encode_abs_state(state_vars,
    #                                               self.params.sub_spaces[self.active_sub_agent])
    #             self.sub_agents[self.active_sub_agent].update(self.termination_state[self.active_sub_agent],
    #                                                           action,
    #                                                           self.inactive_reward[self.active_sub_agent],
    #                                                           new_abs_state,
    #                                                           False)
    #     else:
    #         next_state, reward, done = self.step(action)
    #         self.active_reward[self.active_sub_agent] += reward
    #         next_state_vars = self.state_decodings[next_state]
    #
    #         for ix, x in enumerate(self.sub_agents):
    #             abs_state = self.encode_abs_state(state_vars, self.params.sub_spaces[ix])
    #             abs_next_state = self.encode_abs_state(next_state_vars, self.params.sub_spaces[ix])
    #             x.update(abs_state, action, reward, abs_next_state, done)
    #
    #             if ix != self.active_sub_agent:
    #                 self.inactive_reward[ix] += reward
    #
    #             if done:
    #                 if self.termination_state[ix] is not None:
    #                     x.update(self.termination_state[ix], action, self.inactive_reward[ix], abs_state, True)
    #
    #         if done:
    #             if self.entry_state is not None:
    #                 self.update(self.entry_state, self.active_sub_agent, self.active_reward[self.active_sub_agent],
    #                         self.current_state, True)
    #             self.entry_state = None
    #             self.inactive_reward[ix] = 0
    #             self.active_reward[ix] = 0
    #             self.termination_state[ix] = None
    #
    #         return reward, done
    #     return 0, False

