import random
import numpy as np
import copy
from agents.DQN import DQNAgent, DQNMiniAgent
from helpers import functions


class DQNLiAAgent(DQNAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'DQNLiA_alt'
        self.sub_spaces = copy.copy(params.sub_spaces)
        self.action_space = self.params.action_space
        self.model = params.INIT_MODEL
        self.target_model = copy.copy(self.params.INIT_MODEL)

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def state_into_inputs(self, state):
        input_state = []
        ground_state = np.array(state)
        for ss in self.sub_spaces:
            input_state.append(np.reshape(ground_state[0][ss], [1, len(ss)]))
        return input_state

    def do_step(self):
        ground_state = np.reshape(self.last_n_states, [1, self.params.observation_space])
        input_state = self.state_into_inputs(ground_state)
        action = self.e_greedy_action([*input_state])
        next_state, reward, done = self.step(action)
        self.step_count += 1
        self.lia_replay()
        if done:
            self.decay(self.params.DECAY_RATE)
        return reward, done

    def lia_replay(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, self.params.BATCH_SIZE)
        values = []
        states = [[] for ss in range(len(self.sub_spaces))]
        for ground_state, action, reward, ground_next_state, done in minibatch:
            input_state = self.state_into_inputs(ground_state)
            state = [*input_state]
            input_next_state = self.state_into_inputs(ground_next_state)
            next_state = [*input_next_state]
            q_update = reward
            if not done:
                q_update = (reward + self.params.DISCOUNT *
                            np.amax(self.target_model.predict([*next_state])[0]))
            q_values = self.model.predict([*state])
            q_values[0][action] = q_update
            # Filtering out states and targets for training
            for isx, ss in enumerate(self.sub_spaces):
                states[isx].append(state[isx][0])
            values.append(q_values[0])

        self.until_retrain += 1
        if self.until_retrain >= self.retrain_steps:
            self.until_retrain = 0
            self.target_model.set_weights(self.model.get_weights())

        self.model.fit(x=[*states], y=np.array(values), verbose=0)

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
