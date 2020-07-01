import random
import numpy as np
import copy
from agents.DQN import DQNAgent, DQNMiniAgent
from helpers import functions


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

        self.last_abstraction = None

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

        self.remember(np.reshape(self.last_n_states, [1, self.params.observation_space]), abstraction, reward,
                          np.reshape(next_last_n_states, [1, self.params.observation_space]), done)

        # for sax, sa in enumerate(self.sub_agents):
        #     abs_last_n_states = self.last_n_states[self.params.sub_spaces[sax]]
        #     abs_next_last_n_states = next_last_n_states[self.params.sub_spaces[sax]]
        #     sa.remember(np.reshape(abs_last_n_states, [1, sa.params.observation_space]), action, reward, np.reshape(abs_next_last_n_states, [1, sa.params.observation_space]), done)
        self.current_state = next_state
        self.last_n_states = next_last_n_states
        return next_state, reward, done

    def e_greedy_LiA_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            ab_index = self.random_action()
            action = self.sub_agents[ab_index].random_action()
        else:
            ab_index = self.greedy_action(state)
            abs_state = list(state[0][self.params.sub_spaces[ab_index]])
            abs_state = np.reshape(abs_state, [1, len(self.params.sub_spaces[ab_index])])
            action = self.sub_agents[ab_index].greedy_action(abs_state)
        return ab_index, action

    def boltzmann_LiA_action(self, state):
        q_values = self.model.predict(state)
        softmaxed = functions.softmax(q_values/self.temperature)
        ab_value = np.random.choice(softmaxed[0], p=softmaxed[0])
        ab_index = int(np.argmax(softmaxed[0] == ab_value))
        abs_state = list(state[0][self.params.sub_spaces[ab_index]])
        abs_state = np.reshape(abs_state, [1, len(self.params.sub_spaces[ab_index])])
        q_values = self.sub_agents[ab_index].model.predict(abs_state)
        softmaxed = functions.softmax(q_values/self.sub_agents[ab_index].temperature)
        action_value = np.random.choice(softmaxed[0], p=softmaxed[0])
        action = np.argmax(softmaxed[0] == action_value)
        return ab_index, action

    def do_step(self):
        state = np.reshape(self.last_n_states, [1, self.params.observation_space])
        abstraction, action = self.e_greedy_LiA_action(state)
        next_state, reward, done = self.step_LiA(abstraction, action)
        self.step_count += 1
        self.lia_replay()
        if done:
            self.decay(self.params.DECAY_RATE)
            for ab in self.sub_agents:
                ab.decay(self.params.DECAY_RATE)
        return reward, done

    def lia_replay(self):
        if len(self.memory) < self.params.BATCH_SIZE:
            return
        minibatch = random.sample(self.memory, self.params.BATCH_SIZE)

        states, values = [], []
        for state, action, reward, next_state, done in minibatch:
            q_update = reward
            if not done:
                q_update = (reward + self.params.DISCOUNT *
                            np.amax(self.target_model.predict(next_state)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            # Filtering out states and targets for training
            states.append(state[0])
            values.append(q_values[0])

        self.until_retrain += 1
        if self.until_retrain >= self.retrain_steps:
            self.until_retrain = 0
            self.target_model.set_weights(self.model.get_weights())

        self.model.fit(np.array(states), np.array(values), verbose=0)

        for ax, ab in enumerate(self.sub_agents):
            states, values = [], []
            for state, action, reward, next_state, done in minibatch:
                state = state[0][ab.params.sub_spaces[ax]]
                state = np.reshape(state, [1, ab.params.observation_space])
                next_state = next_state[0][ab.params.sub_spaces[ax]]
                next_state = np.reshape(next_state, [1, ab.params.observation_space])
                q_update = reward
                if not done:
                    q_update = (reward + ab.params.DISCOUNT *
                                np.amax(ab.target_model.predict(next_state)[0]))
                q_values = ab.model.predict(state)
                q_values[0][action] = q_update
                # Filtering out states and targets for training
                states.append(state[0])
                values.append(q_values[0])

            ab.until_retrain += 1
            if ab.until_retrain >= ab.retrain_steps:
                ab.until_retrain = 0
                ab.target_model.set_weights(ab.model.get_weights())

            ab.model.fit(np.array(states), np.array(values), verbose=0)

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
