from agents.base import BaseAgent
import numpy as np
from collections import deque
import random, copy
from helpers import functions

class DQNAgent(BaseAgent):
    def __init__(self, env, params):
        self.name = 'DQN'
        self.env = env
        self.action_space = params.action_space
        self.params = params
        self.model = params.INIT_MODEL
        self.target_model = copy.copy(self.model)
        self.current_state = None
        self.last_n_states = []
        self.memory = deque(maxlen=params.MEMORY_SIZE)

        self.until_retrain = 0
        self.retrain_steps = params.retrain_steps

        self.last_action = None
        self.step_count = 0

        self.temperature = 0.5

    def step(self, action):
        if 'AtariARIWrapper' in str(self.env):
            raw_state, reward, done, next_state_info = self.env.step(action)
            next_state = self.info_into_state(next_state_info, None)
        elif 'PLE' in str(self.env):
            act = self.env.getActionSet()[action]
            reward = self.env.act(act)
            next_state_obs = self.env.getGameState()
            next_state = self.pygame_obs_into_state(next_state_obs, None)
            done = self.env.game_over()
        else:
            next_state, reward, done, next_state_info = self.env.step(action)

        temp = copy.copy(self.last_n_states[int(self.params.observation_space/self.params.REPEAT_N_FRAMES):])
        temp = np.append(temp, next_state)
        next_last_n_states = temp

        self.remember(np.reshape(self.last_n_states, [1, self.params.observation_space]), action, reward,
                          np.reshape(next_last_n_states, [1, self.params.observation_space]), done)

        self.current_state = next_state
        self.last_n_states = next_last_n_states
        return next_state, reward, done

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def greedy_action(self, state):
        if len(state) == 1:
            q_values = self.model.predict(state)
        else:
            q_values = self.model.predict([*state])
        return np.argmax(q_values[0])

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()
        return self.greedy_action(state)

    def boltzmann_action(self, state):
        q_values = self.model.predict(state)
        softmaxed = functions.softmax(q_values/self.temperature)
        action_value = np.random.choice(softmaxed[0], p=softmaxed[0])
        return np.argmax(softmaxed[0] == action_value)

    def replay(self):
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

    def do_step(self):
        state = np.reshape(self.last_n_states, [1, self.params.observation_space])
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        self.step_count += 1
        self.replay()
        if done:
            self.decay(self.params.DECAY_RATE)
        return reward, done


class DQNMiniAgent(DQNAgent):
    def __init__(self, env, params):
        super().__init__(env, params)
        self.name = 'DQNMini'
        self.env = env
        self.params = copy.copy(params)
        self.action_space = self.params.action_space
        self.model = self.params.INIT_MODEL
        self.target_model = copy.copy(self.model)
        self.current_state = self.reset()
        self.memory = deque(maxlen=self.params.MEMORY_SIZE)

        self.until_retrain = 0
        self.retrain_steps = params.retrain_steps




