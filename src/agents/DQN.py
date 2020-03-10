from agents.base import BaseAgent
import numpy as np
from collections import deque
import random, copy


class DQNAgent(BaseAgent):
    def __init__(self, env, params):
        self.name = 'DQN'
        self.env = env
        self.action_space = env.action_space.n
        self.params = copy.copy(params)
        self.model = params.INIT_MODEL
        self.target_model = copy.copy(self.model)
        self.current_state = self.reset()
        self.memory = deque(maxlen=params.MEMORY_SIZE)

        self.until_retrain = 0
        self.retrain_steps = params.retrain_steps

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        print(done)
        self.remember(np.reshape(self.current_state, [1, self.params.observation_space]), action, reward, np.reshape(next_state, [1, self.params.observation_space]), done)
        self.current_state = next_state
        return next_state, reward, done

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def e_greedy_action(self, state):
        if random.uniform(0, 1) < self.params.EPSILON:
            return self.random_action()
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

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
        self.model.fit(np.array(states), np.array(values), verbose=0)

        self.until_retrain += 1
        if self.until_retrain >= self.retrain_steps:
            self.until_retrain = 0
            self.target_model.set_weights(self.model.get_weights())

        self.decay(self.params.DECAY_RATE)

    def run_episode(self):
        state = np.reshape(self.current_state, [1, self.params.observation_space])
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        self.replay()
        return reward, done



