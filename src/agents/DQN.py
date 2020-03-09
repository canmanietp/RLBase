from agents.base import BaseAgent
import numpy as np
from collections import deque
import random


class DQNAgent(BaseAgent):
    def __init__(self, env, params):
        self.name = 'DQN'
        self.env = env
        self.action_space = env.action_space.n
        self.params = params
        self.model = params.INIT_MODEL
        self.b_model = self.model
        self.current_state = self.reset()
        self.memory = deque(maxlen=params.MEMORY_SIZE)

        self.until_retrain = 0
        self.retrain_steps = 4

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
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
        minibatch = random.sample(self.memory, self.params.BATCH_SIZE)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.params.DISCOUNT *
                          np.amax(self.b_model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        self.until_retrain += 1
        if self.until_retrain >= self.retrain_steps:
            self.until_retrain = 0
            self.b_model.set_weights(self.model.get_weights())
        self.decay(self.params.DECAY_RATE)

    def run_episode(self):
        state = np.reshape(self.current_state, [1, self.params.observation_space])
        action = self.e_greedy_action(state)
        next_state, reward, done = self.step(action)
        if len(self.memory) > self.params.BATCH_SIZE:
            self.replay()
        return reward, done



