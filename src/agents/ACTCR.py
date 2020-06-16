from agents.base import BaseAgent
import numpy as np
from collections import deque
import random, copy
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop


def OurModel(input_shape, action_space, lr):
    X_input = Input(input_shape)
    X = Flatten(input_shape=input_shape)(X_input)

    X = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    X = Dense(256, activation="elu", kernel_initializer='he_uniform')(X)
    X = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X)
    value = Dense(1, kernel_initializer='he_uniform')(X)

    Actor = Model(inputs=X_input, outputs=action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=lr))

    Critic = Model(inputs=X_input, outputs=value)
    Critic.compile(loss='mse', optimizer=RMSprop(lr=lr))

    return Actor, Critic


class ACTCRAgent(BaseAgent):
    def __init__(self, env, params):
        self.name = 'DQN'
        self.env = env
        self.action_space = params.action_space
        self.params = params
        self.input_shape = np.array((params.REPEAT_N_FRAMES, int(params.observation_space / params.REPEAT_N_FRAMES)))
        self.Actor, self.Critic = OurModel(input_shape=self.input_shape, action_space=self.action_space, lr=self.params.LEARNING_RATE)
        self.current_state = None
        self.last_n_states = []
        self.memory = deque(maxlen=params.MEMORY_SIZE)

        self.step_count = 0

    def remember(self, state, action, reward):
        action_onehot = np.zeros([self.action_space])
        action_onehot[action] = 1
        self.memory.append((state, action_onehot, reward))

    def step(self, action):
        if 'AtariARIWrapper' in str(self.env):
            next_state, reward, done, next_state_info = self.env.step(action)
            next_state = self.info_into_state(next_state_info, None)
        elif 'PLE' in str(self.env):
            act = self.env.getActionSet()[action]
            reward = self.env.act(act)
            next_state_obs = self.env.getGameState()
            next_state = self.pygame_obs_into_state(next_state_obs, None)
            done = self.env.game_over()
        else:
            next_state, reward, done, next_state_info = self.env.step(action)

        state = np.reshape(self.last_n_states, self.input_shape)
        state = np.expand_dims(state, axis=0)

        self.remember(state, action, reward)

        temp = copy.copy(self.last_n_states[int(self.params.observation_space / self.params.REPEAT_N_FRAMES):])
        temp = np.append(temp, next_state)
        next_last_n_states = temp

        self.current_state = next_state
        self.last_n_states = next_last_n_states

        return next_state, reward, done

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        prediction = self.Actor.predict(state)[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            # if reward[i] != 0:  # reset the sum, since this was a game boundary (pong specific!)
            #     running_add = 0
            running_add = running_add * self.params.DISCOUNT + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r)  # normalizing the result
        discounted_r /= np.std(discounted_r)  # divide by standard deviation
        return discounted_r

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def replay(self):
        # reshape memory to appropriate shape for training
        mem_states = []
        mem_actions = []
        mem_rewards = []

        for state, action, reward in self.memory:
            mem_states.append(state)
            mem_actions.append(action)
            mem_rewards.append(reward)

        states = np.vstack(mem_states)
        actions = np.vstack(mem_actions)

        # Compute discounted rewards
        discounted_r = self.discount_rewards(mem_rewards)

        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)

        self.memory = []

    def do_step(self):
        state = np.reshape(self.last_n_states, self.input_shape)
        state = np.expand_dims(state, axis=0)
        action = self.act(state)
        next_state, reward, done = self.step(action)
        self.step_count += 1
        if done:
            self.replay()
            self.decay(self.params.DECAY_RATE)
        return reward, done
