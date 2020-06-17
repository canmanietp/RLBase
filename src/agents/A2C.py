from agents.base import BaseAgent
import numpy as np
from collections import deque
import random, copy
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95


def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1][0][0] * masks[i] - values[i][0][0]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


def ppo_loss(oldpolicy_probs, advantages, rewards, values):
    def loss(y_true, y_pred):
        newpolicy_probs = y_pred
        ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
        actor_loss = -K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(rewards - values))
        total_loss = critic_discount * critic_loss + actor_loss - entropy_beta * K.mean(
            -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
        return total_loss

    return loss

def one_hot_encoding(probs):
    one_hot = np.zeros_like(probs)
    one_hot[:, np.argmax(probs, axis=1)] = 1
    return one_hot


class A2CAgent(BaseAgent):
    def __init__(self, env, params):
        self.name = 'A2C'
        self.env = env
        self.action_space = params.action_space
        self.params = params
        self.input_shape = np.array((params.REPEAT_N_FRAMES, int(params.observation_space / params.REPEAT_N_FRAMES)))
        self.model_actor = self.get_model_actor(self.input_shape, self.action_space)
        self.model_critic = self.get_model_critic(self.input_shape)
        self.current_state = None
        self.last_n_states = []

        self.dummy_n = np.zeros((1, 1, self.action_space))
        self.dummy_1 = np.zeros((1, 1, 1))

        self.states = []
        self.actions = []
        self.actions_onehot = []
        self.actions_probs = []
        self.values = []
        self.masks = []
        self.rewards = []
        self.next_states = []
        self.memory = deque(maxlen=self.params.MEMORY_SIZE)

        self.step_count = 0

    def get_model_actor(self, input_dims, output_dims):
        state_input = Input(shape=input_dims)
        oldpolicy_probs = Input(shape=(1, output_dims,))
        advantages = Input(shape=(1, 1,))
        rewards = Input(shape=(1, 1,))
        values = Input(shape=(1, 1,))

        # Classification block
        x = Dense(512, activation='relu', name='fc1')(state_input)
        x = Dense(256, activation='relu', name='fc2')(x)
        out_actions = Dense(self.action_space, activation='softmax', name='predictions')(x)

        model = Model(inputs=[state_input, oldpolicy_probs, advantages, rewards, values],
                      outputs=[out_actions])
        model.compile(optimizer=Adam(lr=1e-4), loss=[ppo_loss(
            oldpolicy_probs=oldpolicy_probs,
            advantages=advantages,
            rewards=rewards,
            values=values)])
        # model.summary()
        return model

    def get_model_critic(self, input_dims):
        state_input = Input(shape=input_dims)

        # Classification block
        x = Dense(512, activation='relu', name='fc1')(state_input)
        x = Dense(256, activation='relu', name='fc2')(x)
        out_actions = Dense(1, activation='tanh')(x)

        model = Model(inputs=[state_input], outputs=[out_actions])
        model.compile(optimizer=Adam(lr=1e-4), loss='mse')
        # model.summary()
        return model

    def remember(self, state, action, reward, done):
        print('remember', state.shape)
        action_dist = self.model_actor.predict([state, self.dummy_n, self.dummy_1, self.dummy_1, self.dummy_1], steps=1)
        action_onehot = np.zeros(self.action_space)
        action_onehot[action] = 1
        self.states.append(state)
        self.actions.append(action)
        self.actions_onehot.append(action_onehot)
        q_value = self.model_critic.predict([state], steps=1)
        self.values.append(q_value)
        self.masks.append(not done)
        self.rewards.append(reward)
        self.actions_probs.append(action_dist)

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

        state = np.reshape(self.last_n_states, self.input_shape)
        state_input = np.expand_dims(state, axis=0)
        print('step', state_input.shape)

        temp = copy.copy(self.last_n_states[int(self.params.observation_space / self.params.REPEAT_N_FRAMES):])
        temp = np.append(temp, next_state)
        next_last_n_states = temp

        self.current_state = next_state
        self.last_n_states = next_last_n_states

        self.remember(state_input, action, reward, done)
        return next_state, reward, done

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        print('act', state.shape)
        action_dist = self.model_actor.predict([state, self.dummy_n, self.dummy_1, self.dummy_1, self.dummy_1], steps=1)
        action = np.random.choice(self.action_space, p=action_dist[0][0])
        return action

    def decay(self, decay_rate):
        if self.params.EPSILON > self.params.EPSILON_MIN:
            self.params.EPSILON *= decay_rate

    def replay(self):
        print("replay", self.values, self.masks, self.rewards)
        returns, advantages = get_advantages(self.values, self.masks, self.rewards)
        actor_loss = self.model_actor.fit(
            [self.states, self.actions_probs, advantages, np.reshape(self.rewards, newshape=(-1, 1, 1)), self.values[:-1]],
            [(np.reshape(self.actions_onehot, newshape=(-1, self.action_space)))], verbose=False, shuffle=True, epochs=8)
        critic_loss = self.model_critic.fit([self.states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                       verbose=False)

        print(actor_loss, critic_loss)

    def do_step(self):
        state = np.reshape(self.last_n_states, self.input_shape)
        state = np.expand_dims(state, axis=0)
        print("do step", state)
        action = self.act(state)
        next_state, reward, done = self.step(action)

        self.step_count += 1

        if self.step_count % 25 == 0:
            # add last value
            q_value = self.model_critic.predict(state, steps=1)
            self.values.append(q_value)
            self.replay()

        if done:
            self.decay(self.params.DECAY_RATE)

        return reward, done
