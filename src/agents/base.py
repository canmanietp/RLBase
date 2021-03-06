import numpy as np
import helpers.unnest


class BaseAgent:
    def __init__(self, env):
        self.name = 'Base'
        self.env = env
        self.observation_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.sa_visits = np.zeros([self.observation_space, self.action_space])
        self.inadmissible_actions = [[] for os in range(self.observation_space)]
        self.current_state = self.reset()

    def reset(self):
        if 'AtariARIWrapper' in str(self.env):
            self.env.reset()
            _, _, _, info = self.env.step(self.random_action())
            self.current_state = self.info_into_state(info, None)
            self.last_n_states = np.repeat((self.current_state), 4)
            # self.last_two_states = np.concatenate((self.current_state, self.current_state))
        elif 'PLE' in str(self.env):
            self.env.init()
            obs = self.env.getGameState()
            self.current_state = self.pygame_obs_into_state(obs, None)
        else:
            self.current_state = self.env.reset()
        return self.current_state

    def random_action(self, state=None):
        if state is not None and self.inadmissible_actions[state] != []:
            admissible_actions = [x for x in range(self.action_space) if x not in self.inadmissible_actions[state]]
        else:
            admissible_actions = None
        if admissible_actions is not None:
            return np.random.choice(admissible_actions)
        return np.random.randint(self.action_space)

    def get_state(self):
        return self.current_state

    def set_state(self, state):
        if hasattr(self.env, 'env'):
            self.env.env.s = state
        else:
            self.env.s = state
        self.current_state = state

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.sa_visits[self.current_state][action] += 1
        self.current_state = next_state

        return next_state, reward, done

    def info_into_state(self, info, abstraction):
        state = []
        if abstraction is None:
            for i, lab in enumerate(info['labels']):
                state.append(info['labels'][lab])
        else:
            for i, lab in enumerate(info['labels']):
                if i in abstraction:
                    state.append(info['labels'][lab])
        return np.array(state)

    def pygame_obs_into_state(self, obs, abstraction):
        state = list(helpers.unnest.flatten(obs))
        if abstraction is None:
            return state
        else:
            return list(np.array(state)[abstraction])

    def do_step(self):
        action = self.random_action
        next_state, reward, done = self.step(action)
        return reward, done

    def do_episode(self):
        done = False
        ep_reward = 0
        while not done:
            reward, done = self.do_step()
            ep_reward += reward
        return ep_reward

