import gym
from gym.envs.toy_text.taxifuel import TaxiFuelEnv
import numpy as np
from matplotlib import pyplot as plt
from Agents.Q import QAgent
from Agents.QLiA import QLiAAgent
from Agents.QIiB import QIiBAgent


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


if __name__ == "__main__":
    env = TaxiFuelEnv()  # gym.make('Taxi-v3')
    size_state_vars = [5, 5, 5, 4, 14]  # [5, 5, 5, 4]
    abstractions = [(0, 1, 2, 4), (0, 1, 2, 3, 4)]
    init_alpha = 0.5
    alpha_min = 0.1
    init_epsilon = 0.3
    epsilon_min = 0.001
    discount = 0.95
    num_episodes = 2500

    agents = []
    agents.append(QAgent(env, init_alpha, alpha_min, init_epsilon, epsilon_min, discount))
    agents.append(QLiAAgent(env, init_alpha, alpha_min, init_epsilon, epsilon_min, discount, size_state_vars, abstractions, phi=0.5, phi_min=0.0001))
    agents.append(QIiBAgent(env, init_alpha, alpha_min, init_epsilon, epsilon_min, discount, size_state_vars, abstractions, phi=0.5, phi_min=0.0001))

    episode_rewards = [[] for q in range(len(agents))]
    starting_states = []

    print("-- Starting Experiment --")
    for j, agent in enumerate(agents):
        print("Running trial for agent:{0}".format(agent.name))
        for i in range(num_episodes):
            agent.reset()
            if j == 0:
                state = agent.current_state
                starting_states.append(state)
            else:
                state = starting_states[i]
                agent.set_state(state)

            done = False
            ep_reward = 0

            while not done:
                reward, done = agent.run_episode()
                ep_reward += reward

            episode_rewards[j].append(ep_reward)
            agent.decay(0.999)

    episode_rewards = np.array(episode_rewards)

    for er in episode_rewards:
        plt.plot(moving_average(er, int(num_episodes/100)))

    plt.legend([a.name for a in agents], loc='lower right')
    plt.show()


