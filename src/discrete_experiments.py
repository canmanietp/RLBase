import numpy as np
from matplotlib import pyplot as plt
import pickle
import time, os, datetime

from envs.taxi import TaxiEnv
from envs.taxifuel import TaxiFuelEnv
from envs.rm_office import OfficeEnv
from envs.coffeemail import CoffeeMailEnv
from envs.coffee import CoffeeEnv

from agents.Q import QAgent
from agents.QLiA import QLiAAgent
from agents.QIiB import QIiBAgent
from learning_parameters import DiscreteParameters
from helpers import plotting


def get_params_coffee():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.99
    sub_spaces = [[0, 1, 2, 3], [0, 1, 2, 3, 4]]
    size_state_vars = [5, 5, 2, 2, 2]
    num_episodes = 500
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_coffeemail():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.999
    sub_spaces = [[0, 1, 2, 4], [0, 1, 2, 3, 4, 5, 6, 7]]
    size_state_vars = [5, 5, 2, 2, 2, 2, 2, 2]
    num_episodes = 10000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_office():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.999
    sub_spaces = [[0, 1, 2, 4], [0, 1, 2, 3, 4, 5]]
    size_state_vars = [9, 12, 2, 2, 2, 2]
    num_episodes = 30000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_taxifuel():
    init_alpha = 0.5
    alpha_min = 0.1
    init_epsilon = 0.5
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.999
    sub_spaces = [[0, 1, 2, 4], [0, 1, 2, 3], [0, 1, 2, 3, 4]]
    size_state_vars = [5, 5, 5, 4, 14]
    num_episodes = 200000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_taxi():
    init_alpha = 0.5
    alpha_min = 0.05
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.3
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.99
    sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    size_state_vars = [5, 5, 5, 4]
    num_episodes = 1000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params(env_name):
    if env_name == 'taxi':
        env = TaxiEnv()
        params = get_params_taxi()
    elif env_name == 'taxifuel':
        env = TaxiFuelEnv()
        params = get_params_taxifuel()
    elif env_name == 'office':
        env = OfficeEnv()
        params = get_params_office()
    elif env_name == 'coffeemail':
        env = CoffeeMailEnv()
        params = get_params_coffeemail()
    elif env_name == 'coffee':
        env = CoffeeEnv()
        params = get_params_coffee()
    else:
        print("Error: Unknown environment")
        return
    return env, params


def run_discrete_experiment(num_trials, env_name, algs, verbose=False):
    date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_dir = "tmp/{}".format(date_string)
    os.mkdir(exp_dir)
    env, params = get_params(env_name)
    average_every = int(params.num_episodes / 100)

    trial_rewards = []
    trial_times = []

    for t in range(num_trials):
        agents = []
        for alg in algs:
            if alg == 'Q':
                agents.append(QAgent(env, params))
            elif alg == 'QLiA':
                agents.append(QLiAAgent(env, params))
            elif alg == 'QIiB':
                agents.append(QIiBAgent(env, params))
            else:
                print("Unknown algorithm - {}".format(alg))

        episode_rewards = [[] for q in range(len(agents))]
        starting_states = []
        times_to_run = []
        plt.figure()

        print("{} -- Starting Trial {} -- ".format(datetime.datetime.now().strftime("%H:%M:%S"), t + 1))
        for j, agent in enumerate(agents):
            t0 = time.time()
            print("{} Running agent: {}".format(datetime.datetime.now().strftime("%H:%M:%S"), agent.name))
            for i in range(params.num_episodes):
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
                if verbose:
                    print("{} Episode {}, reward={}".format(datetime.datetime.now().strftime("%H:%M:%S"), i, ep_reward))
                agent.decay(agent.params.DECAY_RATE)
            run_time = time.time() - t0
            print("{} Finished running in {} seconds".format(datetime.datetime.now().strftime("%H:%M:%S"), run_time))
            times_to_run.append(run_time)

            plt.plot(plotting.moving_average(episode_rewards[j], average_every), label=agent.name)
            plt.legend([a.name for a in agents], loc='lower right')
            plt.savefig('{}/trial_{}'.format(exp_dir, t + 1))

        plt.close()
        trial_rewards.append(episode_rewards)
        trial_times.append(times_to_run)

    for trial in np.average(trial_rewards, axis=0):
        plt.plot(plotting.moving_average(trial, average_every))
    plt.legend([a for a in algs], loc='lower right')
    plt.savefig('{}/final'.format(exp_dir))

    file1 = open('{}/params.txt'.format(exp_dir), "w")
    file1.write("Environment: {}\n"
                "Number of trials: {}\n"
                "Number of episodes: {}\n"
                "Algorithms: {}\n"
                "Running times: {}\n"
                "init_alpha={}\n"
                "alpha_min={}\n"
                "init_epsilon={}\n"
                "epsilon_min={}\n"
                "init_phi={}\n"
                "phi_min={}\n"
                "discount={}\n"
                "sub_spaces={}\n"
                "size_state_vars={}".format(env, num_trials,
                                            params.num_episodes, algs, trial_times, params.ALPHA, params.ALPHA_MIN,
                                            params.EPSILON, params.EPSILON_MIN,
                                            params.PHI, params.PHI_MIN, params.DISCOUNT,
                                            params.sub_spaces, params.size_state_vars))
    file1.close()
    trial_rewards = np.array(trial_rewards)
    pickle.dump(trial_rewards, open('{}/save.p'.format(exp_dir), "wb"))
    return
