import numpy as np
from matplotlib import pyplot as plt
import time, os, argparse, datetime, copy

from envs.taxi import TaxiEnv
from envs.taxifuel import TaxiFuelEnv
from envs.rm_office import OfficeEnv
from envs.coffeemail import CoffeeMailEnv
from envs.coffee import CoffeeEnv

from agents.Q import QAgent
from agents.QLiA import QLiAAgent
from agents.QIiB import QIiBAgent
from learning_parameters import Parameters


def get_params_coffee():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    subspaces = [(0, 1, 2, 3), (0, 1, 2, 3, 4)]
    size_state_vars = [5, 5, 2, 2, 2]
    num_episodes = 1000
    return Parameters(init_alpha, alpha_min, init_epsilon, epsilon_min, discount, num_episodes, init_phi, phi_min, subspaces, size_state_vars)


def get_params_coffeemail():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    subspaces = [(0, 1, 2, 4), (0, 1, 2, 3, 4, 5, 6, 7)]
    size_state_vars = [5, 5, 2, 2, 2, 2, 2, 2]
    num_episodes = 10000
    return Parameters(init_alpha, alpha_min, init_epsilon, epsilon_min, discount, num_episodes, init_phi, phi_min, subspaces, size_state_vars)


def get_params_office():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    subspaces = [(0, 1, 2, 4), (0, 1, 2, 3, 4, 5)]
    size_state_vars = [9, 12, 2, 2, 2, 2]
    num_episodes = 30000
    return Parameters(init_alpha, alpha_min, init_epsilon, epsilon_min, discount, num_episodes, init_phi, phi_min, subspaces, size_state_vars)


def get_params_taxifuel():
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    subspaces = [(0, 1, 2, 4), (0, 1, 2, 3, 4)]
    size_state_vars = [5, 5, 5, 4, 14]
    num_episodes = 200000
    return Parameters(init_alpha, alpha_min, init_epsilon, epsilon_min, discount, num_episodes, init_phi, phi_min, subspaces, size_state_vars)


def get_params_taxi():
    init_alpha = 0.5
    alpha_min = 0.05
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.3
    phi_min = 0.001
    discount = 0.99
    subspaces = [(0, 1, 2), (0, 1, 2, 3)]
    size_state_vars = [5, 5, 5, 4]
    num_episodes = 2000
    return Parameters(init_alpha, alpha_min, init_epsilon, epsilon_min, discount, num_episodes, init_phi, phi_min, subspaces, size_state_vars)


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


def run_experiment(num_trials, env_name, algs, verbose=False):
    date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_dir = "tmp/{}".format(date_string)
    os.mkdir(exp_dir)
    env, params = copy.copy(get_params(env_name))

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
        t0 = time.time()
        plt.figure()
        average_every = int(params.num_episodes / 100)

        print("-- Starting Trial {} -- ".format(t+1))
        for j, agent in enumerate(agents):
            print("Running agent: {0}".format(agent.name))
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
                    print("Episode {}, reward={}".format(i, ep_reward))
                agent.decay(0.99)
            t1 = time.time()
            print("Finished running in {} seconds".format(t1 - t0))
            t0 = t1

            ma = np.cumsum(episode_rewards[j], dtype=float)
            ma[average_every:] = ma[average_every:] - ma[:-average_every]
            ma = ma[average_every - 1:] / average_every
            plt.plot(ma, label=agent.name)
            plt.legend([a.name for a in agents], loc='lower right')
            plt.savefig('{}/trial_{}'.format(exp_dir, t+1))

        plt.close()

    file1 = open('{}/params.txt'.format(exp_dir), "w")
    file1.write("Environment: {}\n"
                "Number of trials: {}\n"
                "Number of episodes: {}\n"
                "init_alpha={}\n"
                "alpha_min={}\n"
                "init_epsilon={}\n"
                "epsilon_min={}\n"
                "init_phi={}\n"
                "phi_min{}\n"
                "discount={}\n"
                "subspaces={}\n"
                "size_state_vars={}".format(env, num_trials, params.num_episodes, params.init_alpha, params.alpha_min, params.init_epsilon, params.epsilon_min, params.init_phi, params.phi_min, params.discount, params.subspaces, params.size_state_vars))
    file1.close()

    # Write to environment results directory the parameters
    # pickle the results

    # episode_rewards = np.array(episode_rewards)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithms', nargs='+', default=['Q','QLiA'], type=str)
    parser.add_argument('--env', default='taxi', type=str)
    parser.add_argument('--num_trials', default=1, type=int)
    parser.add_argument('--verbose', default=False, type=bool)

    args = parser.parse_args()

    # Running the experiment
    alg_names = []
    for alg in args.algorithms:
        alg_names.append(alg)
    env_name = args.env
    num_trials = args.num_trials
    verbose = args.verbose

    run_experiment(num_trials, env_name, alg_names, verbose)
