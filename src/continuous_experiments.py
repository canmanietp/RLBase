import gym
import numpy as np
import pandas as pd
import time, sys, os, datetime, copy, csv
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
# from silence_tensorflow import silence_tensorflow
from matplotlib import pyplot as plt

# silence_tensorflow()

from agents.DQN import DQNAgent
from agents.DQNLiA import DQNLiAAgent
from agents.DQNVP import DQNVPAgent
from learning_parameters import ContinuousParameters
from helpers import plotting

os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"


def get_params_waterworld():
    memory_size = 50000
    batch_size = 32
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.9
    decay_rate = 0.99
    num_episodes = 500
    retrain_steps = 150
    observation_space = 13
    action_space = 4
    learning_rate = 0.00001
    sub_spaces = []
    sub_models = []
    meta_model = []
    model = Sequential()
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return ContinuousParameters(init_model=model, meta_model=meta_model, sub_models=sub_models, memory_size=memory_size,
                                batch_size=batch_size, learning_rate=learning_rate, epsilon=init_epsilon,
                                epsilon_min=epsilon_min,
                                discount=discount, decay=decay_rate, observation_space=observation_space,
                                action_space=action_space,
                                num_episodes=num_episodes, retrain_steps=retrain_steps, phi=init_phi, phi_min=phi_min,
                                sub_spaces=sub_spaces)


def get_params_pong():
    memory_size = 1000000
    batch_size = 32
    init_epsilon = 0.3
    epsilon_min = 0.01
    init_phi = 0.3
    phi_min = 0.01
    discount = 0.95
    decay_rate = 0.999
    num_episodes = 10000
    retrain_steps = 1000
    observation_space = 8*2
    action_space = 6
    learning_rate = 0.0001
    sub_spaces = [[1, 5, 9, 13], [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13]]
    # --- Regular DQN model (input: full state, output: action)
    model = Sequential()
    model.add(Dense(128, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # --- Meta DQN model (input: full state, output: abstraction)
    meta_model = Sequential()
    meta_model.add(Dense(128, input_dim=observation_space, activation='relu'))
    meta_model.add(Dense(64, activation='relu'))
    meta_model.add(Dense(24, activation='relu'))
    meta_model.add(Dense(12, activation='relu'))
    meta_model.add(Dense(len(sub_spaces), activation='linear'))
    meta_model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # --- Submodel 1 (input: subspace1, output: action)
    sub_model = Sequential()
    sub_model.add(Dense(64, input_dim=len(sub_spaces[0]), activation='relu'))
    sub_model.add(Dense(24, activation='relu'))
    sub_model.add(Dense(12, activation='relu'))
    sub_model.add(Dense(action_space, activation='linear'))
    sub_model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # --- Submodel 2 (input: subspace2, output:action)
    sub_model2 = Sequential()
    sub_model2.add(Dense(128, input_dim=len(sub_spaces[1]), activation='relu'))
    sub_model2.add(Dense(64, activation='relu'))
    sub_model2.add(Dense(24, activation='relu'))
    sub_model2.add(Dense(12, activation='relu'))
    sub_model2.add(Dense(action_space, activation='linear'))
    sub_model2.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    sub_models = [sub_model, sub_model2]
    return ContinuousParameters(init_model=model, meta_model=meta_model, sub_models=sub_models, memory_size=memory_size,
                                batch_size=batch_size, learning_rate=learning_rate, epsilon=init_epsilon,
                                epsilon_min=epsilon_min,
                                discount=discount, decay=decay_rate, observation_space=observation_space,
                                action_space=action_space,
                                num_episodes=num_episodes, retrain_steps=retrain_steps, phi=init_phi, phi_min=phi_min,
                                sub_spaces=sub_spaces)


def get_params_mspacman():
    memory_size = 1000000
    batch_size = 32
    init_epsilon = 0.5
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.95
    decay_rate = 0.999
    num_episodes = 10000
    retrain_steps = 100
    observation_space = 17
    action_space = 9
    learning_rate = 0.0001
    sub_spaces = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], []]
    # --- Regular DQN model (input: full state, output: action)
    model = Sequential()
    model.add(Dense(64, input_dim=observation_space, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    meta_model = None
    sub_models = None
    return ContinuousParameters(init_model=model, meta_model=meta_model, sub_models=sub_models, memory_size=memory_size,
                                batch_size=batch_size,
                                learning_rate=learning_rate, epsilon=init_epsilon, epsilon_min=epsilon_min,
                                discount=discount, decay=decay_rate, observation_space=observation_space,
                                action_space=action_space,
                                num_episodes=num_episodes, retrain_steps=retrain_steps, phi=init_phi, phi_min=phi_min,
                                sub_spaces=sub_spaces)


def get_params_coffeemail():
    memory_size = 100000
    batch_size = 64
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.95
    decay_rate = 0.95
    num_episodes = 100
    retrain_steps = 20
    observation_space = 4
    action_space = 4
    learning_rate = 0.001
    # --- Regular DQN model (input: full state, output: action)
    model = Sequential()
    model.add(Dense(12, input_dim=observation_space, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # for abstraction methods
    sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    # -- Model for choosing sub_space agent (input: full state, output: sub_agent)
    meta_model = Sequential()
    meta_model.add(Dense(24, input_dim=observation_space, activation='relu'))
    meta_model.add(Dense(24, activation='relu'))
    meta_model.add(Dense(len(sub_spaces), activation='linear'))
    meta_model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # --- DQN model for sub_space1 (input: sub_space1, output: action)
    sub_model1 = Sequential()
    sub_model1.add(Dense(24, input_dim=len(sub_spaces[0]), activation='relu'))
    sub_model1.add(Dense(24, activation='relu'))
    sub_model1.add(Dense(action_space, activation='linear'))
    sub_model1.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # --- DQN model for sub_space1 (input: sub_space2, output: action)
    sub_model2 = Sequential()
    sub_model2.add(Dense(24, input_dim=len(sub_spaces[1]), activation='relu'))
    sub_model2.add(Dense(24, activation='relu'))
    sub_model2.add(Dense(action_space, activation='linear'))
    sub_model2.compile(loss='mse', optimizer=Adam(lr=0.01))
    sub_models = [sub_model1, sub_model2]
    return ContinuousParameters(init_model=model, meta_model=meta_model, sub_models=sub_models, memory_size=memory_size,
                                batch_size=batch_size,
                                learning_rate=learning_rate, epsilon=init_epsilon, epsilon_min=epsilon_min,
                                discount=discount, decay=decay_rate, observation_space=observation_space,
                                action_space=action_space,
                                num_episodes=num_episodes, retrain_steps=retrain_steps, phi=init_phi, phi_min=phi_min,
                                sub_spaces=sub_spaces)


def get_params_cartpole():
    memory_size = 10000
    batch_size = 32
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.3
    phi_min = 0.001
    discount = 0.95
    decay_rate = 0.99
    num_episodes = 30
    retrain_steps = 10
    observation_space = 4
    action_space = 2
    learning_rate = 0.01
    sub_spaces = [[1, 3]]
    # --- Regular DQN model (input: full state, output: action)
    model = Sequential()
    model.add(Dense(24, input_dim=observation_space, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_space, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # -- Model for choosing sub_space agent (input: full state, output: sub_agent)
    meta_model = Sequential()
    meta_model.add(Dense(24, input_dim=observation_space, activation='relu'))
    meta_model.add(Dense(24, activation='relu'))
    meta_model.add(Dense(len(sub_spaces), activation='linear'))
    meta_model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # --- DQN model for sub_space1 (input: sub_space1, output: action)
    sub_model1 = Sequential()
    sub_model1.add(Dense(12, input_dim=len(sub_spaces[0]), activation='relu'))
    sub_model1.add(Dense(12, activation='relu'))
    sub_model1.add(Dense(action_space, activation='linear'))
    sub_model1.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    sub_models = [sub_model1, copy.copy(model)]  # , sub_model2]
    return ContinuousParameters(init_model=model, meta_model=meta_model, sub_models=sub_models, memory_size=memory_size,
                                batch_size=batch_size,
                                learning_rate=learning_rate, epsilon=init_epsilon, epsilon_min=epsilon_min,
                                discount=discount, decay=decay_rate, observation_space=observation_space,
                                action_space=action_space,
                                num_episodes=num_episodes, retrain_steps=retrain_steps, phi=init_phi, phi_min=phi_min,
                                sub_spaces=sub_spaces)


def get_params(env_name, alg=None):
    if env_name == 'cartpole':
        from envs.cartpole import CartPoleEnv
        env = CartPoleEnv()
        params = get_params_cartpole()
    elif env_name == 'coffeemail':
        from envs.coffeemail_continuous import CoffeeMailContinuousEnv
        env = CoffeeMailContinuousEnv()
        params = get_params_coffeemail()
    elif env_name == 'mspacman':
        from envs.atariari.benchmark.wrapper import AtariARIWrapper
        env = AtariARIWrapper(gym.make('MsPacmanNoFrameskip-v4'))
        params = get_params_mspacman()
    elif env_name == 'pong':
        from envs.atariari.benchmark.wrapper import AtariARIWrapper
        env = AtariARIWrapper(gym.make('Pong-v0'))
        params = get_params_pong()
    elif env_name == 'waterworld':
        sys.path.append('envs/PyGame-Learning-Environment/')
        from ple.games.waterworld import WaterWorld
        from ple.ple import PLE
        _pygame = WaterWorld()
        env = PLE(_pygame, fps=30, display_screen=True, force_fps=False)
        params = get_params_waterworld()
        return env, params
    else:
        print("Error: Unknown environment")
        return
    return env, params


def run_continuous_experiment(num_trials, env_name, algs, verbose=False, render=False):
    date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_dir = "tmp/{}".format(date_string)
    os.mkdir(exp_dir)
    env, params = get_params(env_name)
    average_every = int(params.num_episodes / 10)

    trial_rewards = []
    trial_times = []

    for t in range(num_trials):
        agents = []
        for alg in algs:
            if alg == 'DQN':
                agents.append(DQNAgent(env, copy.copy(params)))
            elif alg == 'DQNLiA':
                agents.append(DQNLiAAgent(env, copy.copy(params)))
            elif alg == 'DQNVP':
                agents.append(DQNVPAgent(env, copy.copy(params)))
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
            csvfile = open('{}/trial_{}_agent_{}.csv'.format(exp_dir, t + 1, agent.name), 'w', newline='')
            writer = csv.writer(csvfile)
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
                    reward, done = agent.do_step()
                    ep_reward += reward
                    if render:
                        agent.env.render()

                episode_rewards[j].append(ep_reward)
                writer.writerow([i, ep_reward])
                if verbose:
                    print(
                        "{} Episode {}, reward={}, exploration={}".format(datetime.datetime.now().strftime("%H:%M:%S"),
                                                                          i, ep_reward, agent.params.EPSILON))
            csvfile.close()
            run_time = time.time() - t0
            print("{} Finished running in {} seconds".format(datetime.datetime.now().strftime("%H:%M:%S"), run_time))
            times_to_run.append(run_time)

            plt.plot(plotting.moving_average(episode_rewards[j], average_every), label=agent.name)
            plt.legend([a.name for a in agents], loc='lower right')
            plt.savefig('{}/trial_{}'.format(exp_dir, t + 1))

        plt.close()
        trial_rewards.append(episode_rewards)
        pd.DataFrame(np.transpose(episode_rewards)).to_csv('{}/trial_{}.csv'.format(exp_dir, t + 1), header=None,
                                                           index=None)

    for trial in np.average(trial_rewards, axis=0):
        plt.plot(plotting.moving_average(trial, average_every))

    plt.legend([a for a in algs], loc='lower right')
    plt.savefig('{}/final'.format(exp_dir))

    for ia, alg in enumerate(algs):
        env, params = get_params(env_name, alg)
        file = open('{}/params_agent{}.txt'.format(exp_dir, alg), "w")
        file.write("Environment: {}\n"
                   "Number of trials: {}\n"
                   "Number of episodes: {}\n"
                   "Running times: {}\n"
                   "init_learning_rate={}\n"
                   "init_epsilon={}\n"
                   "epsilon_min={}\n"
                   "init_phi={}\n"
                   "phi_min={}\n"
                   "discount={}\n"
                   "sub_spaces={}".format(env, num_trials,
                                          params.num_episodes, trial_times, params.LEARNING_RATE,
                                          params.EPSILON, params.EPSILON_MIN,
                                          params.PHI, params.PHI_MIN, params.DISCOUNT,
                                          params.sub_spaces))

        file.close()

    return
