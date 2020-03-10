import numpy as np
import pickle
import time, os, datetime, copy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from silence_tensorflow import silence_tensorflow
from matplotlib import pyplot as plt


silence_tensorflow()

from envs.cartpole import CartPoleEnv

from agents.DQN import DQNAgent
from agents.DQNLiA import DQNLiAAgent
from learning_parameters import ContinuousParameters
from helpers import plotting


def get_params_cartpole():
    memory_size = 10000
    batch_size = 32
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.99
    num_episodes = 30
    retrain_steps = 5
    observation_space = 4
    action_space = 2
    learning_rate = 0.01
    sub_spaces = [[0, 1, 2, 3]]
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
    sub_model1.add(Dense(24, input_dim=len(sub_spaces[0]), activation='relu'))
    sub_model1.add(Dense(24, activation='relu'))
    sub_model1.add(Dense(action_space, activation='linear'))
    sub_model1.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    sub_model2 = model
    sub_models = [sub_model1, sub_model2]
    return ContinuousParameters(init_model=model, meta_model=meta_model, sub_models=sub_models, memory_size=memory_size,
                                batch_size=batch_size,
                                learning_rate=learning_rate, epsilon=init_epsilon, epsilon_min=epsilon_min,
                                discount=discount, decay=decay_rate, observation_space=observation_space,
                                num_episodes=num_episodes, retrain_steps=retrain_steps, phi=init_phi, phi_min=phi_min,
                                sub_spaces=sub_spaces)


def get_params(env_name):
    if env_name == 'cartpole':
        env = CartPoleEnv()
        params = get_params_cartpole()
    else:
        print("Error: Unknown environment")
        return
    return env, params


def run_continuous_experiment(num_trials, env_name, algs, verbose=False):
    date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_dir = "tmp/{}".format(date_string)
    os.mkdir(exp_dir)
    env, params = get_params(env_name)
    average_every = 1

    trial_rewards = []
    trial_times = []

    for t in range(num_trials):
        agents = []
        for alg in algs:
            if alg == 'DQN':
                agents.append(DQNAgent(env, params))
            elif alg == 'DQNLiA':
                agents.append(DQNLiAAgent(env, params))
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
                    print(i, reward)

                episode_rewards[j].append(ep_reward)
                if verbose:
                    print("{} Episode {}, reward={}".format(datetime.datetime.now().strftime("%H:%M:%S"), i, ep_reward))
            run_time = time.time() - t0
            print("{} Finished running in {} seconds".format(datetime.datetime.now().strftime("%H:%M:%S"), run_time))
            times_to_run.append(run_time)

            plt.plot(plotting.moving_average(episode_rewards[j], average_every), label=agent.name)
            plt.legend([a.name for a in agents], loc='lower right')
            plt.savefig('{}/trial_{}'.format(exp_dir, t + 1))

        plt.close()
        trial_rewards.append(episode_rewards)

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
                "Model: {}\n"
                "memory_size: {}\n"
                "batch_size: {}\n"
                "learning_rate={}\n"
                "init_epsilon={}\n"
                "epsilon_min={}\n"
                "init_phi={}\n"
                "phi_min={}\n"
                "discount={}\n"
                "sub_spaces={}".format(env, num_trials,
                                       params.num_episodes, algs, trial_times, params.INIT_MODEL,
                                       params.MEMORY_SIZE, params.BATCH_SIZE,
                                       params.LEARNING_RATE,
                                       params.EPSILON, params.EPSILON_MIN,
                                       params.PHI, params.PHI_MIN, params.DISCOUNT,
                                       params.sub_spaces))
    file1.close()
    trial_rewards = np.array(trial_rewards)
    pickle.dump(trial_rewards, open('{}/save.p'.format(exp_dir), "wb"))
    return
