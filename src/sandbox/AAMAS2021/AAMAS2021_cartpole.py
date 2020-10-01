from envs.cartpole import CartPoleEnv
import numpy as np
import pickle
import mdptoolbox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import datetime, os
import gym

from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

from stable_baselines import DQN, PPO1, SAC
from stable_baselines.gail import generate_expert_traj

DISCOUNT = 0.99
test_init_data = [1, 5, 10, 20, 50, 100, 200, 500]
repeat_trials = 30
env = CartPoleEnv()
state_limits = [[-2.4, 2.4], [-3, 3], [-0.5, 0.5], [-2, 2]]

int_vars = np.array([0])
ext_vars = np.array([1, 2, 3])
action_space = 2
n_bins = [4, 8, 8, 8]
state_size_vars = np.array(n_bins) + 1
observation_space = np.product(state_size_vars)
num_test_episodes = 20

#       # Train a DQN agent for 1e5 timesteps and generate 10 trajectories
#       # data will be saved in a numpy archive
# model = PPO1('MlpPolicy', 'CartPole-v0', verbose=2)
# test = generate_expert_traj(model, 'expert_cartpole_{}'.format(1000), n_timesteps=int(1e5), n_episodes=1000)
#
# quit()


def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def discretize_state(state):
    bins = []
    for nx, n in enumerate(n_bins):
        min = state_limits[nx][0]
        max = state_limits[nx][1]
        bins.append(np.linspace(min, max, n))

    discrete_state = []

    for sx, st_var in enumerate(state):
        inds = np.digitize(st_var, bins[sx] + 1)
        discrete_state.append(inds)

    return discrete_state


def encode_state(discretized_state):
    i = discretized_state[0]
    i *= n_bins[1]
    i += discretized_state[1]
    i *= n_bins[2]
    i += discretized_state[2]
    i *= n_bins[3]
    i += discretized_state[3]
    return i


def generate_trajectories(num_starting_states):
    Q_table = np.zeros([np.product(np.array(n_bins)), action_space])
    episode_rewards = []
    epsilon = 0.5
    alpha = 0.5
    good_trajectories = []
    visited_states = []
    n_trajectories = 0
    for i in range(100000):
        ep_reward = 0
        decoded_state = discretize_state(env.reset())
        state = encode_state(decoded_state)
        trajectory = []
        ep_visited = []
        for t in range(200):
            # print(env.env.state, state)
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(action_space)
            else:
                action = np.argmax(Q_table[state])
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            decoded_next_state = discretize_state(next_state)
            next_state = encode_state(decoded_next_state)
            trajectory.append([decoded_state, action, reward, decoded_next_state])
            ep_visited.append(state)
            if done:
                reward = -200
                Q_table[state][action] += alpha * (
                        reward + (not done) * DISCOUNT * max(Q_table[next_state]) - Q_table[state][action])
                break
            else:
                Q_table[state][action] += alpha * (
                        reward + (not done) * DISCOUNT * max(Q_table[next_state]) - Q_table[state][action])
            decoded_state = decoded_next_state
            state = next_state
        # if i > 10 and np.average(episode_rewards[:-10]) > 150 and ep_reward == 200:
        #     good_trajectories = good_trajectories + trajectory
        #     visited_states = visited_states + ep_visited
        #     n_trajectories += 1
        #     print("{} collected {} trajectories".format(datetime.datetime.now(), n_trajectories))
        #     if n_trajectories >= num_starting_states:
        #         return Q_table, good_trajectories, visited_states
        episode_rewards.append(ep_reward)
        print(i, ep_reward, np.average(episode_rewards[:-10]))
        if epsilon > 0.01:
            epsilon *= 0.99
        if alpha > 0.01:
            alpha *= 0.999

    Q_save = []

    for s in range(observation_space):
        Q_save.append([s, Q_table[s]])

    exp_dir = "{}_{}".format(datetime.datetime.now(), 'Cartpole')
    os.mkdir(exp_dir)

    filename = '{}/Q_values_{}'.format(exp_dir, str(env))
    outfile = open(filename, 'wb')
    pickle.dump(Q_save, outfile)
    outfile.close()

    # print("ERROR: Could not generate enough good data")

# plt.plot(moving_average(episode_rewards, 50))
# plt.show()


def generate_expert_trajectories(model, num_starting_states):
    episode_rewards = []
    trajectories = []
    visited_states = []
    successes = 0
    while successes < num_starting_states:
        ep_reward = 0
        state = env.reset()
        decoded_state = discretize_state(state)
        possible_trajectory = []
        possible_visited_states = []
        for i in range(5):
            action = model.predict([state])[0]
                # print(state, model.predict([state]))
                # print(state, model.predict_proba([state]))
            next_state, reward, done, _ = env.step(action)
            decoded_next_state = discretize_state(next_state)
            possible_trajectory.append([decoded_state, action, reward, decoded_next_state])
            if encode_state(decoded_state) not in visited_states:
                possible_visited_states.append(encode_state(decoded_state))
            ep_reward += reward
            if done:
                break
            state = next_state
            decoded_state = decoded_next_state
        if 1:
            successes += 1
            trajectories = trajectories + possible_trajectory
            visited_states = visited_states + possible_visited_states
        episode_rewards.append(ep_reward)
    return trajectories, visited_states


def run_experiments(num_starting_states):
    expert = np.load('expert_cartpole_{}.npz'.format(1000))
    # reg = MLPRegressor(max_iter=1000).fit(expert['obs'], np.ravel(expert['actions']))
    expert_model = MLPClassifier(max_iter=1000).fit(expert['obs'], np.ravel(expert['actions']).astype(int))

    # Generate n good trajectories
    print("{} generating trajectories ...".format(datetime.datetime.now()))
    trajectories, visited_states = generate_expert_trajectories(expert_model, num_starting_states) # generate_trajectories(num_starting_states)

    possible_externals = [list(map(int, l)) for l in list(itertools.permutations([''.join(str(k)) for k in range(8)], len(ext_vars)))]

    # Create a correction MDP and solve it
    print("{} creating MDP".format(datetime.datetime.now()))
    transitions = np.zeros([action_space, observation_space + 1, observation_space + 1])  # maxS + 1 is a sink state
    rewards = np.zeros([action_space, observation_space + 1, observation_space + 1])
    abstract_valid_states = []
    recorded = []
    for t in trajectories:
        internals = list(np.array(t[0])[int_vars])
        action = t[1]
        next_internals = list(np.array(t[3])[int_vars])
        # check if you already recorded an entry for this (state, action) pair
        # print(internals, next_internals)
        if [internals, action] not in recorded:
            for pe in possible_externals:
                imagined_state = encode_state((internals + pe))
                if np.sum(transitions[0][imagined_state]) <= 0. and np.sum(transitions[1][imagined_state]) <= 0.:
                    if imagined_state in visited_states:
                        imagined_next_state = observation_space
                        rewards[action][imagined_state][imagined_next_state] = 1
                    else:
                        imagined_next_state = encode_state((next_internals + pe))
                        rewards[action][imagined_state][imagined_next_state] = 0
                    transitions[action][imagined_state][imagined_next_state] = 1
                    if np.sum(transitions[action][imagined_state]) > 1:
                        print("ERROR: Transition probabilities sum to more than 1")
                        print(transitions[action][imagined_state])
                        quit()
                    abstract_valid_states.append(imagined_state)
            recorded.append([internals, action])

    # Add all transitions to sink states with -1 reward for actions that aren't covered and states that aren't covered
    for a in range(action_space):
        for s in range(observation_space):
            if np.sum(transitions[a][s]) == 0:
                transitions[a][s][observation_space] = 1  # Going to the sink state
                rewards[a][s][observation_space] = -1  # With negative reward
        # Don't forget to define the transition for the sink state
        transitions[a][observation_space][observation_space] = 1  # Sink state transitions to itself
        rewards[a][observation_space][observation_space] = 0  # With no reward

    print("{} MDP created".format(datetime.datetime.now()))

    print("{} solving MDP".format(datetime.datetime.now()))
    pi = mdptoolbox.mdp.PolicyIteration(transitions, rewards, DISCOUNT)
    pi.run()
    print("{} solved MDP".format(datetime.datetime.now()))

    # for ip, p in enumerate(pi.policy):
    #     print(ip, p, pi.V[ip])

    # Run some trials
    print("{} Running REStore".format(datetime.datetime.now()))
    test_init_states = []
    restore_episode_rewards = []
    restore_episode_calls = []
    test_init_states = []
    for e in range(num_test_episodes):
        ep_reward = 0
        raw_state = env.reset()
        test_init_states.append(raw_state)
        decoded_state = discretize_state(raw_state)
        state = encode_state(decoded_state)
        call = 0
        for t in range(200):
            if state in visited_states:
                action = expert_model.predict([raw_state])[0]
            elif state in abstract_valid_states:
                action = pi.policy[state]
            else:
                if call == 0:
                    call = 1
                action = expert_model.predict([raw_state])[0]
                # action = env.action_space.sample()
            raw_next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            next_state = encode_state(discretize_state(raw_next_state))
            state = next_state
            raw_state = raw_next_state
            if done:
                break

        restore_episode_rewards.append(ep_reward)
        restore_episode_calls.append(call)
#     # REStore REinforcement learning with STate Restoration

    random_episode_rewards = []
    random_episode_calls = []
    print("{} Running random".format(datetime.datetime.now()))
    for e in range(num_test_episodes):
        ep_reward = 0
        raw_state = env.reset(test_init_states[e])
        decoded_state = discretize_state(raw_state)
        state = encode_state(decoded_state)
        call = 0
        for t in range(200):
            if state in visited_states:
                action = expert_model.predict([raw_state])[0]
            else:
                if call == 0:
                    call = 1
                action = expert_model.predict([raw_state])[0]
                # action = env.action_space.sample()
            raw_next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            next_state = encode_state(discretize_state(raw_next_state))
            state = next_state
            raw_state = raw_next_state
            if done:
                break
        random_episode_rewards.append(ep_reward)
        random_episode_calls.append(call)

    Q_episode_rewards = []
    # Q_episode_rewards = np.repeat(200, num_test_episodes)
    Q_episode_calls = np.repeat(0, num_test_episodes)
    print("{} Running oracle".format(datetime.datetime.now()))
    for e in range(num_test_episodes):
        ep_reward = 0
        state = env.reset(test_init_states[e])
        for t in range(200):
            action = expert_model.predict([state])[0]
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if done:
                break
        Q_episode_rewards.append(ep_reward)
#
    experimental_runs = [restore_episode_rewards, random_episode_rewards, Q_episode_rewards]
    experimental_calls = [restore_episode_calls, random_episode_calls, Q_episode_calls]  # reset_episode_rewards,
    return experimental_runs, experimental_calls


experiment_returns = []
experiment_steps = []
experiment_stds = []

for tid in test_init_data:
    trial_returns = []
    trial_steps = []
    for r in range(repeat_trials):
        ret, ste = run_experiments(tid)
        trial_returns.append(ret)
        trial_steps.append(ste)
        print("{} finished trial {} with rewards {} and steps {}".format(datetime.datetime.now(), r, np.average(ret, axis=1), np.average(ste, axis=1)))
    returns = np.average(trial_returns, axis=0)
    steps = np.average(trial_steps, axis=0)
    averaged_returns = np.average(returns, axis=1)
    averaged_steps = np.average(steps, axis=1)
    std_returns = np.std(returns, axis=1)
    experiment_returns.append(averaged_returns)
    experiment_steps.append(averaged_steps)
    experiment_stds.append(std_returns)
    print("{} done {} ".format(datetime.datetime.now(), tid))
    print(experiment_returns[-1], experiment_steps[-1])

fig, axs = plt.subplots(2)
gs1 = gridspec.GridSpec(2, 1)
gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

test_init_data = np.transpose(test_init_data)
experiment_returns = np.transpose(experiment_returns)
experiment_steps = np.transpose(experiment_steps)
experiment_stds = np.transpose(experiment_stds)
linestyles=['-', '-', '--']
CB_color_cycle = ['#377eb8', '#ff7f00', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
algs = ['LOST', 'Imitator', 'Optimal']
for it, trial in enumerate(experiment_returns):
    std = experiment_stds[it]
    c = np.abs(1.96 * np.divide(std, trial, out=np.zeros_like(std), where=trial != 0))
    for ix, init_data in enumerate(c):
        if init_data > experiment_returns[-1][ix]:
            c[ix] = experiment_returns[-1][ix]

    if it == len(experiment_returns) - 1:
        # plt.errorbar(test_init_data, trial, yerr=c, linestyle=linestyles[it], color=CB_color_cycle[it], alpha=0.5)
        axs[1].plot(test_init_data, trial, linestyle=linestyles[it], color='gray')
        axs[0].plot(test_init_data, experiment_steps[it], linestyle=linestyles[it], color='gray')
    else:
        axs[1].errorbar(test_init_data, trial, yerr=c, linestyle=linestyles[it], color=CB_color_cycle[it], alpha=0.5)
        axs[1].plot(test_init_data, trial, linestyle=linestyles[it], color=CB_color_cycle[it])
        axs[0].plot(test_init_data, experiment_steps[it], linestyle=linestyles[it], color=CB_color_cycle[it])

    # plt.fill_between(test_init_data, (trial - c), (trial + c), alpha=.1)

# plt.ylim(-50, 10)
axs[0].set_xticklabels([])
# axs[0].spines["bottom"].set_visible(False)
axs[0].set_ylabel('Calls to Expert (per episode)')
# axs[0].yaxis.set_label_position("right")
# axs[0].yaxis.tick_right()

# axs[1].set_yscale("symlog")
axs[1].set_ylabel('Total Returns (per episode)')
axs[1].legend(algs, loc='lower right')
# axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel('Number of Expert Trajectories Given')

# plt.subplots_adjust(wspace=0, hspace=0)
plt.show()