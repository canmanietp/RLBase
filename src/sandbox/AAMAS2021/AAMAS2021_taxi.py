from envs.taxi import TaxiEnv
from envs.taxilarge import TaxiLargeEnv
import numpy as np
import pickle
import mdptoolbox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import datetime

discount = 0.9

test_init_data = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]
repeat_trials = 5
env = TaxiEnv()
state_size_vars = [5, 5, 5, 4]
Rmin = -20

# test_init_data = [1, 10, 20, 50, 75, 100, 200, 500, 1000, 5000]
# repeat_trials = 10
# env = TaxiLargeEnv()
# state_size_vars = [10, 10, 9, 8]
# Rmin = -20

int_vars = np.array([0, 1])
ext_vars = np.array([2, 3])
action_space = env.action_space.n
observation_space = env.observation_space.n
max_steps = 100
num_test_episodes = 100


def run_experiments(num_starting_states):
    # Need to generate trajectories from some initial states using a trained agent

    # Generate initial states
    possible_externals = [list(map(int, l)) for l in list(itertools.permutations([''.join(str(k)) for k in range(state_size_vars[3])], len(ext_vars)))]  # [[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 0], [2, 1], [2, 3], [3, 0], [3, 1], [3, 2]]
    # randoms = 2
    initial_states = []
    #
    # for pe in possible_externals:
    #     for r in range(randoms):
    #         internals = [np.random.randint(5), np.random.randint(5)]
    #         initial_states.append(internals + pe)

    print("{} generating random initial states".format(datetime.datetime.now()))

    while len(initial_states) < num_starting_states:
        rand_state = env.observation_space.sample()
        state_vars = list(env.decode(rand_state))
        if state_vars[2] != state_vars[3]:  # This only happens in a terminating state
            initial_states.append(state_vars)

    # print(initial_states)
    # Import pre-trained oracle agent
    experience = pickle.load(open('Q_values_{}'.format(str(env)), "rb"))

    # print(experience)
    trajectories = []
    visited_states = []

    print("{} rolling out trajectories".format(datetime.datetime.now()))

    for i_s in initial_states:
        done = False
        env.reset()
        env.s = env.encode(*i_s)
        state = env.s
        step = 0
        while not done:
            # print(state, list(env.decode(state)), [i for i in experience if i[0] == state][0][2])
            action = np.argmax([i for i in experience if i[0] == state][0][2])
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # print(list(env.decode(next_state)))
            trajectories.append([list(env.decode(state)), action, reward, list(env.decode(next_state)), done])
            state = next_state
            visited_states.append(state)
            if step > max_steps:
                done = True
            step += 1

    # Train the imitation Q_table
    # Q_imitation_table = np.full([observation_space, action_space], Rmin / (1 - discount))
    # for n in range(len(trajectories)):
    #     sample = trajectories[n]
    #     state = env.encode(*sample[0])
    #     action = sample[1]
    #     reward = sample[2]
    #     next_state = env.encode(*sample[3])
    #     done = sample[4]
    #     Q_imitation_table[state][action] += reward + (not done) * discount * max(Q_imitation_table[next_state]) - Q_imitation_table[state][action]

    # Create an imitation policy
    imitation_policy = {}

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
        imitation_policy[env.encode(*t[0])] = action
        # check if you already recorded an entry for this (state, action) pair
        if [internals, action] not in recorded:
            for pe in possible_externals:
                imagined_state = env.encode(*(internals + pe))
                if imagined_state in visited_states:
                    imagined_next_state = observation_space
                    rewards[action][imagined_state][imagined_next_state] = 1
                else:
                    imagined_next_state = env.encode(*(next_internals + pe))
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
    pi = mdptoolbox.mdp.PolicyIteration(transitions, rewards, discount)
    pi.run()
    print("{} solved MDP".format(datetime.datetime.now()))

    # for ip, p in enumerate(pi.policy):
    #     print(list(env.decode(ip)), p, pi.V[ip])

    # Run some trials
    print("{} Running REStore".format(datetime.datetime.now()))
    test_init_states = []
    restore_episode_steps = []
    restore_episode_rewards = []
    for e in range(num_test_episodes):
        state = env.reset()
        test_init_states.append(state)
        done = False
        ep_reward = 0
        step = 0
        call = 0
        while not done:
            if state in imitation_policy:
                action = imitation_policy[state]
            elif state in abstract_valid_states:
                action = pi.policy[state]
            else:
                if call == 0:
                    call = 1
                # action = env.action_space.sample()
                action = np.argmax([i for i in experience if i[0] == state][0][2])
            if not done:
                next_state, reward, done, _ = env.step(action)
                # print(state, action, reward, next_state)
                ep_reward += reward
                state = next_state
                if step > max_steps:
                    done = True
                step += 1
        restore_episode_rewards.append(ep_reward)
        restore_episode_steps.append(call) # step)

    # REStore REinforcement learning with STate Restoration

    random_episode_rewards = []
    random_episode_steps = []
    print("{} Running random".format(datetime.datetime.now()))
    for e in range(num_test_episodes):
        state = test_init_states[e]
        env.s = state
        done = False
        ep_reward = 0
        step = 0
        call = 0
        while not done:
            if state in imitation_policy:
                action = imitation_policy[state]
            else:
                if call == 0:
                    call = 1
                # action = env.action_space.sample()
                action = np.argmax([i for i in experience if i[0] == state][0][2])
            if not done:
                next_state, reward, done, _ = env.step(action)
                ep_reward += reward
                state = next_state
                if step > max_steps:
                    done = True
                step += 1
        random_episode_rewards.append(ep_reward)
        random_episode_steps.append(call)

    nn_episode_rewards = []
    nn_episode_steps = []
    print("{} Running NN".format(datetime.datetime.now()))
    for e in range(num_test_episodes):
        state = test_init_states[e]
        env.s = state
        done = False
        ep_reward = 0
        step = 0
        while not done:
            if state in imitation_policy:
                action = imitation_policy[state]
            else:
                closest_states = []
                dist = float("inf")
                state_vars = np.array(list(env.decode(state)))
                for os in imitation_policy:
                    os_state_vars = np.array(list(env.decode(os)))
                    os_dist = np.linalg.norm(state_vars - os_state_vars)
                    if os_dist < dist:
                        closest_states = [os]
                        dist = os_dist
                    elif os_dist == dist:
                        closest_states.append(os)
                action = imitation_policy[np.random.choice(closest_states)]
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state

            if step > max_steps:
                done = True
            step += 1
        nn_episode_rewards.append(ep_reward)
        nn_episode_steps.append(0)

    Q_episode_rewards = []
    Q_episode_steps = []
    print("{} Running oracle".format(datetime.datetime.now()))
    for e in range(num_test_episodes):
        state = test_init_states[e]
        env.s = state
        done = False
        ep_reward = 0
        step = 0
        while not done:
            action = np.argmax([i for i in experience if i[0] == state][0][2])
            # np.random.choice(np.flatnonzero(Q_imitation_table[state] == Q_imitation_table[state].max()))
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            state = next_state
            if step > max_steps:
                done = True
            step += 1
        Q_episode_rewards.append(ep_reward)
        Q_episode_steps.append(0)

    # learn_Q_episode_rewards = []
    # learn_Q_episode_steps = []
    # learn_Q_table = np.zeros([observation_space, action_space])
    # print("{} Running learn Q".format(datetime.datetime.now()))
    # for e in range(num_test_episodes):
    #     state = test_init_states[e]
    #     env.s = state
    #     done = False
    #     ep_reward = 0
    #     step = 0
    #     while not done:
    #         action = np.argmax(learn_Q_table[state])
    #         # np.random.choice(np.flatnonzero(Q_imitation_table[state] == Q_imitation_table[state].max()))
    #         next_state, reward, done, _ = env.step(action)
    #         ep_reward += reward
    #         learn_Q_table[state][action] += (state in imitation_policy and imitation_policy[state] == action) + reward + max(learn_Q_table[next_state]) - learn_Q_table[state][action]
    #         state = next_state
    #         if step > max_steps:
    #             done = True
    #         step += 1
    #     learn_Q_episode_rewards.append(ep_reward)
    #     learn_Q_episode_steps.append(0)

    experimental_runs = [restore_episode_rewards, random_episode_rewards, nn_episode_rewards, Q_episode_rewards]  # reset_episode_rewards,
    experimental_steps = [restore_episode_steps, random_episode_steps, nn_episode_steps, Q_episode_steps]  # reset_episode_rewards,
    return experimental_runs, experimental_steps


def moving_average(a, n=50):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


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
        print("{} finished trial {}".format(datetime.datetime.now(), r))
    returns = np.average(trial_returns, axis=0)
    steps = np.average(trial_steps, axis=0)
    averaged_returns = np.average(returns, axis=1)
    averaged_steps = np.average(steps, axis=1)
    std_returns = np.std(returns, axis=1)
    experiment_returns.append(averaged_returns)
    experiment_steps.append(averaged_steps)
    experiment_stds.append(std_returns)
    print("{} done {}".format(datetime.datetime.now(), tid))
    print(experiment_returns[-1], experiment_steps[-1])

fig, axs = plt.subplots(2)
gs1 = gridspec.GridSpec(2, 1)
gs1.update(wspace=0.025, hspace=0.05)  # set the spacing between axes.

test_init_data = np.transpose(test_init_data)
experiment_returns = np.transpose(experiment_returns)
experiment_steps = np.transpose(experiment_steps)
experiment_stds = np.transpose(experiment_stds)
linestyles=['-', '-', '-', '--']
CB_color_cycle = ['#377eb8', '#ff7f00', '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
algs = ['LOST', 'Imitator', 'NearestNeighbour', 'Optimal']
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
axs[0].set_ylabel('% of episodes where expert was called')
# axs[0].yaxis.set_label_position("right")
# axs[0].yaxis.tick_right()

# axs[1].set_yscale("symlog")
axs[1].set_ylabel('Total returns (per episode)')
axs[1].legend(algs, loc='lower right')
# axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel('Number of expert trajectories given')

# plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

# plt.figure(2)
# test_init_data = np.transpose(test_init_data)
# experiment_steps = np.transpose(experiment_steps)
# linestyles=['-', '-', '--']
# CB_color_cycle = ['#377eb8', '#ff7f00', 'gray',
#                   '#f781bf', '#a65628', '#984ea3',
#                   '#999999', '#e41a1c', '#dede00']
# algs = ['REStore', 'Random', 'Oracle']
# for it, trial in enumerate(experiment_steps):
#     plt.plot(test_init_data, trial, linestyle=linestyles[it], color=CB_color_cycle[it])
#
# plt.yscale("symlog")
# # plt.ylim(-50, 10)
# plt.legend(algs)
# plt.ylabel('Calls to Expert')
# plt.xlabel('Number of Expert Trajectories Given')
# plt.show()

# plt.plot(moving_average(experiment_10_returns[0]))
# plt.plot(moving_average(experiment_10_returns[1]))
# plt.plot(moving_average(experiment_10_returns[2]))
# plt.plot(moving_average(experiment_10_returns[3]))
