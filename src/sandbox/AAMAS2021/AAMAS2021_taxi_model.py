from envs.taxi import TaxiEnv
from envs.taxilarge import TaxiLargeEnv
import numpy as np
import pickle
import mdptoolbox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import datetime
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

discount = 0.9

test_init_data = [10, 20, 50, 75, 100, 200, 500, 600, 800]
repeat_trials = 10
state_size_vars = [5, 5, 5, 4]
Rmin = -20
max_steps = 100
num_test_episodes = 20
env = TaxiEnv()
int_vars = np.array([0, 1])
ext_vars = np.array([2, 3])
action_space = env.action_space.n
observation_space = env.observation_space.n

# test_init_data = [1, 10, 20, 50, 75, 100, 200, 500, 1000, 5000]
# repeat_trials = 2
# env = TaxiLargeEnv()
# state_size_vars = [10, 10, 9, 8]
# Rmin = -20


def run_experiments(num_starting_states):
    # Need to generate trajectories from some initial states using a trained agent

    # Generate initial states
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

    # Create an imitation policy
    imitation_policy = {}

    # Model the transitions
    # For each internal variable, train a regression model on (var, action, next_var)
    X = []
    y = []

    for t in trajectories:
        internals = list(np.array(t[0])[int_vars])
        action = t[1]
        next_internals = list(np.array(t[3])[int_vars])
        imitation_policy[env.encode(*t[0])] = action
        X.append([*internals, action])
        y.append(np.array(next_internals) - np.array(internals))

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    reg = MLPRegressor(max_iter=1000).fit(X_train, y_train)
    print(reg.score(X_test, y_test))
    # reg = LinearRegression().fit(X, y)
    # print([2, 3, 1], reg.predict(np.array([[2, 3, 1]])), np.rint(reg.predict(np.array([[2, 3, 1]]))))

    # Run some trials
    print("{} Running REStore".format(datetime.datetime.now()))
    test_init_states = []
    restore_episode_steps = []
    restore_episode_rewards = []
    for e in range(num_test_episodes):
        # print(e)
        state = env.reset()
        test_init_states.append(state)
        done = False
        ep_reward = 0
        step = 0
        call = 0
        while not done:
            if state in imitation_policy:
                action = imitation_policy[state]
            else:
                # Do rollouts and pick the best action??
                state_vars = list(env.decode(state))
                internal_state = list(np.array(state_vars)[int_vars])
                external_state = list(np.array(state_vars)[ext_vars])
                num_steps_to_goal = []
                actions = []
                for a0 in range(action_space):
                    pred_next_internal_state = np.abs(np.rint(reg.predict(np.array([[*internal_state, a0]])))[0]) + np.array(internal_state)
                    pred_next_state = int(env.encode(*(np.rint(list(pred_next_internal_state) + external_state))))
                    if pred_next_state in imitation_policy:
                        actions.append(a0)
                        num_steps_to_goal.append(1)
                    else:
                        for a1 in range(action_space):
                            pred_next_internal_state = np.abs(np.rint(reg.predict(np.array([[*pred_next_internal_state, a1]])))[0])
                            pred_next_state = int(env.encode(*(np.rint(list(pred_next_internal_state) + external_state))))
                            if pred_next_state in imitation_policy:
                                actions.append(a0)
                                num_steps_to_goal.append(2)
                            else:
                                for a2 in range(action_space):
                                    pred_next_internal_state = np.abs(np.rint(reg.predict(np.array([[*pred_next_internal_state, a2]])))[0])
                                    pred_next_state = int(env.encode(*(np.rint(list(pred_next_internal_state) + external_state))))
                                    if pred_next_state in imitation_policy:
                                        actions.append(a0)
                                        num_steps_to_goal.append(3)
                                    # else:
                                    #     for a3 in range(action_space):
                                    #         pred_next_internal_state = np.abs(np.rint(reg.predict(np.array([[*pred_next_internal_state, a3]])))[0])
                                    #         pred_next_state = int(env.encode(*(np.rint(list(pred_next_internal_state) + external_state))))
                                    #         if pred_next_state in imitation_policy:
                                    #             actions.append(a0)
                                    #             num_steps_to_goal.append(4)
                                            # else:
                                            #     for a4 in range(action_space):
                                            #         pred_next_internal_state = \
                                            #         np.abs(np.rint(reg.predict(np.array([[*pred_next_internal_state, a4]])))[0])
                                            #         pred_next_state = int(env.encode(
                                            #             *(np.rint(list(pred_next_internal_state) + external_state))))
                                            #         if pred_next_state in imitation_policy:
                                            #             actions.append(a0)
                                            #             num_steps_to_goal.append(5)
                if actions:
                    # print(actions, np.flatnonzero(num_steps_to_goal == np.array(num_steps_to_goal).min()), num_steps_to_goal, state_vars)
                    action = actions[int(np.random.choice(np.flatnonzero(num_steps_to_goal == np.array(num_steps_to_goal).min())))]
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
        restore_episode_steps.append(call)  # step)

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
                # reward = 0
                # next_state = -1
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

    experimental_runs = [restore_episode_rewards, random_episode_rewards, Q_episode_rewards]  # reset_episode_rewards,
    experimental_steps = [restore_episode_steps, random_episode_steps, Q_episode_steps]  # reset_episode_rewards,
    return experimental_runs, experimental_steps


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
linestyles=['-', '-', '--']
CB_color_cycle = ['#377eb8', '#ff7f00', 'gray',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
algs = ['LOST', 'Baseline', 'Optimal']
for it, trial in enumerate(experiment_returns):
    std = experiment_stds[it]
    c = np.abs(1.96 * np.divide(std, trial, out=np.zeros_like(std), where=trial != 0))
    for ix, init_data in enumerate(c):
        if init_data > experiment_returns[-1][ix]:
            c[ix] = experiment_returns[-1][ix]

    if it == len(experiment_returns) - 1:
        # plt.errorbar(test_init_data, trial, yerr=c, linestyle=linestyles[it], color=CB_color_cycle[it], alpha=0.5)
        axs[1].plot(test_init_data, trial, linestyle=linestyles[it], color=CB_color_cycle[it])
    else:
        axs[1].errorbar(test_init_data, trial, yerr=c, linestyle=linestyles[it], color=CB_color_cycle[it], alpha=0.5)
        axs[1].plot(test_init_data, trial, linestyle=linestyles[it], color=CB_color_cycle[it])

    axs[0].plot(test_init_data, experiment_steps[it], linestyle=linestyles[it], color=CB_color_cycle[it])

    # plt.fill_between(test_init_data, (trial - c), (trial + c), alpha=.1)

# plt.ylim(-50, 10)
axs[0].set_xticklabels([])
# axs[0].spines["bottom"].set_visible(False)
axs[0].set_ylabel('Calls to Expert')
# axs[0].yaxis.set_label_position("right")
# axs[0].yaxis.tick_right()

# axs[1].set_yscale("symlog")
axs[1].set_ylabel('Episode Returns')
axs[1].legend(algs, loc='lower right')
# axs[1].spines['top'].set_visible(False)
axs[1].set_xlabel('Number of Expert Trajectories Given')

# plt.subplots_adjust(wspace=0, hspace=0)
plt.show()