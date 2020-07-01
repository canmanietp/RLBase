from envs.taxi import TaxiEnv
import numpy as np
import copy, random

env = TaxiEnv()
state_size_vars = [5, 5, 5, 4]
convergence_criteria = 1e-15
action_space = 6


def encode_abs_state(state, abstraction):
    abs_state = [state[k] for k in abstraction]
    var_size = copy.copy([state_size_vars[k] for k in abstraction])
    var_size.pop(0)
    encoded_state = 0

    for e in range(len(abs_state) - 1):
        encoded_state += abs_state[e] * np.prod(var_size)
        var_size.pop(0)

    encoded_state += abs_state[-1]
    return encoded_state


def e_greedy_action(Q_table, state):
    if random.uniform(0, 1) < EPSILON:
        return np.random.randint(action_space)
    if all([x == 0 for x in Q_table[state]]):
        return np.random.randint(action_space)
    qv = Q_table[state]
    return np.random.choice(np.flatnonzero(qv == qv.max()))


def init_Qs():
    Q_tables = []
    for sz in abs_size:
        table = np.zeros((sz, action_space))
        Q_tables.append(table)
    return Q_tables


num_episodes = 500

abs_vars = [[0], [1], [2], [3], [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [0, 1, 2], [0, 1, 3], [0, 2, 3], [0, 1, 2, 3]]
abs_size = [5, 5, 5, 4, 25, 25, 20, 25, 20, 20, 125, 125, 100, 500]
active_ab = [copy.copy(abs_vars) for os in range(env.observation_space.n)]
last_converged_value = [float("-inf") for o in range(env.observation_space.n)]
last_ab = [-1 for ob in range(env.observation_space.n)]
converged = False
Q_tables = init_Qs()
history = []
sa_visits = np.zeros((env.observation_space.n, action_space))
episode_rewards = []
EPSILON = 0.3
epsilon_min = 0.001
ALPHA = 0.3
DISCOUNT = 0.95

for i in range(num_episodes):
    state = env.reset()
    state_vars = list(env.decode(state))
    done = False
    ep_reward = 0

    while not done:
        ab_id = abs_vars.index(active_ab[state][0])
        abs_state = encode_abs_state(state_vars, abs_vars[ab_id])
        action = e_greedy_action(Q_tables[ab_id], abs_state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        history.append((state, action, next_state, reward, done))
        sa_visits[state][action] += 1
        next_state_vars = list(env.decode(next_state))
        abs_next_state = encode_abs_state(next_state_vars, abs_vars[ab_id])

        ab_id = abs_vars.index(active_ab[state][0])
        td_error = reward + (not done) * DISCOUNT * max(Q_tables[ab_id][abs_next_state]) - \
                   Q_tables[ab_id][abs_state][action]
        prev_Q = Q_tables[ab_id][abs_state][action]
        Q_tables[ab_id][abs_state][action] += ALPHA * td_error
        converged_value = Q_tables[ab_id][abs_state][action]
        converged = abs(converged_value - prev_Q) < convergence_criteria

        if converged and ab_id != len(abs_vars) - 1:
            if np.sum(Q_tables[ab_id][abs_state]) > last_converged_value[state]:
                # print(i, "GOOD", abs_vars[ab_id])
                last_converged_value[state] = np.sum(Q_tables[ab_id][abs_state])
                # print("good")
                # print(state, active_ab[state], last_ab[state], abs_vars[ab_id])
                if last_ab[state] in active_ab[state] and last_ab[state] != abs_vars[-1]:
                    active_ab[state].remove(last_ab[state])  # remove last abstraction
                last_ab[state] = abs_vars[ab_id]  # save current abstraction
                active_ab[state].pop(0)  # remove self
                # print(state, active_ab[state], abs_vars[ab_id])
            else:
                # print(i, "NO GOOD", abs_vars[ab_id])
                # print("bad")
                # print(state, active_ab[state], abs_vars[ab_id])
                if abs_vars[ab_id] != abs_vars[-1]:
                    active_ab[state].remove(abs_vars[ab_id])  # remove self
                if last_ab[state] != abs_vars[ab_id]:
                    active_ab[state].insert(0, last_ab[state])  # add last abstraction
                # print(state, active_ab[state])

                # for st in range(env.observation_space.n):
                #     st_vars = list(env.decode(st))
                #     abs_st = encode_abs_state(st_vars, abs_vars[ab_id])
                #     if abs_st == abs_state:
                #         last_ab[st] = last_ab[abs_vars.index(active_ab[state][0])]
                #         active_ab[st] = copy.copy(active_ab[state])
                #         ab_id = abs_vars.index(active_ab[state][0])
                #         new_abs_state = encode_abs_state(st_vars, abs_vars[ab_id])
                #         last_converged_value[st] = np.sum(Q_tables[ab_id][new_abs_state])

        for abba in active_ab[state]:
            abba_id = abs_vars.index(abba)
            if abba_id != ab_id:
                abs_state = encode_abs_state(state_vars, abs_vars[abba_id])
                abs_next_state = encode_abs_state(next_state_vars, abs_vars[abba_id])
                td_error = reward + (not done) * DISCOUNT * max(Q_tables[abba_id][abs_next_state]) - Q_tables[abba_id][abs_state][action]
                Q_tables[abba_id][abs_state][action] += ALPHA * td_error

        if done:
            if EPSILON > epsilon_min:
                EPSILON *= 0.99

        state = next_state
        state_vars = next_state_vars

    print("Episode", i, "reward", ep_reward)
    episode_rewards.append(ep_reward)

for obs in range(env.observation_space.n):
    if active_ab[obs] != abs_vars and len(active_ab[obs]) != 1:
        state_vars = list(env.decode(obs))
        ab_id = abs_vars.index(active_ab[obs][0])
        abs_state = encode_abs_state(state_vars, abs_vars[ab_id])
        print(state_vars, active_ab[obs], np.argmax(Q_tables[ab_id][abs_state]))

print("Build-up", np.mean(episode_rewards[-100:], axis=0))

Q_table = np.zeros((env.observation_space.n, action_space))
sa_visits = np.zeros((env.observation_space.n, action_space))
q_episode_rewards = []
EPSILON = 0.3
epsilon_min = 0.001
ALPHA = 0.3
DISCOUNT = 0.95

for i in range(num_episodes):
    state = env.reset()
    done = False
    ep_reward = 0

    while not done:
        action = e_greedy_action(Q_table, state)
        next_state, reward, done, _ = env.step(action)
        ep_reward += reward
        history.append((state, action, next_state, reward, done))
        sa_visits[state][action] += 1

        td_error = reward + (not done) * DISCOUNT * max(Q_table[next_state]) - Q_table[state][
            action]
        Q_table[state][action] += ALPHA * td_error

        if EPSILON > epsilon_min:
            EPSILON *= 0.99

        state = next_state

    # print("Episode", i, "reward", ep_reward)
    q_episode_rewards.append(ep_reward)

print("Q", np.mean(q_episode_rewards[-100:], axis=0))

import matplotlib.pyplot as plt

plt.plot(episode_rewards)
plt.plot(q_episode_rewards)
plt.legend(['BuildUp', 'Q'])
plt.show()



