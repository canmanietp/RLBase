import gym
import numpy as np
from envs.atariari.benchmark.wrapper import AtariARIWrapper
import math

num_meta_states = 2000
num_actions = 6
memory_averages = []

def info_into_state(info, abstraction):
    state = []
    if abstraction is None:
        for i, lab in enumerate(info['labels']):
            state.append(info['labels'][lab])
    else:
        for i, lab in enumerate(info['labels']):
            if i in abstraction:
                state.append(info['labels'][lab])
    return np.array(state)


# def calculate_memory_interval_averages(memory):
#     step = int(len(memory) / num_meta_states)
#     for i, _ in enumerate(memory[::step]):
#         sub_list = memory[i * step:] if (i + 1) * step > len(memory) else memory[i * step:(i + 1) * step]
#         memory_averages.append(np.average(sub_list, axis=0))


# def state_into_metastate(state, memory):
#     if memory_averages == []:
#         calculate_memory_interval_averages(memory)
#     lowest_distance = float("inf")
#     meta_state = -1
#     for im, meme in enumerate(memory_averages):
#         distance = np.linalg.norm(meme - state)
#         if distance < lowest_distance:
#             meta_state = im
#             lowest_distance = distance
#
#     return meta_state


def state_into_metastate(state, memory):
    lowest_distance = float("inf")
    meta_state = -1
    divisions = int(len(memory) / num_meta_states)
    distance = float("inf")
    for im, meme in enumerate(memory):
        if im % divisions != 0:
            distance += np.linalg.norm(meme - state)
        elif im != 0:
            if distance < lowest_distance:
                meta_state = int(im/divisions) - 1
                lowest_distance = distance
            distance = 0

    return meta_state


env = AtariARIWrapper(gym.make('Pong-v0'))
env.reset()
e = 0
memory = []
done = False
random_action = np.random.randint(num_actions)

while e < 20:
    next_state, reward, done, next_state_info = env.step(random_action)
    current_state = info_into_state(next_state_info, None)
    memory.append(current_state)
    if e % 4 == 0:
        random_action = np.random.randint(num_actions)
    if done:
        e += 1
        done = False
        env.reset()

print("finished collecting memory")

# env.reset()
#
# done = False
# i = 0
#
# while not done:
#     next_state, reward, done, next_state_info = env.step(np.random.randint(0, 6))
#     current_state = info_into_state(next_state_info, None)
#     print(i, current_state, state_into_metastate(current_state, memory))
#     i += 1

Q_table = np.zeros([num_meta_states*2, num_actions])
num_abstractions = 2
Q_LIA_table = np.zeros([num_meta_states*2, num_abstractions])
Q_you_ballx = np.zeros([255*255, num_actions])
Q_you_bally = np.zeros([255*255, num_actions])
# Q_them_ballx = np.zeros([255*255, num_actions])
# Q_them_bally = np.zeros([255*255, num_actions])
epsilon = 0.3
epsilon_min = 0.01
alpha = 0.01
gamma = 0.999

# for i in range(10000):
#     env.reset()
#     episode_reward = 0
#     done = False
#
#     _, _, _, info = env.step(np.random.randint(num_actions))
#     state = info_into_state(info, None)
#     meta_state = state_into_metastate(state, memory)
#
#     while not done:
#         if np.random.uniform(0, 1) < epsilon:
#             action = np.random.randint(num_actions)
#         else:
#             action = np.argmax(Q_table[meta_state])
#
#         next_obs, reward, done, next_state_info = env.step(action)
#         env.render()
#         episode_reward += reward
#         next_state = info_into_state(next_state_info, None)
#         next_meta_state = state_into_metastate(next_state, memory)
#         print(i, epsilon, state, meta_state, action, reward)
#
#         Q_table[meta_state][action] += alpha * (reward + gamma * max(Q_table[next_meta_state]) - Q_table[meta_state][action])
#
#         state = next_state
#         meta_state = next_meta_state
#
#     epsilon = epsilon * 0.99
#
#     print("End of episode", i, "reward:", episode_reward)

for i in range(10000):
    env.reset()
    episode_reward = 0
    done = False

    _, _, _, info = env.step(np.random.randint(num_actions))
    state = info_into_state(info, None)
    meta_state = state_into_metastate(state, memory)

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            abstraction = np.random.randint(num_abstractions)
        else:
            abstraction = np.argmax(Q_LIA_table[meta_state])

        you_ballx_state = state[0]*255 + state[4]
        you_bally_state = state[0]*255 + state[5]
        # them_ballx_state = state[2]*255 + state[4]
        # them_bally_state = state[2]*255 + state[5]

        action = -1

        if abstraction == 0:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q_you_ballx[you_ballx_state])
        elif abstraction == 1:
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q_you_bally[you_bally_state])
        # elif abstraction == 2:
        #     if np.random.uniform(0, 1) < epsilon:
        #         action = np.random.randint(num_actions)
        #     else:
        #         action = np.argmax(Q_them_ballx[them_ballx_state])
        # elif abstraction ==3:
        #     if np.random.uniform(0, 1) < epsilon:
        #         action = np.random.randint(num_actions)
        #     else:
        #         action = np.argmax(Q_them_bally[them_bally_state])

        next_obs, reward, done, next_state_info = env.step(action)
        # env.render()
        episode_reward += reward
        next_state = info_into_state(next_state_info, None)
        if reward!= 0:
            print("Episode", i, "score:", next_state[6], next_state[7])
        next_meta_state = state_into_metastate(next_state, memory)
        # print(i, epsilon, state, meta_state, action, reward)

        Q_LIA_table[meta_state][abstraction] += alpha * (reward + gamma * max(Q_table[next_meta_state]) - Q_table[meta_state][abstraction])
        you_ballx_next_state = next_state[0]*255 + next_state[4]
        you_bally_next_state = next_state[0]*255 + next_state[5]
        them_ballx_next_state = next_state[2]*255 + next_state[4]
        them_bally_next_state = next_state[2]*255 + next_state[5]
        Q_you_ballx[you_ballx_state] += alpha * (reward + gamma * max(Q_you_ballx[you_ballx_next_state]) - Q_you_ballx[you_ballx_state][action])
        Q_you_bally[you_bally_state] += alpha * (reward + gamma * max(Q_you_bally[you_bally_next_state]) - Q_you_bally[you_bally_state][action])
        # Q_them_ballx[them_ballx_state] += alpha * (reward + gamma * max(Q_them_ballx[them_ballx_next_state]) - Q_them_ballx[them_ballx_state][action])
        # Q_them_bally[them_bally_state] += alpha * (reward + gamma * max(Q_them_bally[them_bally_next_state]) - Q_them_bally[them_bally_state][action])

        state = next_state
        meta_state = next_meta_state

    if epsilon > epsilon_min:
        epsilon = epsilon * 0.99

    print("End of episode", i, "reward:", episode_reward, "epsilon", epsilon)






