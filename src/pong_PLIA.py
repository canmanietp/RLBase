import gym
import numpy as np
from envs.atariari.benchmark.wrapper import AtariARIWrapper
import math

num_meta_states = 9998
num_actions = 6

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


def state_into_metastate(state, memory):
    lowest_distance = float("inf")
    meta_state = -1
    divisions = math.floor(len(memory) / num_meta_states)
    distance = float("inf")
    for im, meme in enumerate(memory):
        if im % divisions != 0:
            distance += np.linalg.norm(meme - state)
        else:
            if distance < lowest_distance:
                meta_state = math.floor(im/divisions)
                lowest_distance = distance
            distance = 0

    return meta_state


env = AtariARIWrapper(gym.make('Pong-v0'))
env.reset()
e = 0
memory = []
done = False

while e < 100000:
    next_state, reward, done, next_state_info = env.step(np.random.randint(num_actions))
    current_state = info_into_state(next_state_info, None)
    memory.append(current_state)
    if done:
        e += 1
        done = False

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

Q_table = np.zeros([num_meta_states+2, num_actions])
epsilon = 0.3
alpha = 0.001
gamma = 0.999

for i in range(10000):
    env.reset()
    episode_reward = 0
    done = False

    _, _, _, info = env.step(np.random.randint(num_actions))
    state = info_into_state(info, None)
    meta_state = state_into_metastate(state, memory)

    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.randint(num_actions)
        else:
            action = np.argmax(Q_table[meta_state])

        next_obs, reward, done, next_state_info = env.step(action)
        # env.render()
        episode_reward += reward
        next_state = info_into_state(next_state_info, None)
        next_meta_state = state_into_metastate(next_state, memory)
        print(i, epsilon, state, meta_state, action, reward)

        Q_table[meta_state][action] += alpha * (reward + gamma * max(Q_table[next_meta_state]) - Q_table[meta_state][action])

        meta_state = next_meta_state

    epsilon = epsilon * 0.99

    print("End of episode", i, "reward:", episode_reward)






