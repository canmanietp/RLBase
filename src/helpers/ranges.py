def get_ranges(memory, observation_space):
    min_max = [[float("inf"), float("-inf")] for v in list(range(observation_space))]
    for s, _, _, _, _ in memory:
        for iv, var in enumerate(s[0]):
            if var < min_max[iv][0]:
                min_max[iv][0] = var
            if var > min_max[iv][1]:
                min_max[iv][1] = var
    return min_max


def get_var_ranges(agent, history, state_vars):
    ranges = {}

    for v in state_vars:
        min = float("inf")
        max = float("-inf")
        for s in range(agent.observation_space):
            decoded = list(agent.env.decode(s))
            if decoded[v] < min:
                min = decoded[v]
            if decoded[v] > max:
                max = decoded[v]

        ranges[v] = [min, max]

    return ranges

