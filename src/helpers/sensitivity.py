import random, operator
import numpy as np
import helpers.ranges


def do_sensitivity_analysis(agent, ranges, sars, state_variables, num_samples=10):
    # state variables = [0, 1, 2]
    # history = [[s0, a0, r0, s1], ...]

    state_var_sensitivity = []

    for s, a, r, ns in sars:  # range(agent.observation_space):
        var_sensitivities = [0 for v in state_variables]
        for v in state_variables:
            decoded_state = list(agent.env.decode(s))
            Qs = agent.Q_table[s]
            diff = 0.
            count_samps = 1
            tries = 0

            # samples = list(range(ranges[v][0], ranges[v][1]))# np.random.uniform(low=ranges[v][0], high=ranges[v][1]+1, size=num_samples) #
            # for samp in samples:
            while count_samps < num_samples:
                tries += 1
                if tries > 50:
                    diff = -1
                    count_samps = num_samples
                sample = random.randint(ranges[v][0], ranges[v][1])
                if sample != decoded_state[v]:
                    decoded_state[v] = sample
                    state = agent.env.encode(*decoded_state)
                    if np.sum(agent.sa_visits[state]) > 0:
                        Qs_samp = agent.Q_table[state]
                        action_sort1 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs,1), key=operator.itemgetter(1))]
                        action_sort2 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs_samp,1), key=operator.itemgetter(1))]
                        diff += np.sum(np.absolute(np.subtract(action_sort1, action_sort2))) # measure of how different the ranks are   # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  #
                        # np.argmax(Qs) == np.argmax(Qs_samp)  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                        # action_sort1 == action_sort2  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                        # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  # SMALLEST NUMBER IS LEAST INFLUENTIAL
                        count_samps += 1
            var_sensitivities.append(diff)
        state_var_sensitivity.append(var_sensitivities)
    summed = np.sum(state_var_sensitivity, axis=0)
    if all([x == 0 for x in summed]):
        return None
    return summed


def do_sensitivity_analysis_single_state(agent, ranges, s, state_variables, num_samples=5):
    # state variables = [0, 1, 2]
    # history = [[s0, a0, r0, s1], ...]

    var_sensitivities = [0 for v in state_variables]
    for iv, v in enumerate(state_variables):
        decoded_state = list(agent.env.decode(s))
        Qs = agent.Q_table[s]
        diff = 0
        count_samps = 1
        tries = 0

        # sample = list(range(ranges[v][0], ranges[v][1] + 1))  #
        # for samp in sample:
        while count_samps < num_samples:
            tries += 1
            if tries > 30:
                diff = float("-inf")
                count_samps = num_samples
            sample = random.randint(ranges[v][0], ranges[v][1])
            if sample != decoded_state[v]:
                decoded_state[v] = sample
                state = agent.env.encode(*decoded_state)
                if np.sum(agent.sa_visits[state]) > 0 and state != s:
                    Qs_samp = agent.Q_table[state]
                    # action_sort1 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs, 1), key=operator.itemgetter(1))]
                    # action_sort2 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs_samp, 1), key=operator.itemgetter(1))]
                    diff += np.argmax(Qs) == np.argmax(Qs_samp)  # measure of how different the ranks are   # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  #
                    # np.argmax(Qs) == np.argmax(Qs_samp)  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                    # action_sort1 == action_sort2  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                    # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  # SMALLEST NUMBER IS LEAST INFLUENTIAL
                    count_samps += 1
        var_sensitivities[iv] = diff
    # print(var_sensitivities)
    if any([x == float("-inf") for x in var_sensitivities]):
        return None
    return var_sensitivities
