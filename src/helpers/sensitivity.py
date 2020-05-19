import random, operator
import numpy as np
import helpers.ranges


def do_sensitivity_analysis(agent, ranges, sars, state_variables, num_samples=15):
    # state variables = [0, 1, 2]
    # history = [[s0, a0, r0, s1], ...]

    state_var_sensitivity = {}

    for s, a, r, ns in sars:  # range(agent.observation_space):
        if s not in state_var_sensitivity:
            var_sensitivities = []
            decoded_state = list(agent.env.decode(s))
            for v in state_variables:
                samples = np.random.uniform(low=ranges[v][0], high=ranges[v][1]+1, size=num_samples)
                Qs = agent.Q_table[s]
                diff = 0.
                count_samps = 1

                for samp in samples:
                    samp = int(samp)
                    if samp != decoded_state[v]:
                        decoded_state[v] = samp
                        state = agent.env.encode(*decoded_state)
                        Qs_samp = agent.Q_table[state]
                        action_sort1 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs,1), key=operator.itemgetter(1))]
                        action_sort2 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs_samp,1), key=operator.itemgetter(1))]
                        diff += np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  # measure of how different the ranks are   # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  #
                        # np.argmax(Qs) == np.argmax(Qs_samp)  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                        # action_sort1 == action_sort2  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                        # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  # SMALLEST NUMBER IS LEAST INFLUENTIAL
                        count_samps += 1
                var_sensitivities.append(diff / count_samps)
            state_var_sensitivity[s] = var_sensitivities

    return state_var_sensitivity

def do_sensitivity_analysis_single_state(agent, sars, s, state_variables, num_samples=25):
    # state variables = [0, 1, 2]
    # history = [[s0, a0, r0, s1], ...]

    ranges = helpers.ranges.get_var_ranges(agent, sars, state_variables)

    var_sensitivities = []
    decoded_state = list(agent.env.decode(s))
    for v in state_variables:
        samples = np.random.uniform(low=ranges[v][0], high=ranges[v][1]+1, size=num_samples)
        Qs = agent.Q_table[s]
        diff = 0.
        count_samps = 1
        for samp in samples:
            samp = int(samp)
            if samp != decoded_state[v]:
                decoded_state[v] = samp
                state = agent.env.encode(*decoded_state)
                Qs_samp = agent.Q_table[state]
                    # action_sort1 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs,1), key=operator.itemgetter(1))]
                    # action_sort2 = [operator.itemgetter(0)(t) for t in sorted(enumerate(Qs_samp,1), key=operator.itemgetter(1))]
                diff += np.argmax(Qs) == np.argmax(Qs_samp)  # measure of how different the ranks are   # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  #
                    # np.argmax(Qs) == np.argmax(Qs_samp)  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                    # action_sort1 == action_sort2  # BIGGEST NUMBER IS LEAST INFLUENTIAL
                    # np.sum(np.absolute(np.subtract(action_sort1, action_sort2)))  # SMALLEST NUMBER IS LEAST INFLUENTIAL
                count_samps += 1
        var_sensitivities.append(diff / count_samps)
    return var_sensitivities
