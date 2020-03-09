import argparse

import discrete_experiments
import continuous_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithms', nargs='+', default=['Q','QLiA'], type=str)
    parser.add_argument('--env', default='taxi', type=str)
    parser.add_argument('--num_trials', default=1, type=int)
    parser.add_argument('--verbose', default=False, type=bool)

    args = parser.parse_args()

    # Running the experiment
    alg_names = []
    for alg in args.algorithms:
        alg_names.append(alg)
    env_name = args.env
    num_trials = args.num_trials
    verbose = args.verbose

    if 'Q' in alg_names:
        discrete_experiments.run_discrete_experiment(num_trials, env_name, alg_names, verbose)
    elif 'DQN' in alg_names:
        continuous_experiments.run_continuous_experiment(num_trials, env_name, alg_names, verbose)
