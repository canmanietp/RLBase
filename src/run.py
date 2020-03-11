import argparse

if __name__ == "__main__":
    "example: python3 run.py --algorithms 'Q' 'QLiA' --env='office' --num_trials=2"
    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithms', nargs='+', default=['Q', 'QIiB'], type=str)
    parser.add_argument('--env', default='office', type=str)
    parser.add_argument('--num_trials', default=2, type=int)
    parser.add_argument('--verbose', default=False, type=str)

    args = parser.parse_args()

    # Running the experiment
    alg_names = []
    for alg in args.algorithms:
        alg_names.append(alg)
    env_name = args.env
    num_trials = args.num_trials
    verbose = bool(args.verbose)

    if 'Q' in alg_names:
        import discrete_experiments
        discrete_experiments.run_discrete_experiment(num_trials, env_name, alg_names, verbose)
    elif 'DQN' in alg_names:
        import continuous_experiments
        continuous_experiments.run_continuous_experiment(num_trials, env_name, alg_names, verbose)
    else:
        print("Error: Must include a baseline algorithm (such as Q or DQN).")
