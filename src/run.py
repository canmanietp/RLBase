import argparse

if __name__ == "__main__":
    "example: python3 run.py --algorithms 'Q' 'QLiA' --env='office' --num_trials=2"
    parser = argparse.ArgumentParser(prog="run_experiments",
                                     description='Runs a multi-task RL experiment over a particular environment.')
    parser.add_argument('--algorithms', nargs='+', default=['Q'], type=str)
    parser.add_argument('--env', default='taxi', type=str)
    parser.add_argument('--scenario', default=None, type=int)
    parser.add_argument('--num_trials', default=1, type=int)
    parser.add_argument('--verbose', default=False, type=str)
    parser.add_argument('--render', default=False, type=str)

    args = parser.parse_args()

    # Running the experiment
    alg_names = []
    for alg in args.algorithms:
        alg_names.append(alg)
    env_name = args.env
    scenario = args.scenario
    num_trials = args.num_trials
    verbose = bool(args.verbose)
    render = bool(args.render)

    alg_strings = '\t'.join(alg_names)

    if 'DQN' in alg_strings or 'A2C' in alg_strings:
        import continuous_experiments
        continuous_experiments.run_continuous_experiment(num_trials, env_name, alg_names, scenario, verbose, render)
    else:
        import discrete_experiments
        discrete_experiments.run_discrete_experiment(num_trials, env_name, alg_names, verbose, render)
