import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import time, os, datetime

from envs.taxi import TaxiEnv
from envs.taxilarge import TaxiLargeEnv
from envs.taxifuel import TaxiFuelEnv
from envs.taxi_stochastic import TaxiStochasticEnv
from envs.noisy_taxi import NoisyTaxiEnv
from envs.eatfood import EatFoodEnv
from envs.rm_office import OfficeEnv
from envs.coffeemail import CoffeeMailEnv
from envs.coffee import CoffeeEnv
from envs.sysadmin import SysAdminEnv
from envs.fourstate import FourStateEnv
from envs.warehouse import WarehouseEnv

from agents.Q import QAgent
from agents.QLiA import QLiAAgent
from agents.MaxQ import MaxQAgent
from agents.LOARA_known import LOARA_K_Agent
from agents.LOARA_unknown import LOARA_UK_Agent
from agents.LOARA_transfer import LOARA_T_Agent
from agents.MergeTree import MergeTree_Agent
# from agents.QLiA_alt import QLiA_altAgent
# from agents.QLiA_batch import QLiA_batchAgent
# from agents.QLiA_sig import QLiA_sigAgent
# from agents.QAMS import QAMSAgent
# from agents.QBias import QBiasAgent
# from agents.L2Q import L2QAgent
# from agents.Q_sens import QSensAgent
# from agents.QVP import QVPAgent
from agents.QVA import QVAAgent
from learning_parameters import DiscreteParameters
from helpers import plotting
from helpers import sensitivity


def get_params_warehouse(alg, size):
    init_alpha = 1.
    alpha_min = 1.
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.99
    decay_rate = 0.999
    sub_spaces = None
    options = None
    if size == 1:
        if alg in ['LOARA_unknown', 'LOARA_known', 'QVA']:
            sub_spaces = [[0, 1], [0, 2], [1, 2], [0, 1, 2]]
        size_state_vars = [3, 3, 3]
        num_episodes = 5000
    elif size == 2:
        if alg in ['LOARA_known', 'QVA']:
            sub_spaces = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2, 3, 4], [0, 1, 2, 3, 4]]
        if alg in ['LOARA_unknown']:
            sub_spaces = [[1, 2, 3, 4], [0, 1, 2, 3, 4]]  # [0, 1], [0, 2], [0, 3], [0, 4],
        size_state_vars = [5, 3, 3, 3, 3]
        num_episodes = 10000
    elif size == 3:
        if alg in ['LOARA_unknown', 'LOARA_known', 'QVA']:
            sub_spaces = [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]
        size_state_vars = [7, 3, 3, 3, 3, 3, 3]
        num_episodes = 100000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_fourstate(alg):
    init_alpha = 0.5
    alpha_min = 0.01
    init_epsilon = 0.5
    epsilon_min = 0.0
    init_phi = 0.5
    phi_min = 0.0
    discount = 0.9
    decay_rate = 0.995
    sub_spaces = None
    options = None
    if alg in ['LOARA_unknown']:
        sub_spaces = [[0], [1], [0, 1]]  # [2], [1, 2, 3], [0, 2, 3],
    elif alg in ['LOARA_known', 'QVA']:
        sub_spaces = [[0], [1], [0, 1]]  #
    size_state_vars = [2, 2]
    num_episodes = 1000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_coffee():
    init_alpha = 0.3
    alpha_min = 0.1
    init_epsilon = 0.3
    epsilon_min = 0.001
    init_phi = 0.3
    phi_min = 0.001
    discount = 0.99
    decay_rate = 0.99
    sub_spaces = [[0, 1, 2], [0, 1, 2, 3, 4]]
    size_state_vars = [5, 5, 2, 2, 2]
    num_episodes = 400
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_coffeemail(alg):
    init_alpha = 1.
    alpha_min = 1.
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.99
    decay_rate = 0.999
    options = None
    sub_spaces = []
    if alg in ['QLiA', 'QLiA_alt', 'LOARA_known']:
        sub_spaces = [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]
    elif alg in ['LOARA_unknown']:
        sub_spaces = [[0, 1, 2], [0, 1, 3], [0, 1, 2, 3, 4, 5, 6, 7]]
    elif alg in ['MaxQ']:
        # 0 south, 1 north, 2 east, 3 west, 4 take, 5 give, 6 do nothing,
        # 7 go to mail, 8 go to A, 9 go to B, 10 go to coffee,
        # 11 deliver mail to A, 12, deliver mail to B, 13 deliver coffee to A, 14 deliver coffee to B
        # 15 get mail, 16 get coffee, # 17 get and deliver mail, 18 get and deliver coffee, 19 root
        sub_spaces = []
        options = [set(), set(), set(), set(), set(), set(), set(),
                   {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3},
                   {8, 5}, {9, 5}, {8, 5}, {9, 5},
                   {7, 4}, {10, 4}, {15, 11, 12}, {16, 13, 14}, {17, 18, 6}, ]
    size_state_vars = [7, 7, 2, 2, 2, 2, 2, 2]
    num_episodes = 1500
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_office(alg):
    init_alpha = 0.5
    alpha_min = 0.05
    init_epsilon = 0.5
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.95
    decay_rate = 0.995
    if alg in ['QAMS', 'QLiA_T']:
        sub_spaces = [[0, 1, 2, 3, 4, 5]]
    elif alg in ['QLiA', 'QVA', 'QLiA_alt', 'LOARA_unknown']:
        init_alpha = 0.5
        alpha_min = 0.1
        sub_spaces = [[0, 1], [0, 1, 2, 3], [0, 1, 4, 5],
                      [0, 1, 2, 3, 4, 5]]  # [[1, 2, 3, 4, 5], [0, 2, 3, 4, 5], [0, 1, 2, 4, 5]]  #
    elif alg == 'QVP':
        sub_spaces = [[0, 1, 2, 4, 5], [0, 1, 3, 4, 5]]
    else:
        sub_spaces = []
    size_state_vars = [9, 12, 2, 2, 2, 2]
    num_episodes = 3000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_eatfood(alg):
    init_alpha = 0.5
    alpha_min = 0.1
    init_epsilon = 0.3
    epsilon_min = 0.1
    init_phi = 0.3
    phi_min = 0.1
    discount = 0.95
    decay_rate = 0.999
    if alg in ['QAMS', 'QLiA_T']:
        sub_spaces = [[0, 1, 2, 3]]
    elif alg in ['QLiA', 'QLiA_alt', 'LOARA_unknown']:
        sub_spaces = [[0, 1, 3], [0, 1, 2, 3]]
    else:
        sub_spaces = []
    size_state_vars = [5, 5, 25, 25]
    num_episodes = 300000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars)


def get_params_taxifuel(alg):
    init_alpha = 1.
    alpha_min = 1.
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.99
    decay_rate = 0.99
    sub_spaces = None
    options = None
    # if alg in ['Q']:
    #     alpha_min = 0.3
    if alg in ['LOARA_unknown']:
        sub_spaces = [[0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 3, 4]]
    if alg in ['QLiA', 'LOARA_known', 'QVA']:
        sub_spaces = [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 2, 3, 4]]
    elif alg == 'MaxQ':
        # alpha_min = 0.3
        # south 0, north 1, east 2, west 3, pickup 4, dropoff 5, fillup 6, gotoSource 7, gotoDestination 8, gotoFuel 9,
        # get 10, put 11, refuel 12, root 13
        options = [set(), set(), set(), set(), set(), set(), set(), {0, 1, 2, 3}, {0, 1, 2, 3}, {0, 1, 2, 3},
                   {4, 7}, {5, 8}, {6, 9}, {11, 10, 12}, ]
    size_state_vars = [5, 5, 5, 4, 14]
    num_episodes = 50000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_taxilarge(alg):
    init_alpha = 1.
    alpha_min = 1.
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.99
    decay_rate = 0.995
    sub_spaces = None
    options = None
    if alg in ['QAMS', 'QLiA', 'QVA', 'QLiA_alt', 'LOARA_known']:
        sub_spaces = [[0, 1, 2], [1, 2, 3], [0, 2, 3], [0, 1, 2, 3]]  # [[1, 2, 3], [0, 2, 3], [0, 1, 2, 3]]  #
    elif alg in ['QLiA_T', 'LOARA_unknown', 'LOARA_transfer']:
        sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    elif alg == 'QVP':
        sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    elif alg == 'MaxQ':
        # south, north, east, west, pickup, droppoff, gotoSource, gotoDestination, get, put, root
        options = [set(), set(), set(), set(), set(), set(), {0, 1, 2, 3}, {0, 1, 2, 3}, {4, 6}, {5, 7}, {8, 9}, ]
    size_state_vars = [10, 10, 9, 8]
    num_episodes = 2000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_noisytaxi(alg):
    init_alpha = 1.
    alpha_min = 1.
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.95
    decay_rate = 0.999
    sub_spaces = []
    options = None
    if alg in ['QAMS', 'QLiA', 'QLiA_sig', 'LOARA_unknown']:
        sub_spaces = [[1, 2, 3], [0, 2, 3], [0, 1, 2], [0, 1, 2, 3]]  # [2], [1, 2, 3], [0, 2, 3],
    elif alg in ['LOARA_known', 'QLiA_alt' 'QLiA', 'QLiA_batch', 'QVA']:
        sub_spaces = [[2], [1, 2, 3], [0, 2, 3], [0, 1, 2], [0, 1, 2, 3]]  #
    elif alg == 'QVP':
        sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    elif alg == 'MaxQ':
        # south, north, east, west, pickup, droppoff, gotoSource, gotoDestination, get, put, root
        options = [set(), set(), set(), set(), set(), set(), {0, 1, 2, 3}, {0, 1, 2, 3}, {4, 6}, {5, 7}, {8, 9}, ]
    size_state_vars = [5, 5, 5, 4, 5]
    num_episodes = 4000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_taxi(alg):
    init_alpha = 1.
    alpha_min = 1.
    init_epsilon = 0.5
    epsilon_min = 0.01
    init_phi = 0.5
    phi_min = 0.01
    discount = 0.95
    decay_rate = 0.99
    sub_spaces = None
    options = None
    if alg in ['QAMS', 'QLiA', 'QLiA_sig','LOARA_unknown', 'LOARA_transfer']:
        sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]  # [2], [1, 2, 3], [0, 2, 3], [0, 1], [0, 2], [1, 2],
    elif alg in ['LOARA_known', 'QLiA_alt' 'QLiA', 'QLiA_batch', 'QVA']:
        sub_spaces = [[2], [1, 2, 3], [0, 2, 3], [0, 1, 2], [0, 1, 2, 3]]  #
    elif alg == 'MaxQ':
        # south, north, east, west, pickup, droppoff, gotoSource, gotoDestination, get, put, root
        options = [set(), set(), set(), set(), set(), set(), {0, 1, 2, 3}, {0, 1, 2, 3}, {4, 6}, {5, 7}, {8, 9}, ]
    size_state_vars = [5, 5, 5, 4]
    num_episodes = 600
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params_sysadmin(alg):
    init_alpha = 0.1
    alpha_min = 0.1
    init_epsilon = 0.5
    epsilon_min = 0.001
    init_phi = 0.5
    phi_min = 0.001
    discount = 0.95
    decay_rate = 0.99
    sub_spaces = []
    options = None
    if alg in ['QAMS', 'QLiA', 'QLiA_batch']:
        sub_spaces = [[4, 5, 6, 7, 0, 1, 2], [2, 3, 4, 5, 6, 7, 0]]
    # elif alg == 'QLiA_T':
    #     sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    # elif alg == 'QVP':
    #     sub_spaces = [[0, 1, 2], [0, 1, 2, 3]]
    # elif alg == 'MaxQ':
    #     # south, north, east, west, pickup, droppoff, gotoSource, gotoDestination, get, put, root
    #     options = [set(), set(), set(), set(), set(), set(), {0, 1, 2, 3}, {0, 1, 2, 3}, {4, 6}, {5, 7}, {8, 9}, ]
    # elif alg == 'Q':
    #     init_epsilon = 0.2
    #     epsilon_min = 0.01
    size_state_vars = [2, 2, 2, 2, 2, 2, 2, 2]
    num_episodes = 2000
    return DiscreteParameters(alpha=init_alpha, alpha_min=alpha_min, epsilon=init_epsilon, epsilon_min=epsilon_min,
                              discount=discount, decay=decay_rate, num_episodes=num_episodes, phi=init_phi,
                              phi_min=phi_min, sub_spaces=sub_spaces, size_state_vars=size_state_vars, options=options)


def get_params(env_name, alg=None):
    if env_name == 'taxi':
        env = TaxiEnv()  # TaxiStochasticEnv() #
        params = get_params_taxi(alg)
    elif env_name == 'taxilarge':
        env = TaxiLargeEnv()
        params = get_params_taxilarge(alg)
    elif env_name == 'taxifuel':
        env = TaxiFuelEnv()
        params = get_params_taxifuel(alg)
    elif env_name == 'noisytaxi':
        env = TaxiStochasticEnv()  # NoisyTaxiEnv()
        params = get_params_noisytaxi(alg)
    elif env_name == 'eatfood':
        env = EatFoodEnv()
        params = get_params_eatfood(alg)
    elif env_name == 'office':
        env = OfficeEnv()
        params = get_params_office(alg)
    elif env_name == 'coffeemail':
        env = CoffeeMailEnv()
        params = get_params_coffeemail(alg)
    elif env_name == 'coffee':
        env = CoffeeEnv()
        params = get_params_coffee()
    elif env_name == 'fourstate':
        env = FourStateEnv()
        params = get_params_fourstate(alg)
    elif env_name == 'warehouse1':
        env = WarehouseEnv(1)
        params = get_params_warehouse(alg, 1)
    elif env_name == 'warehouse2':
        env = WarehouseEnv(2)
        params = get_params_warehouse(alg, 2)
    elif env_name == 'warehouse3':
        env = WarehouseEnv(3)
        params = get_params_warehouse(alg, 3)
    elif env_name == 'sysadmin':
        env = SysAdminEnv(size=8)
        params = get_params_sysadmin(alg)
    else:
        print("Error: Unknown environment")
        return
    return env, params


def run_discrete_experiment(num_trials, env_name, algs, verbose=False, render=False):
    date_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_dir = "tmp/{}_{}".format(date_string, env_name)
    os.mkdir(exp_dir)
    env, params = [], []

    trial_rewards = []
    trial_times = []

    for t in range(num_trials):
        agents = []
        for alg in algs:
            env, params = get_params(env_name, alg)
            if alg == 'Q':
                agents.append(QAgent(env, params))
            elif alg == 'QLiA':
                agents.append(QLiAAgent(env, params))
            elif alg == 'QVA':
                agents.append(QVAAgent(env, params))
            elif alg == 'MaxQ':
                agents.append(MaxQAgent(env, params))
            elif alg == 'LOARA_known':
                agents.append(LOARA_K_Agent(env, params))
            elif alg == 'LOARA_unknown':
                agents.append(LOARA_UK_Agent(env, params))
            elif alg == 'LOARA_transfer':
                env, params = get_params(env_name, 'LOARA_unknown')
                agents.append(LOARA_T_Agent(env, params))
            elif alg == 'MergeTree':
                env, params = get_params(env_name, 'LOARA_unknown')
                agents.append(MergeTree_Agent(env, params))
            else:
                print("Unknown algorithm - {}".format(alg))

        episode_rewards = [[] for q in range(len(agents))]
        starting_states = []
        decoded_ss = []
        times_to_run = []

        plt.figure()

        Qbest = []

        print("{} -- Starting Trial {} -- ".format(datetime.datetime.now().strftime("%H:%M:%S"), t + 1))
        for j, agent in enumerate(agents):
            t0 = time.time()
            print("{} Running agent: {}".format(datetime.datetime.now().strftime("%H:%M:%S"), agent.name))
            for i in range(params.num_episodes):
                agent.reset()
                if 'AtariARIWrapper' not in str(agent.env):
                    if 'Noisy' in str(agent.env):
                        agent.env = NoisyTaxiEnv()
                        agent.env.reset()

                    if j == 0:
                        state = agent.current_state
                        # state_vars = list(agent.env.decode(state))
                        # if i < 300:
                        #     while state_vars[2] == 3:
                        #         agent.env.reset()
                        #         state = agent.env.s
                        #         state_vars = list(agent.env.decode(state))
                        #     agent.current_state = state
                        starting_states.append(state)
                        decoded_ss.append(list(env.decode(state)))
                    else:
                        state = starting_states[i]  #
                        agent.set_state(state)
                ep_reward = agent.do_episode()
                episode_rewards[j].append(ep_reward)

                if verbose:
                    print("{} Episode {}, episode reward={}, Last 100 average={}".format(
                        datetime.datetime.now().strftime("%H:%M:%S"), i, ep_reward, np.mean(episode_rewards[j][-100:])))

                if agent.name != 'QLiA_batch':
                    agent.decay(agent.params.DECAY_RATE)

            # Q_save = []
            #
            # if agent.name == 'Q':
            #     for ps in range(10000):
            #         sos = agent.env.reset()
            #         sos_decode = list(agent.env.decode(sos))
            #         if np.sum(agent.sa_visits[sos]) > 0 and sos_decode[2] != sos_decode[3]:
            #             best_act = np.argmax(agent.Q_table[sos])
            #             Q_save.append([*list(agent.env.decode(sos)), best_act])
                # p = 0
                # while p < 1000:
                #     done = False
                #     agent.env.reset()
                #     sos = agent.env.s
                #     timeout = 0
                #     while not done:
                #         best_act = np.argmax(agent.Q_table[sos])
                #         ns, reward, done, info = agent.env.step(best_act)
                #         if timeout > 25:
                #             done = True
                #         # print(p, timeout, [*list(agent.env.decode(sos)), best_act], done)
                #         sos_decode = list(agent.env.decode(sos))
                #         if np.sum(agent.sa_visits[sos]) > 0 and sos_decode[2] != sos_decode[3]:
                #             Q_save.append([*list(agent.env.decode(sos)), best_act])
                #         timeout += 1
                #         if done:
                #             p += 1
                #         sos = ns

            # Qs = pd.DataFrame(Q_save)
            # Qs.to_csv('{}/Q_samples_{}.csv'.format(exp_dir, str(agent.env)), header=None, index=None)

            # # # # # elif agent.name in 'LOARA_unknown':
            # # # # #     for s in range(agent.observation_space):
            # # # # #         print(s, agent.state_bandit_map[s][-1].Q_table)
            # elif agent.name in 'LOARA_known':
            #     for s in range(agent.observation_space):
            #         # if list(agent.env.decode(s)) == [4, 1, 1, 0] or list(agent.env.decode(s)) == [4, 1, 1, 2] or list(agent.env.decode(s)) == [4, 1, 1, 3]:
            #         #     print(list(agent.env.decode(s)),
            #         #           agent.params.sub_spaces[agent.heuristic_bandit_choice(s)],
            #         #           agent.state_bandit_map[s][agent.heuristic_bandit_choice(s)].Q_table,
            #         #           np.argmax(agent.state_bandit_map[s][agent.heuristic_bandit_choice(s)].Q_table) == np.argmax(Qbest[s]),
            #         #           Qbest[s])  # agent.heuristic_bandit_choice(s)
            #         # if list(agent.env.decode(s)) == [0, 1, 0, 0, 0] or list(agent.env.decode(s)) == [1, 1, 0, 0, 0] or list(agent.env.decode(s)) == [2, 1, 0, 0, 0]  or list(agent.env.decode(s)) == [3, 1, 0, 0, 0] or list(agent.env.decode(s)) == [4, 1, 0, 0, 0] :
            #         if np.argmax(agent.state_bandit_map[s][agent.heuristic_bandit_choice(s)].Q_table) != np.argmax(Qbest[s]):
            #             print(list(agent.env.decode(s)),
            #                   agent.params.sub_spaces[agent.heuristic_bandit_choice(s)],
            #                   agent.state_bandit_map[s][-1].Q_table,
            #                   np.argmax(agent.calc_value_abstract_state(s, agent.heuristic_bandit_choice(s))) == np.argmax(Qbest[s]),
            #                   Qbest[s])  # agent.heuristic_bandit_choice(s)

            run_time = time.time() - t0
            print("{} Finished running in {} seconds".format(datetime.datetime.now().strftime("%H:%M:%S"), run_time))
            times_to_run.append(run_time)

            plt.plot(plotting.moving_average(episode_rewards[j], int(params.num_episodes / 10)), label=agent.name)
            plt.legend([a.name for a in agents], loc='lower right')
            plt.savefig('{}/trial_{}'.format(exp_dir, t + 1))

            if t == num_trials - 1 and agent.name == 'LOARA_unknown':
                learned_Q = []
                learned_bandit = []
                for s in range(agent.observation_space):
                    qs = []
                    for b in agent.state_bandit_map[s]:
                        for q in b.Q_table:
                            qs.append(q)
                    learned_Q.append(qs)
                    band, val = agent.smart_bandit_choice(s, 0.)
                    learned_bandit.append(band)
                lq = pd.DataFrame(learned_Q)
                lb = pd.DataFrame(learned_bandit)
                lb.to_csv('{}/learned_bandit_{}.csv'.format(exp_dir, str(agent.env)), header=None, index=None)
                lq.to_csv('{}/learned_bandit_Q_{}.csv'.format(exp_dir, str(agent.env)), header=None, index=None)

        trial_rewards.append(episode_rewards)
        trial_times.append(times_to_run)

        df = pd.DataFrame(np.transpose(episode_rewards))
        df['Starting states'] = decoded_ss
        df.to_csv('{}/trial_{}_rewards.csv'.format(exp_dir, t + 1), header=None, index=None)

    plt.close()
    plt.figure()
    stds = np.std(trial_rewards, axis=0)
    for it, trial in enumerate(np.average(trial_rewards, axis=0)):
        std = stds[it]
        c = 1.98 * np.divide(std, trial, out=np.zeros_like(std), where=trial != 0)
        y = plotting.moving_average(trial, int(params.num_episodes / 10))
        ci = plotting.moving_average(c, int(params.num_episodes / 10))
        x = range(len(y))
        plt.plot(x, y)
        plt.fill_between(x, (y - ci), (y + ci), alpha=.1)

    plt.legend([a for a in algs], loc='lower right')
    plt.savefig('{}/final'.format(exp_dir))

    plt.figure()
    for trial in np.average(trial_rewards, axis=0):
        plt.plot(np.cumsum(trial))

    plt.legend([a for a in algs], loc='lower right')
    plt.savefig('{}/cumsum'.format(exp_dir))
    plt.close()

    for ia, alg in enumerate(algs):
        env, params = get_params(env_name, alg)
        file = open('{}/params_agent{}.txt'.format(exp_dir, alg), "w")
        file.write("Environment: {}\n"
                   "Algorithm : {}\n"
                   "Position: {}\n"
                   "Number of trials: {}\n"
                   "Number of episodes: {}\n"
                   "Running times: {}\n"
                   "init_alpha={}\n"
                   "alpha_min={}\n"
                   "init_epsilon={}\n"
                   "epsilon_min={}\n"
                   "init_phi={}\n"
                   "phi_min={}\n"
                   "discount={}\n"
                   "decay={}\n"
                   "sub_spaces={}\n"
                   "options={}\n"
                   "size_state_vars={}".format(env, alg, ia, num_trials,
                                               params.num_episodes, np.array(trial_times)[:, ia], params.ALPHA,
                                               params.ALPHA_MIN,
                                               params.EPSILON, params.EPSILON_MIN,
                                               params.PHI, params.PHI_MIN, params.DISCOUNT, params.DECAY_RATE,
                                               params.sub_spaces, params.options, params.size_state_vars))
        file.close()
    return
