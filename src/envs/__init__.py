from gym.envs.registration import registry, register, make, spec

register(
    id='CartPole-v100',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)