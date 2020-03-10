from gym.envs.registration import register

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)