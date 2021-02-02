from gym.envs.registration import register

register(
    id='carla-v0',
    entry_point='gym_carla.env:CarlaEnv',
)
