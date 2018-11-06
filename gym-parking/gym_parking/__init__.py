from gym.envs.registration import register

register(
    id='parking-v0',
    entry_point='gym_parking.envs:ParkingEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)