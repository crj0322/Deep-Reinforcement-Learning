import gym
import gym_parking
from test import random_walk

def run_baseline_ddpg(env_name, train=True):
    import numpy as np
    # from stable_baselines.ddpg.policies import MlpPolicy
    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise
    from stable_baselines import DDPG

    env = gym.make(env_name)
    env = DummyVecEnv([lambda: env])

    if train:
        # mlp
        from stable_baselines.ddpg.policies import FeedForwardPolicy
        class CustomPolicy(FeedForwardPolicy):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                layers=[64, 64, 64],
                                                layer_norm=True,
                                                feature_extraction="mlp")

        # the noise objects for DDPG
        n_actions = env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions)+0.15, sigma=0.3 * np.ones(n_actions))
        model = DDPG(CustomPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise, 
            tau=0.01, observation_range=(env.observation_space.low, env.observation_space.high),
            critic_l2_reg=0, actor_lr=1e-3, critic_lr=1e-3, memory_limit=100000)
        model.learn(total_timesteps=1e5)
        model.save("checkpoints/ddpg_" + env_name)

    else:
        model = DDPG.load("checkpoints/ddpg_" + env_name)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            print("state: ", obs, " reward: ", rewards, " done: ", dones, "info: ", info)

    del model # remove to demonstrate saving and loading

def run_baseline_ppo2(env_name, n_cpu=4, train=True):
    from stable_baselines.common.policies import MlpPolicy
    from stable_baselines.common.vec_env import SubprocVecEnv
    from stable_baselines import PPO2

    if train:
        # multiprocess environment
        env = SubprocVecEnv([lambda: gym.make(env_name) for i in range(n_cpu)])
        model = PPO2(MlpPolicy, env, verbose=1)
        model.learn(total_timesteps=100000)
        model.save("checkpoints/ppo2_" + env_name)
    else:
        from stable_baselines.common.vec_env import DummyVecEnv
        env = DummyVecEnv([lambda: gym.make(env_name)])
        model = PPO2.load("checkpoints/ppo2_" + env_name)

        obs = env.reset()
        while True:
            action, _states = model.predict(obs)
            obs, rewards, dones, info = env.step(action)
            env.render()
            print("state: ", obs, " reward: ", rewards, " done: ", dones, "info: ", info)


def main():
    # random_walk('parking-v0')

    run_baseline_ddpg('parking-v0', train=True)
    run_baseline_ddpg('parking-v0', train=False)

if __name__ == '__main__':
    main()
