import gym
import gym_parking

def random_walk(env_name):
    env = gym.make(env_name)
    env.reset()
    for _ in range(1000):
        env.render()

        # take a random action
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        print("state: ", next_state, " reward: ", reward, " done: ", done)

        if done:
            env.reset()

def run_dqn(env_name, train=True):
    env = gym.make(env_name)
    model_weight = "checkpoints/" + env_name + ".ckpt"

    from agent.dqn_agent import QAgent
    q_agent = QAgent(env)
    if train:
        q_agent.train(model_weight)
    else:
        q_agent.test(model_weight)

def run_ddpg(env_name, train=True):
    env = gym.make(env_name)
    model_weight = "checkpoints/" + env_name + ".ckpt"

    from agent.ddpg_agent import DDPGAgent
    ddpg_agent = DDPGAgent(env)
    if train:
        ddpg_agent.train(model_weight)
    else:
        ddpg_agent.test(model_weight)


def main():
    # run_dqn('CartPole-v0', train=False)
    # run_ddpg('Pendulum-v0', train=False)

    # random_walk('parking-v0')
    run_ddpg('parking-v0')

if __name__ == '__main__':
    main()
