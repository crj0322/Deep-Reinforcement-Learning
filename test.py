import sys
import getopt
import gym
import gym_parking

def random_walk(env_name):
    env = gym.make(env_name)
    env.reset()
    for _ in range(1000):
        env.render()

        # take a random action
        action = env.action_space.sample()
        import numpy as np
        # action = np.array([0., -1.])
        action = np.random.random_sample(2)*2 - 1
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

def run_ppo(env_name, train=True, load_weights=False):
    env = gym.make(env_name)
    model_weight = "checkpoints/" + env_name + "_ppo.h5"

    from agent.ppo_agent import PPOAgent
    ppo_agent = PPOAgent(env)
    if train:
        ppo_agent.train(model_weight, render=False, load_weights=load_weights)
    else:
        ppo_agent.test(model_weight)


def main(argv=None):
    if argv is None:
        argv = sys.argv

    train = True
    load_weights = False
    opts, args = getopt.getopt(argv[1:],'-h-t-w',['help','train', 'weights'])
    for opt_name, opt_value in opts:
        if opt_name in ('-h','--help'):
            print("[*] Help info")
            exit()
        if opt_name in ('-t','--test'):
            train = False
        if opt_name in ('-w','--weights'):
            load_weights = True

    # run_dqn('CartPole-v0', train=False)
    # run_ddpg('Pendulum-v0', train=False)
    # run_ppo('MountainCarContinuous-v0', train=True)
    run_ppo('CartPole-v0', train=train, load_weights=load_weights)

    # random_walk('parking-v0')
    # run_ddpg('parking-v0', train=False)

if __name__ == '__main__':
    sys.exit(main())
