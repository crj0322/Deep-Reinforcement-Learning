import gym
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Layer
from keras.optimizers import Adam
import keras.backend as K
from agent.model import mlp_model
from agent.distributions import DiagGaussian
from agent.utils import RunningMeanStd, plot_training_curv, save_frames_as_gif


class PPOAgent:
    def __init__(self, env):
        self.env = env

        self.train_episodes = 2000
        self.max_steps = 200
        self.gamma = 0.9

        # Update parameters
        update_steps = 10
        self.batch_size = 64

        # network parameters
        hidden_size = 32
        learning_rate = 1e-3
        vf_coef = 0.5
        entropy_coef = 0
        layer_num = 1

        # get size
        self.state_size = int(np.prod(np.array(env.observation_space.shape)))
        continuous = False
        if isinstance(env.action_space, gym.spaces.Box):
            continuous = True
            self.action_size = int(np.prod(np.array(env.action_space.shape)))
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_size = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_size = env.action_space.nvec
        elif isinstance(env.action_space, gym.spaces.MultiBinary):
            self.action_size = env.action_space.n

        self.policy = PPO(self.state_size, self.action_size,
            update_steps, continuous, learning_rate, vf_coef, entropy_coef)

        self.rewards_list = []
        
    def data_generator(self, render):
        states = np.zeros([self.batch_size, self.state_size], 'float32')
        logpis = np.zeros(self.batch_size, 'float32')
        rewards = np.zeros(self.batch_size, 'float32')
        advantages = np.zeros(self.batch_size, 'float32')
        dones = np.zeros(self.batch_size + 1, 'int32')
        values = np.zeros(self.batch_size + 1, 'float32')
        
        r_norm = RunningMeanStd(shape=(1,))

        state = self.env.reset()
        total_reward = 0
        t = 0
        while True:
            if render:
                self.env.render()

            action, logpi, value = self.policy.predict(state)
            next_state, reward, done, _ = self.env.step(action)

            # yield batch
            if (t + 1) % self.batch_size == 0:
                # normalize reward
                r_norm.update(rewards)
                rewards = r_norm.normalize(rewards)

                # calculate advantage
                dones[i + 1] = done
                values[i + 1] = value
                last_adv = 0
                for t in reversed(range(self.batch_size)):
                    nonterminal = 1 - dones[t + 1]
                    delta = rewards[t] + self.gamma * values[t + 1] * nonterminal - values[t]
                    advantages[t] = last_adv = delta + self.gamma * nonterminal * last_adv

                gains = advantages + values[0:-1]
                yield states, logpis, advantages, gains

            i = t % self.batch_size
            states[i] = state
            logpis[i] = logpi
            rewards[i] = reward
            values[i] = value
            dones[i] = done
            total_reward += reward
            if done or ((t + 1) % self.max_steps == 0):
                state = self.env.reset()
                self.rewards_list.append((len(self.rewards_list), total_reward))
                print('Episode: {}'.format(len(self.rewards_list)),
                'Total reward: {:.4f}'.format(total_reward))
                total_reward = 0
            else:
                state = next_state

            t += 1
    
    def train(self, save_dir, plot_curv=True, render=False, load_weights=False):
        if load_weights:
            self.policy.load_weights(save_dir)
        
        gen = self.data_generator(render=render)

        while len(self.rewards_list) < self.train_episodes:
            states, logpis, advantages, gains = next(gen)
            actor_loss, critic_loss = \
                self.policy.update(states, advantages, gains, logpis)

            # check
            import math
            if math.isnan(actor_loss) or math.isnan(critic_loss):
                break
        
        self.policy.save_weights(save_dir)

        if plot_curv:
            plot_training_curv(self.rewards_list)

    def test(self, weight_dir, max_steps=200):
        # Restore variables from disk.
        self.policy.load_weights(weight_dir)

        # Initialize the simulation
        state = self.env.reset()
        frames = []

        total_reward = 0.
        for _ in range(max_steps):
            frame = self.env.render(mode='rgb_array')
            frames.append(frame)

            # Get action from Q-network
            action, _, _ = self.policy.predict(state)

            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                print("total reward: ", total_reward)
                self.env.reset()
                total_reward = 0.

        save_frames_as_gif(frames, weight_dir[:-4] + "gif")

class PPO(object):

    def __init__(self,
        state_size,
        action_size,
        update_steps,
        continuous,
        learning_rate,
        vf_coef,
        entropy_coef,
        clip_ratio=0.2):

        self._update_steps = update_steps
        self._model = mlp_model(state_size, action_size, continuous=continuous)
        self._build_func(continuous, clip_ratio, vf_coef, entropy_coef, learning_rate)
    
    def _build_func(self, continuous, clip_ratio, vf_coef, entropy_coef, learning_rate):
        state = self._model.inputs[0]
        logits, value = self._model.outputs

        # policy distribution
        import tensorflow_probability as tfp

        if continuous:
            mean, logstd = tf.split(logits, 2, axis=-1)
            pi =  tfp.distributions.MultivariateNormalDiag(mean, tf.exp(logstd))
        else:
            pi =  tfp.distributions.Categorical(logits)

        # choose action
        sample = pi.sample()

        # policy log probability for loss calculation
        logpi = pi.log_prob(sample)
        self._pred = K.function([state], [sample, logpi, value])

        # calculate loss
        adv = tf.placeholder(tf.float32, [None], name="advantage")
        gain = tf.placeholder(tf.float32, [None], name="gain")
        logpi_old = tf.placeholder(tf.float32, [None], name="logpi_old")

        # policy loss
        ratio = tf.exp(logpi - logpi_old)
        surr1 = ratio * adv
        surr2 = tf.clip_by_value(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        pi_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))

        # vf loss
        vf_loss = vf_coef * tf.reduce_mean(tf.square(gain - value))

        # entropy loss
        entropy_loss = - entropy_coef * tf.reduce_mean(pi.entropy())

        # Define optimizer and training function
        loss = pi_loss + vf_loss + entropy_loss
        optimizer = Adam(lr=learning_rate)
        updates_op = optimizer.get_updates(
            params=self._model.trainable_weights,
            loss=loss)
        self._train_fn = K.function(
            inputs=[state, adv, gain, logpi_old],
            outputs=[pi_loss, vf_loss, entropy_loss],
            updates=updates_op)

    def update(self, state, adv, gain, old_logpi):
        # batch-normalize advantages
        adv = (adv - adv.mean())/(adv.std()+1e-6)

        # update actor critic
        for _ in range(self._update_steps):
            actor_loss, critic_loss, _ = self._train_fn([state, adv, gain, old_logpi])

            # check
            import math
            if math.isnan(actor_loss) or math.isnan(critic_loss):
                print('a loss: ', actor_loss)
                print('c loss', critic_loss)
                break

        return actor_loss, critic_loss

    def predict(self, state):
        if state.ndim < 2:
            state = state[None, :]

        act, logpi, value = self._pred([state])
        act = np.squeeze(act, axis=0)
        # logpi = np.squeeze(logpi, axis=0)

        return act, logpi, value

    def save_weights(self, dir):
        self._model.save_weights(dir)

    def load_weights(self, dir):
        self._model.load_weights(dir)

