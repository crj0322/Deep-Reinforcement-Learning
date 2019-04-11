import gym
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Layer
from keras.optimizers import Adam
import keras.backend as K
from agent.distributions import DiagGaussian
from agent.utils import RunningMeanStd, plot_training_curv, save_frames_as_gif


class PPOAgent:
    def __init__(self, env):
        self.env = env

        self.train_episodes = 1000
        self.max_steps = 200
        self.gamma = 0.99

        # Update parameters
        update_stpes = 10
        self.batch_size = 64

        # network parameters
        hidden_size = 64
        learning_rate = 1e-3
        entropy_coef = 0.01
        layer_num = 1

        # get size
        self.state_size = int(np.prod(np.array(env.observation_space.shape)))
        if isinstance(env.action_space, gym.spaces.Box):
            self.action_size = int(np.prod(np.array(env.action_space.shape)))
        elif isinstance(env.action_space, gym.spaces.Discrete):
            self.action_size = env.action_space.n
        elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
            self.action_size = env.action_space.nvec
        elif isinstance(env.action_space, gym.spaces.MultiBinary):
            self.action_size = env.action_space.n
        
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        self.policy = PPO(self.state_size, self.action_size, self.action_low, self.action_high,
            learning_rate, entropy_coef, update_stpes, hidden_size, layer_num)

        self.rewards_list = []
        
    def data_generator(self, render):
        states = np.zeros([self.batch_size, self.state_size], 'float32')
        actions = np.zeros([self.batch_size, self.action_size], 'float32')
        rewards = np.zeros(self.batch_size, 'float32')
        advantages = np.zeros(self.batch_size, 'float32')
        dones = np.zeros(self.batch_size + 1, 'int32')
        vs = np.zeros(self.batch_size + 1, 'float32')
        
        r_norm = RunningMeanStd(shape=(1,))
        s_norm = RunningMeanStd(shape=(self.state_size,))

        state = self.env.reset()
        total_reward = 0
        t = 0
        while True:
            if render:
                self.env.render()
            
            # normalize state
            s_norm.update(state[None, :])
            state = s_norm.normalize(state)

            action = self.policy.choose_action(state)
            v_t = self.policy.get_v(state)
            next_state, reward, done, _ = self.env.step(action)

            # yield batch
            if (t + 1) % self.batch_size == 0:
                # normalize reward
                r_norm.update(rewards)
                rewards = r_norm.normalize(rewards)

                # calculate advantage
                dones[i + 1] = done
                vs[i + 1] = v_t
                last_adv = 0
                for t in reversed(range(self.batch_size)):
                    nonterminal = 1 - dones[t + 1]
                    delta = rewards[t] + self.gamma * vs[t + 1] * nonterminal - vs[t]
                    advantages[t] = last_adv = delta + self.gamma * nonterminal * last_adv

                gains = advantages + vs[0:-1]
                yield states, actions, advantages[:, None], gains[:, None]

            i = t % self.batch_size
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            vs[i] = v_t
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
    
    def train(self, save_dir, plot_curv=True, render=False):
        gen = self.data_generator(render=render)

        while len(self.rewards_list) < self.train_episodes:
            states, actions, advantages, gains = next(gen)
            actor_loss, critic_loss = \
                self.policy.update(states, actions, advantages, gains)

            # check
            import math
            if math.isnan(actor_loss) or math.isnan(critic_loss):
                break
        
        self.policy.actor.save_weights(save_dir)

        if plot_curv:
            plot_training_curv(self.rewards_list)

    def test(self, weight_dir, max_steps=200):
        # Restore variables from disk.
        self.policy.actor.load_weights(weight_dir)

        # Initialize the simulation
        state = self.env.reset()
        frames = []

        total_reward = 0.
        for _ in range(max_steps):
            frame = self.env.render(mode='rgb_array')
            frames.append(frame)

            # Get action from Q-network
            action = self.policy.choose_action(state)

            state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                print("total reward: ", total_reward)
                self.env.reset()
                total_reward = 0.

        save_frames_as_gif(frames, weight_dir[:-4] + "gif")

class PPO(object):

    def __init__(self, state_size, action_size, action_low, action_high,
        learning_rate, entropy_coef, update_steps,
        hidden_size=64, layer_num=2, epsilon=0.2):

        self._state_size = state_size
        self._action_size = action_size
        self._action_low = action_low
        self._action_high = action_high
        self._lr = learning_rate
        self._entropy_coef = entropy_coef
        self._update_steps = update_steps
        self._hidden_size = hidden_size
        self._layer_num = layer_num
        self._epsilon = epsilon

        self._build_func()
        
    def _build_actor(self):
        inputs = Input(shape=(self._state_size,), dtype='float32')
        x = inputs
        for _ in range(self._layer_num):
            x = Dense(self._hidden_size, activation='tanh')(x)

        mean = Dense(self._action_size, activation=None)(x)
        logstd = NormalLayer(self._action_size)(mean)
        model = Model([inputs], [mean, logstd])
        return model
    
    def _build_critic(self):
        inputs = Input(shape=(self._state_size,), dtype='float32')
        x = inputs
        for _ in range(self._layer_num):
            x = Dense(self._hidden_size, activation='tanh')(x)
        v = Dense(1, activation=None)(x)
        model = Model([inputs], [v])
        return model
    
    def _build_func(self):
        # models
        self.old_actor = self._build_actor()
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # inputs
        state = Input(shape=(self._state_size,), name='state', dtype='float32')
        advantage = Input((1,), name='advantage', dtype='float32')
        action = Input((self._action_size,), name='action', dtype='float32')
        gain = Input((1,), name='gain', dtype='float32')

        # choose action
        old_mean, old_logstd = self.old_actor(state)
        mean, logstd = self.actor(state)

        pi = DiagGaussian(mean, logstd)
        old_pi = DiagGaussian(old_mean, old_logstd)

        sample = pi.sample()
        self._act = K.function([state], [sample])
        
        # get v
        v_t = self.critic(state)
        self._v = K.function([state], [v_t])

        # policy loss
        ratio = tf.exp(pi.logprob(action) - old_pi.logprob(action))

        adv = tf.squeeze(advantage, axis=-1)
        surr1 = ratio * adv
        surr2 = tf.clip_by_value(ratio, 1.0 - self._epsilon, 1.0 + self._epsilon) * adv
        pi_loss = - tf.reduce_mean(tf.minimum(surr1, surr2))

        # vf loss
        vf_loss = tf.reduce_mean(tf.square(gain - v_t))

        # entropy
        entropy_loss = - self._entropy_coef * tf.reduce_mean(pi.entropy())

        # Define optimizer and training function
        loss = pi_loss + vf_loss + entropy_loss
        optimizer = Adam(lr=self._lr)
        updates_op = optimizer.get_updates(
            params=self.actor.trainable_weights + self.critic.trainable_weights,
            loss=loss)
        self._train_fn = K.function(
            inputs=[state, action, advantage, gain],
            outputs=[pi_loss, vf_loss],
            updates=updates_op)

    def _update_old_actor(self):
        self.old_actor.set_weights(self.actor.get_weights())

    def update(self, s, a, adv, gt):
        self._update_old_actor()

        # batch-normalize advantages
        adv = (adv - adv.mean())/(adv.std()+1e-6)

        # update actor critic
        for _ in range(self._update_steps):
            actor_loss, critic_loss = self._train_fn([s, a, adv, gt])

            # check
            import math
            if math.isnan(actor_loss) or math.isnan(critic_loss):
                print('a loss: ', actor_loss)
                print('c loss', critic_loss)
                print(self.actor.get_weights())
                break

        return actor_loss, critic_loss

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self._act([s])[0]
        return np.squeeze(np.clip(a, self._action_low, self._action_high), axis=0)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return np.squeeze(self._v([s])[0], axis=0)


class NormalLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(NormalLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.logstd = self.add_weight(name='logstd',
                                      shape=(self.output_dim,),
                                      initializer='uniform',
                                      trainable=True)
        super(NormalLayer, self).build(input_shape)

    def call(self, x):
        return 0 * x + self.logstd

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
