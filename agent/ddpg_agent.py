from copy import copy
import numpy as np
import tensorflow as tf
from agent.utils import OUNoise, ReplayBuffer, plot_training_curv, init_memery, save_frames_as_gif


class DDPGAgent:
    def __init__(self, env):
        self.env = env

        self.train_episodes = 1000          # max number of episodes to learn from
        self.max_steps = 200                # max steps in an episode
        self.gamma = 0.99                   # future reward discount

        # Memory parameters
        self.memory_size = 10000            # memory capacity
        self.batch_size = 64                # experience mini-batch size

        # network parameters
        hidden_size = 64
        actor_lr = 1e-3
        critic_lr = 1e-2
        actor_reg = 0
        critic_reg = 0
        use_bn = False
        layer_num = 2
        tau = 0.001

        # get size
        self.state_size = int(np.prod(np.array(env.observation_space.shape)))
        self.action_size = int(np.prod(np.array(env.action_space.shape)))
        
        self.action_low = env.action_space.low
        self.action_high = env.action_space.high

        tf.reset_default_graph()

        # Actor (Policy) Model
        self.local_actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high,
            hidden_size=hidden_size, learning_rate=actor_lr, l2_rate=actor_reg,
            bn=use_bn, layer_num=layer_num)
        self.local_actor.build_model("local_actor", True)
        self.target_actor = copy(self.local_actor)
        self.target_actor.build_model("target_actor", False)

        # Critic (Value) Model
        self.local_critic = Critic(self.state_size, self.action_size,
            hidden_size=hidden_size, learning_rate=critic_lr, l2_rate=critic_reg,
            bn=use_bn, layer_num=layer_num)
        self.local_critic.build_model("local_critic", True)
        self.target_critic = copy(self.local_critic)
        self.target_critic.build_model("target_critic", False)

        # create soft update
        local_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='local')
        target_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target')
        self.init_target, self.update_target =\
            self.get_target_updates(local_weights, target_weights, tau)

        # build loss
        self.local_actor.build_loss()
        self.local_critic.build_loss()

        # Noise process
        self.noise = OUNoise(self.action_size)

    def get_target_updates(self, vars, target_vars, tau):
        soft_updates = []
        init_updates = []
        assert len(vars) == len(target_vars)
        for var, target_var in zip(vars, target_vars):
            init_updates.append(tf.assign(target_var, var))
            soft_updates.append(tf.assign(target_var, (1. - tau) * target_var + tau * var))
        assert len(init_updates) == len(vars)
        assert len(soft_updates) == len(vars)
        return tf.group(*init_updates), tf.group(*soft_updates)

    def learn(self, sess, batch):
        """Update QNet parameters using given batch of experience tuples."""
        states = np.array([each.state for each in batch]).astype(np.float32)
        actions = np.array([each.action for each in batch]).astype(np.float32)
        rewards = np.array([each.reward for each in batch]).astype(np.float32).reshape(-1, 1)
        next_states = np.array([each.next_state for each in batch]).astype(np.float32)
        dones = np.array([each.done for each in batch]).astype(np.uint8).reshape(-1, 1)
        
        # Get predicted next-state actions and Q values from target models
        next_actions = self.target_actor.predict(sess, next_states)
        Q_targets_next = self.target_critic.predict(sess, next_states, next_actions)

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        critic_loss = self.local_critic.train(sess, states, actions, Q_targets)
        
        # Train actor model (local)
        action_gradients = self.local_critic.get_action_gradients(sess, states, actions)
        actor_loss = self.local_actor.train(sess, states, action_gradients)

        # Soft-update target models
        sess.run(self.update_target)

        return actor_loss, critic_loss
    
    def train(self, save_dir, plot_curv=True, render=True):
        state, memory = init_memery(self.env, self.memory_size, self.batch_size)

        # Now train with experiences
        saver = tf.train.Saver()
        rewards_list = []
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            sess.run(self.init_target)
            
            for ep in range(1, self.train_episodes):
                total_reward = 0
                t = 0
                while t < self.max_steps:
                    if render:
                        self.env.render()
                    
                    # Get action from Actor
                    # add some noise for exploration
                    action = self.local_actor.predict(sess, state.reshape((-1, self.state_size))) + \
                        self.noise.sample()
                    action = action.reshape(self.action_size,)
                    
                    # Take action, get new state and reward
                    next_state, reward, done, _ = self.env.step(action)
            
                    total_reward += reward

                    # Add experience to memory
                    memory.add(state, action, reward, next_state, done)

                    # Sample mini-batch from memory
                    batch = memory.sample(self.batch_size)
                    actor_loss, critic_loss = self.learn(sess, batch)
                    
                    if done:
                        # the episode ends so no next state
                        next_state = np.zeros(state.shape)
                        
                        # Start new episode
                        state = self.env.reset()
                        break

                    else:
                        state = next_state
                        t += 1

                print('Episode: {}'.format(ep),
                    'Total reward: {:.4f}'.format(total_reward),
                    'Actor loss: {:.4f}'.format(actor_loss),
                    'Critic loss: {:.4f}'.format(critic_loss))
                rewards_list.append((ep, total_reward))
            
            saver.save(sess, save_dir)

            if plot_curv:
                plot_training_curv(rewards_list)

    def test(self, weight_dir, max_steps=200):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Restore variables from disk.
            saver.restore(sess, weight_dir)

            # Initialize the simulation
            state = self.env.reset()
            frames = []

            total_reward = 0.
            for _ in range(max_steps):
                frame = self.env.render(mode='rgb_array')
                frames.append(frame)

                 # Get action from Q-network
                action = self.local_actor.predict(sess, state.reshape((-1, self.state_size)))

                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("total reward: ", total_reward)
                    self.env.reset()
                    total_reward = 0.

            save_frames_as_gif(frames, weight_dir[:-4] + "gif")


class Actor:
    def __init__(self, state_size, action_size, action_low, action_high,
                 hidden_size=64, layer_num=3, learning_rate=1e-4, l2_rate=None, bn=True):
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bn = bn

    def build_model(self, name, trainable):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.states = tf.placeholder(tf.float32, [None, self.state_size], name='states')

            # Add hidden layers
            if self.l2_rate:
                l2_reg = tf.contrib.layers.l2_regularizer(scale=self.l2_rate)
            else:
                l2_reg = None

            if self.bn:
                bn = tf.contrib.layers.batch_norm
            else:
                bn = None
            
            layer = self.states
            for i in range(self.layer_num):
                layer = tf.contrib.layers.fully_connected(layer, self.hidden_size, normalizer_fn=bn,
                    weights_regularizer=l2_reg, trainable=trainable, scope="fc{}".format(i))

            # Add final output layer with sigmoid activation
            w_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            raw_actions = tf.contrib.layers.fully_connected(layer, self.action_size, activation_fn=tf.nn.sigmoid,
                weights_initializer=w_init, weights_regularizer=l2_reg, trainable=trainable, scope="actions")

            # Scale [0, 1] output for each action dimension to proper range
            self.actions = tf.add(tf.multiply(raw_actions, self.action_range), self.action_low)

    def build_loss(self):
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_size], name='action_gradients')

         # Define loss function using action value (Q value) gradients
        self.loss = tf.reduce_mean(tf.multiply(-self.action_gradients, self.actions))

        # Define optimizer and training function
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def predict(self, sess, states):
        feed = {self.states: states}
        actions = sess.run(self.actions, feed_dict=feed)
        return actions

    def train(self, sess, states, action_gradients):
        feed = {self.states: states, self.action_gradients: action_gradients}
        actor_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed)
        return actor_loss


class Critic:
    def __init__(self, state_size, action_size, 
        hidden_size=64, layer_num=3,
        learning_rate=1e-3, l2_rate=1e-2, bn=True):
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.bn = bn

    def build_model(self, name, trainable):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            self.states = tf.placeholder(tf.float32, [None, self.state_size], name='states')
            self.actions = tf.placeholder(tf.float32, [None, self.action_size], name='actions')

            # Add hidden layer(s) for state pathway
            if self.l2_rate:
                l2_reg = tf.contrib.layers.l2_regularizer(scale=self.l2_rate)
            else:
                l2_reg = None

            if self.bn:
                bn = tf.contrib.layers.batch_norm
            else:
                bn = None
            
            layer = tf.concat([self.states, self.actions], axis=1)
            
            for i in range(self.layer_num):
                layer = tf.contrib.layers.fully_connected(layer, self.hidden_size, normalizer_fn=bn,
                    weights_regularizer=l2_reg, trainable=trainable, scope="fc{}".format(i))

            # Add final output layer to prduce action values (Q values)
            w_init = tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3)
            self.Q_values = tf.contrib.layers.fully_connected(layer, 1, activation_fn=None,
                weights_initializer=w_init, weights_regularizer=l2_reg, trainable=trainable, scope="Q_values")

    def build_loss(self):
        self.Q_targets = tf.placeholder(tf.float32, [None, 1], name='Q_targets')

        # Define optimizer and compile model for training with built-in loss function
        self.loss = tf.losses.mean_squared_error(self.Q_targets, self.Q_values)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        self.action_gradients = tf.gradients(self.Q_values, self.actions)

    def predict(self, sess, states, actions):
        feed = {self.states: states, self.actions: actions}
        Q_values = sess.run(self.Q_values, feed_dict=feed)
        return Q_values

    def train(self, sess, states, actions, Q_targets):
        feed = {self.states: states, self.actions: actions, self.Q_targets: Q_targets}
        critic_loss, _ = sess.run([self.loss, self.optimizer], feed_dict=feed)
        return critic_loss
    
    def get_action_gradients(self, sess, states, actions):
        feed = {self.states: states, self.actions: actions}
        action_gradients = sess.run(self.action_gradients, feed_dict=feed)
        return action_gradients[0]
