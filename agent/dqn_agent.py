import numpy as np
import tensorflow as tf
from agent.utils import ReplayBuffer, callback, plot_training_curv, init_memery, save_frames_as_gif


class QAgent:
    def __init__(self, env):
        self.env = env

        self.train_episodes = 1000          # max number of episodes to learn from
        self.max_steps = 200                # max steps in an episode
        self.gamma = 0.99                   # future reward discount

        # Exploration parameters
        self.explore_start = 1.0            # exploration probability at start
        self.explore_stop = 0.01            # minimum exploration probability 
        self.decay_rate = 0.0001            # exponential decay rate for exploration prob

        # Network parameters
        layer_num = 2
        hidden_size = 16
        learning_rate = 1e-3
        l2_rate = 0
        use_bn = False

        # Memory parameters
        self.memory_size = 10000            # memory capacity
        self.batch_size = 64                # experience mini-batch size

        self.solved_condition = lambda x: callback(x, 195.) # cartpole

        # Build QNet
        tf.reset_default_graph()
        self.state_dim = np.prod(np.array(env.observation_space.shape))
        self.action_dim = env.action_space.n
        self.mainQN = QNetwork(self.state_dim, self.action_dim, name='main',
            hidden_size=hidden_size, learning_rate=learning_rate, layer_num=layer_num,
            l2_rate=l2_rate, use_bn=use_bn)

    def act(self, sess, states):
        """Returns action(s) for given state(s) as per current policy."""
        feed = {self.mainQN.inputs_: states.reshape((1, *states.shape))}
        Qs = sess.run(self.mainQN.output, feed_dict=feed)
        actions = np.argmax(Qs)
        return actions

    def learn(self, sess, batch):
        """Update QNet parameters using given batch of experience tuples."""
        states = np.array([each.state for each in batch]).astype(np.float32)
        actions = np.array([each.action for each in batch]).astype(np.float32)
        rewards = np.array([each.reward for each in batch]).astype(np.float32)
        next_states = np.array([each.next_state for each in batch]).astype(np.float32)
        dones = np.array([each.done for each in batch]).astype(np.uint8)
        
        # Train network
        target_Qs = sess.run(self.mainQN.output, feed_dict={self.mainQN.inputs_: next_states})
        
        targets = rewards + self.gamma * np.max(target_Qs, axis=1) * (1 - dones)

        loss, _ = sess.run([self.mainQN.loss, self.mainQN.opt],
                            feed_dict={self.mainQN.inputs_: states,
                                    self.mainQN.targetQs_: targets,
                                    self.mainQN.actions_: actions})

        return loss
    
    def train(self, save_dir, plot_curv=True, render=True):
        state, memory = init_memery(self.env, self.memory_size, self.batch_size)

        # Now train with experiences
        saver = tf.train.Saver()
        rewards_list = []
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            
            step = 0
            for ep in range(1, self.train_episodes):
                total_reward = 0
                t = 0
                while t < self.max_steps:
                    step += 1

                    if render:
                        self.env.render()
                    
                    # Explore or Exploit
                    explore_p = self.explore_stop + (self.explore_start - self.explore_stop)*np.exp(-self.decay_rate*step) 
                    if explore_p > np.random.rand():
                        # Make a random action
                        action = self.env.action_space.sample()
                    else:
                        # Get action from Q-network
                        action = self.act(sess, state)
                    
                    # Take action, get new state and reward
                    next_state, reward, done, _ = self.env.step(action)
            
                    total_reward += reward

                     # Add experience to memory
                    memory.add(state, action, reward, next_state, done)
                    
                    # Sample mini-batch from memory
                    batch = memory.sample(self.batch_size)
                    loss = self.learn(sess, batch)
                    
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
                    'Training loss: {:.4f}'.format(loss),
                    'Explore P: {:.4f}'.format(explore_p))
                rewards_list.append((ep, total_reward))

                # solved condition
                if self.solved_condition and self.solved_condition(rewards_list):
                    break
                
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
                action = self.act(sess, state)

                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print("total reward: ", total_reward)
                    self.env.reset()
                    total_reward = 0.

            save_frames_as_gif(frames, weight_dir[:-4] + "gif")
        

class QNetwork:
    def __init__(self, state_size, action_size, l2_rate=0, use_bn=False,
        learning_rate=0.01, hidden_size=64, layer_num=2, name='QNetwork'):
        if l2_rate:
            l2_reg = tf.contrib.layers.l2_regularizer(scale=l2_rate)
        else:
            l2_reg = None

        if use_bn:
            bn = tf.contrib.layers.batch_norm
        else:
            bn = None

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # state inputs to the Q-network
            self.inputs_ = tf.placeholder(tf.float32, [None, state_size], name='inputs')
            
            # One hot encode the actions to later choose the Q-value for the action
            self.actions_ = tf.placeholder(tf.int32, [None], name='actions')
            one_hot_actions = tf.one_hot(self.actions_, action_size)
            
            # Target Q values for training
            self.targetQs_ = tf.placeholder(tf.float32, [None], name='target')
            
            # ReLU hidden layers
            layer = self.inputs_
            for i in range(layer_num):
                layer = tf.contrib.layers.fully_connected(layer, hidden_size,
                    normalizer_fn=bn, weights_regularizer=l2_reg, scope="fc{}".format(i))

            # Linear output layer
            self.output = tf.contrib.layers.fully_connected(layer, action_size, activation_fn=None,
                weights_regularizer=l2_reg, scope="output")
            
            ### Train with loss (targetQ - Q)^2
            # output has length 2, for two actions. This next line chooses
            # one value from output (per row) according to the one-hot encoded actions.
            Q = tf.reduce_sum(tf.multiply(self.output, one_hot_actions), axis=1)
            
            self.loss = tf.reduce_mean(tf.square(self.targetQs_ - Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
