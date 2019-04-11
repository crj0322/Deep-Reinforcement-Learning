import numpy as np
from collections import namedtuple, deque
import matplotlib.pyplot as plt
from matplotlib import animation


def save_frames_as_gif(frames, filename_gif, dpi=72):
    """
    save a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi = dpi)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filename_gif, writer = 'imagemagick', fps=30)

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def plot_training_curv(rewards_list):
    eps, rews = np.array(rewards_list).T
    smoothed_rews = running_mean(rews, 10)
    plt.plot(eps[-len(smoothed_rews):], smoothed_rews)
    plt.plot(eps, rews, color='grey', alpha=0.3)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

def callback(rewards_list, target_score):
    # stop training if reward exceeds target_score
    _, rews = np.array(rewards_list).T
    is_solved = len(rewards_list) > 100 and np.sum(rews[-101:-1]) / 100 >= target_score
    return is_solved

def init_memery(env, memory_size, batch_size):
        # Initialize the simulation
        env.reset()
        # Take one random step to get the pole and cart moving
        state, reward, done, _ = env.step(env.action_space.sample())

        memory = ReplayBuffer(size=memory_size)

        # Make a bunch of random actions and store the experiences
        for _ in range(batch_size):

            # Make a random action
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            if done:
                # The simulation fails so no next state
                next_state = np.zeros(state.shape)
                # Add experience to memory
                memory.add(state, action, reward, next_state, done)
                
                # Start new episode
                state = env.reset()
            else:
                # Add experience to memory
                memory.add(state, action, reward, next_state, done)
                state = next_state

        return state, memory

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=None, theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu if mu is not None else np.zeros(self.size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size circular buffer to store experience tuples."""

    def __init__(self, size=1000):
        """Initialize a ReplayBuffer object."""
        self.memory = deque(maxlen=size)  # internal memory (list)
        self.experience = namedtuple("Experience",
    field_names=["state", "action", "reward", "next_state", "done"])
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # Create an Experience object, add it to memory
        # Note: If memory is full, start overwriting from the beginning
        exp = self.experience(state, action, reward, next_state, done)
        self.memory.append(exp)
    
    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        # Return a list or tuple of Experience objects sampled from memory
        idx = np.random.choice(np.arange(len(self.memory)), 
                               size=batch_size, 
                               replace=False)
        return [self.memory[ii] for ii in idx]

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.std = np.ones(shape, 'float32')
        self.epsilon = epsilon
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def normalize(self, x):
        return (x - self.mean) / (self.std + self.epsilon)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self._update_mean_var_count_from_moments(
            batch_mean, batch_var, batch_count)

        self.std = np.sqrt(self.var + self.epsilon)

    def _update_mean_var_count_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count
