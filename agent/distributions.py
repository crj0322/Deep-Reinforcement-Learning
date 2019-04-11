import tensorflow as tf
import numpy as np


class DiagGaussian(object):
    
    def __init__(self, mu, log_sigma):
        self.mu        = mu
        self.log_sigma = log_sigma
        self.sigma     = tf.exp(log_sigma)

    def mode(self):
        return self.mu

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mu) / self.sigma), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.log_sigma, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.log_sigma + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mu + self.sigma * tf.random_normal(tf.shape(self.mu))

    def prob(self, x):
        return tf.exp(-self.neglogp(x))

    def logprob(self, x):
        return -self.neglogp(x)

        