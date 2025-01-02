import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer


class Augmentation(Layer):
    """Base augmentation class.
    https://github.com/uchidalab/time_series_augmentation/blob/master/utils/augmentation.py#L8

    Base augmentation class. Contains the random_execute method.

    Methods:
        random_execute: method that returns true or false based 
          on a probability. Used to determine whether an augmentation 
          will be run.
    """

    def __init__(self, seed):
        super().__init__()
        self.seed = seed

    def random_execute(self, prob: float) -> bool:
        """random_execute function.

        Arguments:
            prob: a float value from 0-1 that determines the 
              probability.

        Returns:
            returns true or false based on the probability.
        """

        return tf.random.uniform([], minval=0, maxval=1, seed=self.seed) < prob


class Jitter(Augmentation):
    def __init__(self, seed, sigma=0.05):
        super().__init__(seed)
        self.seed = seed
        self.sigma = sigma

    def call(self, inputs, prob=0.5):
        if self.random_execute(prob=prob):
            noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=self.sigma, seed=self.seed)
            return inputs + noise
        else:
            return inputs


class Scaling(Augmentation):
    def __init__(self, seed, sigma=0.1):
        super().__init__(seed)
        self.seed = seed
        self.sigma = sigma

    def call(self, inputs, prob=0.5):
        if self.random_execute(prob=prob):
            factor = tf.random.normal(shape=tf.shape(inputs), mean=1.0, stddev=self.sigma, seed=self.seed)
            return inputs * factor
        else:
            return inputs


class Flip(Augmentation):
    def __init__(self, seed):
        super(Flip, self).__init__(seed)
        self.seed = seed

    def call(self, inputs, prob=0.5):
        if self.random_execute(prob=prob):
            # Randomly flip each feature
            flips = tf.random.uniform(shape=tf.shape(inputs), minval=-1.0, maxval=1.0, seed=self.seed)
            flips = tf.where(flips > 0.0, tf.ones_like(flips), -tf.ones_like(flips))

            # Randomly permute feature order (N.B. input is (n_inputs,))
            rotate_axis = tf.random.shuffle(tf.range(tf.shape(inputs)[0]), seed=self.seed)
            rotated_inputs = tf.gather(inputs, rotate_axis, axis=0)

            # Apply flips
            return flips * rotated_inputs
        else:
            return inputs
    

class RandomAugmentor(Model):
    """RandomAugmentor class.

    RandomAugmentor class. Chains all the augmentations into 
    one pipeline.

    Attributes:
        time_shift: Instance variable representing the TimeShift layer.
        scaling: Instance variable representing the Scaling layer.
        noise_injection: Instance variable representing the NoiseInjection layer.
        time_warp: Instance variable representing the TimeWarp layer.

    Methods:
        call: chains layers in pipeline together
    """

    def __init__(self, seed):
        super().__init__()
        self.jitter = Jitter(seed)
        self.scaling = Scaling(seed)
        self.flip = Flip(seed)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.jitter(x)
        x = self.scaling(x)
        x = self.flip(x)
        return x