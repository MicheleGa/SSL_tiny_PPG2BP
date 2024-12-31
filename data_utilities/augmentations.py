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

    def __init__(self):
        super().__init__()

    @tf.function
    def random_execute(self, prob: float) -> bool:
        """random_execute function.

        Arguments:
            prob: a float value from 0-1 that determines the 
              probability.

        Returns:
            returns true or false based on the probability.
        """

        return tf.random.uniform([], minval=0, maxval=1) < prob


class Jitter(Augmentation):
    def __init__(self, sigma=0.03, n_input=875):
        super().__init__()
        self.sigma = sigma
        self.n_input = n_input

    def call(self, inputs, prob=0.1):
        if self.random_execute(prob=prob):
            noise = tf.random.normal(shape=(self.n_input,), mean=0.0, stddev=self.sigma)
            return inputs + noise
        else:
            return inputs


class Scaling(Augmentation):
    def __init__(self, sigma=0.1, n_input=875):
        super().__init__()
        self.sigma = sigma
        self.n_input = n_input

    def call(self, inputs, prob=0.1):
        if self.random_execute(prob=prob):
            factor = tf.random.normal(shape=(self.n_input,), mean=1.0, stddev=self.sigma)
            return inputs * factor
        else:
            return inputs


class Rotation(Augmentation):
    def __init__(self, n_input=875):
        super().__init__()
        self.n_input = n_input

    def call(self, inputs, prob=0.1):
        if self.random_execute(prob=prob):
            flip = tf.random.uniform(shape=(self.n_input,), minval=-1, maxval=1, dtype=tf.int32)
            flip = tf.cast(flip, dtype=tf.float32)
            rotate_axis = tf.random.shuffle(tf.range(self.n_input))
            return flip * tf.gather(inputs, rotate_axis)
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

    def __init__(self, n_input):
        super().__init__()
        self.jitter = Jitter(n_input=n_input)
        self.scaling = Scaling(n_input=n_input)
        self.rotation = Rotation(n_input=n_input)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.jitter(x)
        x = self.scaling(x)
        x = self.rotation(x)
        return x