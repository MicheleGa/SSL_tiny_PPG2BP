import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Mean


class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Implements an LR scheduler that warms up the learning rate for some training steps
    (usually at the beginning of the training) and then decays it
    with CosineDecay (see https://arxiv.org/abs/1608.03983)
    """

    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )
    

class BarlowLoss(Loss):
    """BarlowLoss class.

    BarlowLoss class. Creates a loss function based on the cross-correlation
    matrix.

    Attributes:
        batch_size: the batch size of the dataset
        lambda_amt: the value for lambda(used in cross_corr_matrix_loss)

    Methods:
        __init__: gets instance variables
        call: gets the loss based on the cross-correlation matrix
          make_diag_zeros: Used in calculating off-diagonal section
          of loss function; makes diagonals zeros.
        cross_corr_matrix_loss: creates loss based on cross correlation
          matrix.
    """

    def __init__(self, batch_size: int):
        """__init__ method.

        Gets the instance variables

        Arguments:
            batch_size: An integer value representing the batch size of the
              dataset. Used for cross correlation matrix calculation.
        """

        super().__init__()
        self.lambda_amt = 5e-3
        self.batch_size = batch_size

    def get_off_diag(self, c: tf.Tensor) -> tf.Tensor:
        """get_off_diag method.

        Makes the diagonals of the cross correlation matrix zeros.
        This is used in the off-diagonal portion of the loss function,
        where we take the squares of the off-diagonal values and sum them.

        Arguments:
            c: A tf.tensor that represents the cross correlation
              matrix

        Returns:
            Returns a tf.tensor which represents the cross correlation
            matrix with its diagonals as zeros.
        """

        zero_diag = tf.zeros(c.shape[-1])
        return tf.linalg.set_diag(c, zero_diag)

    def cross_corr_matrix_loss(self, c: tf.Tensor) -> tf.Tensor:
        """cross_corr_matrix_loss method.

        Gets the loss based on the cross correlation matrix.
        We want the diagonals to be 1's and everything else to be
        zeros to show that the two augmented images are similar.

        Loss function procedure:
        take the diagonal of the cross-correlation matrix, subtract by 1,
        and square that value so no negatives.

        Take the off-diagonal of the cc-matrix(see get_off_diag()),
        square those values to get rid of negatives and increase the value,
        and multiply it by a lambda to weight it such that it is of equal
        value to the optimizer as the diagonal(there are more values off-diag
        then on-diag)

        Take the sum of the first and second parts and then sum them together.

        Arguments:
            c: A tf.tensor that represents the cross correlation
              matrix

        Returns:
            Returns a tf.tensor which represents the cross correlation
            matrix with its diagonals as zeros.
        """

        # subtracts diagonals by one and squares them(first part)
        c_diff = tf.pow(tf.linalg.diag_part(c) - 1, 2)

        # takes off diagonal, squares it, multiplies with lambda(second part)
        off_diag = tf.pow(self.get_off_diag(c), 2) * self.lambda_amt

        # sum first and second parts together
        loss = tf.reduce_sum(c_diff) + tf.reduce_sum(off_diag)

        return loss

    def normalize(self, output: tf.Tensor) -> tf.Tensor:
        """normalize method.

        Normalizes the model prediction.

        Arguments:
            output: the model prediction.

        Returns:
            Returns a normalized version of the model prediction.
        """

        return (output - tf.reduce_mean(output, axis=0)) / tf.math.reduce_std(
            output, axis=0
        )

    def cross_corr_matrix(self, z_a_norm: tf.Tensor, z_b_norm: tf.Tensor) -> tf.Tensor:
        """cross_corr_matrix method.

        Creates a cross correlation matrix from the predictions.
        It transposes the first prediction and multiplies this with
        the second, creating a matrix with shape (n_dense_units, n_dense_units).
        See build_twin() for more info. Then it divides this with the
        batch size.

        Arguments:
            z_a_norm: A normalized version of the first prediction.
            z_b_norm: A normalized version of the second prediction.

        Returns:
            Returns a cross correlation matrix.
        """
        return (tf.transpose(z_a_norm) @ z_b_norm) / self.batch_size

    def call(self, z_a: tf.Tensor, z_b: tf.Tensor) -> tf.Tensor:
        """call method.

        Makes the cross-correlation loss. Uses the CreateCrossCorr
        class to make the cross corr matrix, then finds the loss and
        returns it(see cross_corr_matrix_loss()).

        Arguments:
            z_a: The prediction of the first set of augmented data.
            z_b: the prediction of the second set of augmented data.

        Returns:
            Returns a (rank-0) tf.Tensor that represents the loss.
        """

        z_a_norm, z_b_norm = self.normalize(z_a), self.normalize(z_b)
        c = self.cross_corr_matrix(z_a_norm, z_b_norm)
        loss = self.cross_corr_matrix_loss(c)
        return loss


def Conv_1D_block_2(inputs, model_width, kernel, strides, nl):
    # This function defines a 1D convolution operation with BN and activation.
    x = tf.keras.layers.Conv1D(model_width, kernel, strides=strides, padding="same", kernel_initializer="he_normal")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    elif nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    return x

def _squeeze(inputs):
    # This function defines a squeeze structure.

    input_channels = int(inputs.shape[-1])

    x = tf.keras.layers.GlobalAveragePooling1D()(inputs)
    x = tf.keras.layers.Dense(input_channels, activation='relu')(x)
    x = tf.keras.layers.Dense(input_channels, activation='hard_sigmoid')(x)
    x = tf.keras.layers.Reshape((1, input_channels))(x)
    x = tf.keras.layers.Multiply()([inputs, x])

    return x

def bottleneck_block_2(inputs, filters, kernel, e, s, squeeze, nl, alpha):
    # This function defines a basic bottleneck structure.

    input_shape = tf.keras.backend.int_shape(inputs)

    tchannel = int(e)
    cchannel = int(alpha * filters)

    r = s == 1 and input_shape[2] == filters

    x = Conv_1D_block_2(inputs, tchannel, 1, 1, nl)

    x = tf.keras.layers.SeparableConv1D(filters, kernel, strides=s, depth_multiplier=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if nl == 'HS':
        x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    if nl == 'RE':
        x = tf.keras.activations.relu(x, max_value=6.0)

    if squeeze:
        x = _squeeze(x)

    x = tf.keras.layers.Conv1D(cchannel, 1, strides=1, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    if r:
        x = tf.keras.layers.Add()([x, inputs])

    return x


def MobileNet_v3_Small(params):
    input_shape = params["n_steps"]
    alpha = params["alpha"]
    
    inputs = tf.keras.Input((input_shape, 1))

    x = Conv_1D_block_2(inputs, 16, 3, strides=2, nl='HS')
    x = bottleneck_block_2(x, 16, 3, e=16, s=2, squeeze=True, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=72, s=2, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 24, 3, e=88, s=1, squeeze=False, nl='RE', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=96, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 40, 5, e=240, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 48, 5, e=120, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 48, 5, e=144, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=288, s=2, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = bottleneck_block_2(x, 96, 5, e=576, s=1, squeeze=True, nl='HS', alpha=alpha)
    x = Conv_1D_block_2(x, 576, 1, strides=1, nl='HS')
    x = x * tf.keras.activations.relu(x + 3.0, max_value=6.0) / 6.0
    x = tf.keras.layers.Conv1D(1280, 1, padding='same')(x)
    outputs = tf.keras.layers.Flatten()(x)
    model = tf.keras.Model(inputs, outputs)

    return model


class BarlowModel(Model):
    """BarlowModel class.

    BarlowModel class. Responsible for making predictions and handling
    gradient descent with the optimizer.

    Attributes:
        model: the barlow model architecture.
        loss_tracker: the loss metric.

    Methods:
        train_step: one train step; do model predictions, loss, and
            optimizer step.
        metrics: Returns metrics.
    """

    def __init__(self, config):
        super().__init__()
        # encoder network
        # Mobile Net from https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Codes/MobileNet_1DCNN.py
        params = config["encoder_params"]
        self.mobile_net = MobileNet_v3_Small(params)
        print(self.mobile_net.summary())
        self.model = self.build_twin(params)
        print(self.model.summary())
        self.loss_tracker = Mean(name="loss")

    def build_twin(self, params) -> Model:
        """build_twin method.

        Builds a barlow twins model consisting of an encoder(resnet-34)
        and a projector, which generates embeddings for the images

        Returns:
            returns a barlow twins model
        """

        # number of dense neurons in the projector
        n_dense_units = params["n_dense_units"]

        last_layer = self.mobile_net.layers[-1].output

        # intermediate layers of the projector network
        n_layers = params["n_dense_layers"]
        for i in range(n_layers):
            dense = Dense(n_dense_units, name=f"projector_dense_{i}")
            if i == 0:
                x = dense(last_layer)
            else:
                x = dense(x)
            x = BatchNormalization(name=f"projector_bn_{i}")(x)
            x = ReLU(name=f"projector_relu_{i}")(x)

        x = Dense(n_dense_units, name=f"projector_dense_{n_layers}")(x)

        return Model(self.mobile_net.input, x)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, batch: tf.Tensor) -> tf.Tensor:
        """train_step method.

        Do one train step. Make model predictions, find loss, pass loss to
        optimizer, and make optimizer apply gradients.

        Arguments:
            batch: one batch of data to be given to the loss function.

        Returns:
            Returns a dictionary with the loss metric.
        """

        # get the two augmentations from the batch
        y_a, y_b = batch

        with tf.GradientTape() as tape:
            # get two versions of predictions
            z_a, z_b = self.model(y_a, training=True), self.model(y_b, training=True)
            loss = self.loss(z_a, z_b)

        grads_model = tape.gradient(loss, self.model.trainable_variables)

        self.optimizer.apply_gradients(zip(grads_model, self.model.trainable_variables))
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}