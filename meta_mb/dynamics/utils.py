from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def normalize(data_array, mean, std):
    return (data_array - mean) / (std + 1e-10)


def denormalize(data_array, mean, std):
    return data_array * (std + 1e-10) + mean


def train_test_split(obs, act, delta, test_split_ratio=0.2):
    assert obs.shape[0] == act.shape[0] == delta.shape[0]
    dataset_size = obs.shape[0]
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    split_idx = int(dataset_size * (1-test_split_ratio))

    idx_train = indices[:split_idx]
    idx_test = indices[split_idx:]
    assert len(idx_train) + len(idx_test) == dataset_size

    return obs[idx_train, :], act[idx_train, :], delta[idx_train, :], \
           obs[idx_test, :], act[idx_test, :], delta[idx_test, :]



def create_decnn(name,
               kernel_sizes,
               strides,
               num_filters,
               hidden_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    assert len(kernel_sizes) == len(strides) == len(num_filters)

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        for idx, (kernel_size, stride, filter) in enumerate(zip(kernel_sizes, strides, num_filters)):
            x = tf.layers.conv2d_transpose(x,
                                           filters=filter,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           name='conv_%d' % idx,
                                           activation=hidden_nonlinearity,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init,
                                           )

        output_var = tf.contrib.layers.flatten(x)

    return input_var, output_var


def create_cnn(name,
               kernel_sizes,
               strides,
               num_filters,
               hidden_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    assert len(kernel_sizes) == len(strides) == len(num_filters)

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    x = input_var

    for idx, (kernel_size, stride, filter) in enumerate(zip(kernel_sizes, strides, num_filters)):
        x = tf.layers.conv2d(x,
                             filters=filter,
                             kernel_size=kernel_size,
                             strides=stride,
                             name='conv_%d' % idx,
                             activation=hidden_nonlinearity,
                             kernel_initializer=w_init,
                             bias_initializer=b_init,
                             )

    output_var = tf.contrib.layers.flatten(x)

    return input_var, output_var


def create_dnn(name,
               output_dim,
               kernel_sizes,
               strides,
               num_filters,
               hidden_nonlinearity,
               output_nonlinearity,
               hidden_dim,
               n_channels,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (tuple): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    assert len(kernel_sizes) == len(strides) == len(num_filters)

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    with tf.variable_scope(name):
        x = input_var

        x = tf.reshape(x, [-1, hidden_dim, hidden_dim, n_channels])

        for idx, (kernel_size, stride, filter) in enumerate(zip(kernel_sizes, strides, num_filters)):
            x = tf.layers.conv2d_transpose(x,
                                           filters=filter,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           name='conv_t_%d' % idx,
                                           activation=hidden_nonlinearity,
                                           kernel_initializer=w_init,
                                           bias_initializer=b_init,
                                           )

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x,
                            np.prod(output_dim),
                            name='output',
                            activation=output_nonlinearity,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            # reuse=reuse,
                            )

        output_var = tf.reshape(x, (-1,) + output_dim)

    return input_var, output_var


def create_mlp(name,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               batch_normalization=False,
               reuse=False,
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None

    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')
    x = input_var

    for idx, hidden_size in enumerate(hidden_sizes):
        if batch_normalization == 'traning':
            x = tf.layers.batch_normalization(x, training=True)
        elif batch_normalization == 'testing':
            x = tf.layers.batch_normalization(x, training=False)

        x = tf.layers.dense(x,
                            hidden_size,
                            name='hidden_%d' % idx,
                            activation=hidden_nonlinearity,
                            kernel_initializer=w_init,
                            bias_initializer=b_init,
                            reuse=reuse,
                            )

    if batch_normalization == 'traning':
        x = tf.layers.batch_normalization(x, training=True)
    elif batch_normalization == 'testing':
        x = tf.layers.batch_normalization(x, training=False)

    output_var = tf.layers.dense(x,
                                 output_dim,
                                 name='output',
                                 activation=output_nonlinearity,
                                 kernel_initializer=w_init,
                                 bias_initializer=b_init,
                                 reuse=reuse,
                                 )

    return input_var, output_var


def create_rnn(name,
               cell_type,
               output_dim,
               hidden_sizes,
               hidden_nonlinearity,
               output_nonlinearity,
               input_dim=None,
               input_var=None,
               state_var=None,
               w_init=tf.contrib.layers.xavier_initializer(),
               b_init=tf.zeros_initializer(),
               reuse=False,
               ):
    """
    Creates a MLP network
    Args:
        name (str): scope of the neural network
        output_dim (int): dimension of the output
        hidden_sizes (tuple): tuple with the hidden sizes of the fully connected network
        hidden_nonlinearity (tf): non-linearity for the activations in the hidden layers
        output_nonlinearity (tf or None): output non-linearity. None results in no non-linearity being applied
        input_dim (tuple): dimensions of the input variable e.g. (None, action_dim)
        input_var (tf.placeholder or tf.Variable or None): Input of the network as a symbolic variable
        w_init (tf.initializer): initializer for the weights
        b_init (tf.initializer): initializer for the biases
        reuse (bool): reuse or not the network

    Returns:
        input_var (tf.placeholder or tf.Variable): Input of the network as a symbolic variable
        output_var (tf.Tensor): Output of the network as a symbolic variable

    """

    assert input_var is not None or input_dim is not None
    if input_var is None:
        input_var = tf.placeholder(dtype=tf.float32, shape=input_dim, name='input')

    if state_var is None:
        create_hidden = True
    else:
        create_hidden = False

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell = []
        if state_var is None:
            state_var = []

        for idx, hidden_size in enumerate(hidden_sizes):
            if cell_type == 'lstm':
                cell.append(tf.nn.rnn_cell.LSTMCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    c = tf.placeholder(tf.float32, (None, hidden_size), name='cell_state_%d' % idx)
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(tf.contrib.rnn.LSTMStateTuple(c, h))
            elif cell_type == 'gru':
                cell.append(tf.nn.rnn_cell.GRUCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(h)
            elif cell_type == 'rnn':
                cell.append(tf.nn.rnn_cell.RNNCell(hidden_size, activation=hidden_nonlinearity))
                if create_hidden:
                    h = tf.placeholder(tf.float32, (None, hidden_size), name='hidden_state_%d' % idx)
                    state_var.append(h)
            else:
                raise NotImplementedError

        if len(hidden_sizes) > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell(cell)
            if create_hidden:
                state_var = tuple(state_var)
        else:
            cell = cell[0]
            if create_hidden:
                state_var = state_var[0]

        outputs, next_state_var = tf.nn.dynamic_rnn(cell,
                                                    input_var,
                                                    initial_state=state_var,
                                                    time_major=False,
                                                    )

        if output_dim is None:
            output_var = outputs
        else:
            output_var = tf.layers.dense(outputs,
                                         output_dim,
                                         name='output',
                                         activation=output_nonlinearity,
                                         kernel_initializer=w_init,
                                         bias_initializer=b_init,
                                         )

    return input_var, state_var,  output_var, next_state_var, cell


"""================================================================================================================"""
def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val

class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network.
    """
    def __init__(self, x_dim):
        """Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.

        Returns: None.
        """
        self.fitted = False
        with tf.variable_scope("Scaler"):
            self.mu = tf.get_variable(
                name="scaler_mu", shape=[1, x_dim], initializer=tf.constant_initializer(0.0),
                trainable=False
            )
            self.sigma = tf.get_variable(
                name="scaler_std", shape=[1, x_dim], initializer=tf.constant_initializer(1.0),
                trainable=False
            )

        self.cached_mu, self.cached_sigma = np.zeros([0, x_dim]), np.ones([1, x_dim])

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu.load(mu)
        self.sigma.load(sigma)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.

        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.

        Returns: None.
        """
        self.cached_mu = self.mu.eval()
        self.cached_sigma = self.sigma.eval()

    def load_cache(self):
        """Loads values from the cache

        Returns: None.
        """
        self.mu.load(self.cached_mu)
        self.sigma.load(self.cached_sigma)
