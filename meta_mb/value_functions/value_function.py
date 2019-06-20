import tensorflow as tf
import numpy as np

from meta_mb.utils.serializable import Serializable


class SumQFunction(Serializable):
    def __init__(self,
                 observation_shape,
                 action_shape,
                 q_functions):
        self._Serializable__initialize(locals())

        self.q_functions = q_functions

        assert len(observation_shape) == 1, observation_shape
        self._Do = observation_shape[0]
        assert len(action_shape) == 1, action_shape
        self._Da = action_shape[0]

        self._observations_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self._Do), name='observations')
        self._actions_ph = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self._Da), name='actions')

        self._output = self.output_for(
            self._observations_ph, self._actions_ph, reuse=True)

    def output_for(self, observations, actions, reuse=False):
        outputs = [
            qf.output_for(observations, actions, reuse=reuse)
            for qf in self.q_functions
        ]
        output = tf.add_n(outputs)
        return output

    def _eval(self, observations, actions):
        feeds = {
            self._observations_ph: observations,
            self._actions_ph: actions
        }

        return tf.compat.v1.keras.backend.get_session().run(self._output, feeds)

    def get_param_values(self):
        all_values_list = [qf.get_param_values() for qf in self.q_functions]

        return np.concatenate(all_values_list)

    def set_param_values(self, all_values):
        param_sizes = [qf.get_param_values().size for qf in self.q_functions]
        split_points = np.cumsum(param_sizes)[:-1]

        all_values_list = np.split(all_values, split_points)

        for values, qf in zip(all_values_list, self.q_functions):
            qf.set_param_values(values)


from .mlp import MLPFunction

class NNVFunction(MLPFunction):

    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='vf'):
        Serializable.quick_init(self, locals())

        # self._Do = env_spec.observation_space.flat_dim
        self._Do = np.prod(env_spec.observation_space.shape)
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        super(NNVFunction, self).__init__(
            name, (self._obs_pl,), hidden_layer_sizes)


class NNQFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='qf'):
        Serializable.quick_init(self, locals())

        # self._Da = env_spec.action_space.flat_dim
        self._Da = np.prod(env_spec.action_space.shape)
        # self._Do = env_spec.observation_space.flat_dim
        self._Do = np.prod(env_spec.observation_space.shape)

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        super(NNQFunction, self).__init__(
            name, (self._obs_pl, self._action_pl), hidden_layer_sizes)


class NNDiscriminatorFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), num_skills=None):
        assert num_skills is not None
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        # self._Da = env_spec.action_space.flat_dim
        # self._Do = env_spec.observation_space.flat_dim
        self._Da = np.prod(env_spec.action_space.shape)
        slef._Do = np.prod(env_spec.observation_space.shape)

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )
        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._name = 'discriminator'
        self._input_pls = (self._obs_pl, self._action_pl)
        self._layer_sizes = list(hidden_layer_sizes) + [num_skills]
        self._output_t = self.get_output_for(*self._input_pls)
