from collections import OrderedDict
from copy import deepcopy

from . import vanilla
from pdb import set_trace as st
import tensorflow as tf

def get_preprocessor_from_params(env, preprocessor_params, *args, **kwargs):
    if preprocessor_params is None:
        return None

    preprocessor_type = preprocessor_params.get('type', None)
    preprocessor_kwargs = deepcopy(preprocessor_params.get('kwargs', {}))

    if preprocessor_type is None:
        return None

    preprocessor = PREPROCESSOR_FUNCTIONS[
        preprocessor_type](
            *args,
            **preprocessor_kwargs,
            **kwargs)

    return preprocessor

def create_double_value_function(value_fn, *args, **kwargs):
    # TODO(hartikainen): The double Q-function should support the same
    # interface as the regular ones. Implement the double min-thing
    # as a Keras layer.
    value_fns = tuple(value_fn(*args, **kwargs) for i in range(2))
    return value_fns

VALUE_FUNCTIONS = {
    'double_feedforward_Q_function': lambda *args, **kwargs: (
        create_double_value_function(
            vanilla.create_feedforward_Q_function, *args, **kwargs)),
}

def get_convnet_preprocessor(name='convnet_preprocessor', **kwargs):
   from meta_mb.models.convnet import convnet_model
   preprocessor = convnet_model(name=name, **kwargs)
   return preprocessor


def get_Q_function_from_variant(variant, env, *args, **kwargs):
    Q_params = deepcopy(variant['Q_params'])
    Q_type = deepcopy(Q_params['type'])
    Q_kwargs = deepcopy(Q_params['kwargs'])

    observation_preprocessors_params = Q_kwargs.pop(
        'observation_preprocessors_params', {}).copy()
    observation_shapes = tf.TensorShape(env.observation_space.shape)
    action_shape = tf.TensorShape(env.action_space.shape)
    input_shapes = {
        'observations': observation_shapes,
        'actions': action_shape,
    }

    action_preprocessor = None
    observation_preprocessors = None
    preprocessors = {
        'observations': observation_preprocessors,
        'actions': action_preprocessor,
    }

    Q_function = VALUE_FUNCTIONS[Q_type](
        input_shapes=input_shapes,
        *args,
        preprocessors=preprocessors,
        **Q_kwargs,
        **kwargs)

    return Q_function
