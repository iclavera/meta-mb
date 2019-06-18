import tensorflow as tf
from tensorflow.python.keras.engine import training_utils

from meta_mb.models.feedforward import feedforward_model
from meta_mb.models.utils import flatten_input_structure, create_inputs
from meta_mb.utils.keras import PicklableModel


def create_feedforward_Q_function(input_shapes,
                                  *args,
                                  preprocessors=None,
                                  name='feedforward_Q',
                                  **kwargs):
    inputs_flat = create_inputs(input_shapes)
    preprocessors_flat = (
        flatten_input_structure(preprocessors)
        if preprocessors is not None
        else tuple(None for _ in inputs_flat))

    assert len(inputs_flat) == len(preprocessors_flat), (
        inputs_flat, preprocessors_flat)

    preprocessed_inputs = [
        preprocessor(input_) if preprocessor is not None else input_
        for preprocessor, input_
        in zip(preprocessors_flat, inputs_flat)
    ]

    Q_function = feedforward_model(
        *args,
        output_size=1,
        name=name,
        **kwargs)

    Q_function = PicklableModel(inputs_flat, Q_function(preprocessed_inputs))
    return Q_function
