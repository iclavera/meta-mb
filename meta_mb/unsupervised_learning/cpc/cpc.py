import os
import keras
from keras import backend as K
import tensorflow as tf

from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, res_block, make_periodic_lr, cross_entropy_loss

def network_encoder(x, code_size):

    ''' Define the network mapping images to embeddings '''

    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='linear')(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)

    return x

def network_encoder_resnet(x, code_size):
    x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu')(x)
    x = res_block(x, 64)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = res_block(x, 64)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)
    x = res_block(x, 64)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=256, activation='relu')(x)
    x = keras.layers.Dense(units=code_size, activation='linear', name='encoder_embedding')(x)
    return x


def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)

    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)
    # x = keras.layers.BatchNormalization()(x)

    return x


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(keras.layers.Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = preds[:, :, None, :] * y_encoded # this should be broadcasted to N x T_pred x (negative_samples + 1) x code_size
        ret = K.sum(dot_product, axis=-1)

        return ret

    def compute_output_shape(self, input_shape):
        return input_shape[1][:3]


def network_cpc(image_shape, action_dim, include_action, terms, predict_terms, negative_samples, code_size,
                learning_rate, encoder_arch='default', context_network='stack', context_size=32, predict_action=False,
                code_size_action=1, contrastive=True):

    ''' Define the CPC network combining encoder and autoregressive model '''
    if predict_action:
        """
        if predict_action, x_input will be [o_t, o_{t+k+1}]
        y_input is of shape k x (negative + 1) x action_dim
        """
        include_action = False
        terms += 1

    # Set learning phase (https://stackoverflow.com/questions/42969779/keras-error-you-must-feed-a-value-for-placeholder-tensor-bidirectional-1-keras)
    K.set_learning_phase(1)

    # Define encoder model
    encoder_input = keras.layers.Input(image_shape)
    if encoder_arch == 'default':
        encoder_output = network_encoder(encoder_input, code_size)
    elif encoder_arch == 'resnet':
        encoder_output = network_encoder_resnet(encoder_input, code_size)
    encoder_model = keras.models.Model(encoder_input, encoder_output, name='encoder')
    encoder_model.summary()

    if predict_action:
        action_encoder_input = keras.layers.Input((action_dim,))
        # action_encoder_output = keras.layers.Dense(32, activation='relu')(action_encoder_input)
        # action_encoder_output = keras.layers.Dense(code_size_action, activation='linear')(action_encoder_output)
        action_encoder_output = action_encoder_input
        action_encoder_model = keras.models.Model(action_encoder_input, action_encoder_output, name='action_encoder')

    # Define context network
    x_input = keras.layers.Input((terms, image_shape[0], image_shape[1], image_shape[2]), name='x_input')
    action_input = keras.layers.Input((terms, action_dim), name='action_input')
    x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)

    if context_network == 'stack':
        context = keras.layers.Reshape((code_size * terms,))(x_encoded)
        if include_action:
            action_flat = keras.layers.Reshape((action_dim * predict_terms, ))(action_input)
            context = keras.layers.Lambda(lambda x: K.concatenate(x, axis=-1))([context, action_flat])
        context = keras.layers.Dense(512, activation='relu')(context)
        context = keras.layers.Dense(context_size, name='context_output')(context)
    elif context_network == 'rnn':
        context = network_autoregressive(x_encoded)
        if include_action:
            action_flat = keras.layers.Reshape((action_dim * predict_terms,))(action_input)
            context = keras.layers.Lambda(lambda x: K.concatenate(x, axis=-1))([context, action_flat])
            context = keras.layers.Dense(512, activation='relu')(context)
        context = keras.layers.Dense(context_size, name='context_output')(context)

    if include_action:
        context_network = keras.models.Model(inputs=[x_input, action_input], outputs=context, name='context_network')
    else:
        context_network = keras.models.Model(inputs=[x_input], outputs=context, name='context_network')
    context_network.summary()

    # Define rest of the model
    if include_action:
        context_output = context_network([x_input, action_input])
    else:
        context_output = context_network([x_input])
    if predict_action:
        if contrastive:
            preds = network_prediction(context_output, code_size_action, predict_terms)
        else:
            preds = keras.layers.Dense(64, activation='relu')(context_output)
            preds = keras.layers.Dense(16, activation='relu')(preds)
            preds = keras.layers.Dense(predict_terms * action_dim, activation='linear')(preds)
    else:
        preds = network_prediction(context_output, code_size, predict_terms)

    if predict_action:

        if contrastive:
            y_input = keras.layers.Input((predict_terms, (negative_samples + 1), action_dim), name='y_input')
            y_input_flat = keras.layers.Reshape((predict_terms * (negative_samples + 1), action_dim))(y_input)
            y_encoded_flat = keras.layers.TimeDistributed(action_encoder_model)(y_input_flat)
            y_encoded = keras.layers.Reshape((predict_terms, (negative_samples + 1), code_size_action))(y_encoded_flat)

    else:
        y_input = keras.layers.Input((predict_terms, (negative_samples + 1), image_shape[0], image_shape[1], image_shape[2]),
                                     name = 'y_input')
        y_input_flat = keras.layers.Reshape((predict_terms * (negative_samples + 1), *image_shape))(y_input)
        y_encoded_flat = keras.layers.TimeDistributed(encoder_model)(y_input_flat)
        y_encoded = keras.layers.Reshape((predict_terms, (negative_samples + 1), code_size))(y_encoded_flat)

    # Loss
    if contrastive:
        logits = CPCLayer()([preds, y_encoded])

        # Model
        cpc_model = keras.models.Model(inputs=[x_input, action_input, y_input], outputs=logits)

        # Compile model
        cpc_model.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss=cross_entropy_loss,
            metrics=['categorical_accuracy']
        )

    else:
        cpc_model = keras.models.Model(inputs=[x_input], outputs=preds)
        # Compile model
        cpc_model.compile(
            optimizer=keras.optimizers.Adam(lr=learning_rate),
            loss='mean_squared_error',
            metrics=['mean_squared_error']
        )

    cpc_model.summary()

    return cpc_model

class CPCEncoder:
    def __init__(self, path, model=None, image_shape=(64, 64, 3)):
        if model is None:
            self.encoder = keras.models.load_model(path)
        else:
            self.encoder = model
        self.image_shape = image_shape

    def encode(self, imgs):
        if imgs.ndim == len(self.image_shape):
            return self.encoder.predict(imgs[None, ...])[0]
        return self.encoder.predict(imgs)


class CPCContextNet:
    def __init__(self, path, model=None, image_shape=(64, 64, 3)):
        if model is None:
            self.encoder = keras.models.load_model(path)
        else:
            self.encoder = model
        self.image_shape = image_shape

    def encode(self, imgs):
        if imgs.ndim == 1 + len(self.image_shape):
            return self.encoder.predict(imgs[None, ...])[0]
        return self.encoder.predict(imgs)