import os
import keras
from keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Progbar

from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, res_block, make_periodic_lr, cross_entropy_loss
from meta_mb.logger import logger

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
    # x = keras.layers.LeakyReLU()(x)
    # x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, activation='linear')(x)
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


class CPC:
    def __init__(self, image_shape, action_dim, include_action, terms, predict_terms, negative_samples, code_size,
                 encoder_arch='default', context_network='stack', context_size=32, grad_penalty=True, lambd=1.,
                 rew_loss_weight=0., rew_contrastive=True, action_loss_weight=0., action_contrastive=False):

        self.image_shape = image_shape

        ''' Define the CPC network combining encoder and autoregressive model '''
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

        # Define context network
        self.x_input_ph = tf.placeholder(dtype=tf.float32, shape=(None, terms, ) + image_shape)
        x_input = keras.layers.Input(tensor=self.x_input_ph, name='x_input')
        self.action_input_ph =  tf.placeholder(dtype=tf.float32,
                                                          shape=(None, predict_terms, (negative_samples + 1), action_dim))
        action_input = keras.layers.Input(tensor=self.action_input_ph, name='action_input')
        action_input_context = keras.layers.Input(tensor=self.action_input_ph[:, :, 0, :], name='action_input')
        x_encoded = keras.layers.TimeDistributed(encoder_model)(x_input)

        if context_network == 'stack':
            context = keras.layers.Reshape((code_size * terms,))(x_encoded)
            context = keras.layers.Dense(512, activation='relu')(context)
            context_noaction = keras.layers.Dense(context_size, activation='linear')(context)
            if include_action:
                action_flat = keras.layers.Reshape((action_dim * predict_terms, ))(action_input_context)
                context = keras.layers.Lambda(lambda x: K.concatenate(x, axis=-1))([context_noaction, action_flat])
            context = keras.layers.Dense(context_size, name='context_output', activation='linear')(context)
        elif context_network == 'rnn':
            context = network_autoregressive(x_encoded)
            context_noaction = keras.layers.Dense(context_size, activation='linear')(context)
            if include_action:
                action_flat = keras.layers.Reshape((action_dim * predict_terms,))(action_input_context)
                context = keras.layers.Lambda(lambda x: K.concatenate(x, axis=-1))([context_noaction, action_flat])
                context = keras.layers.Dense(512, activation='relu')(context)
            context = keras.layers.Dense(context_size, name='context_output')(context)

        if include_action:
            context_network = keras.models.Model(inputs=[x_input, action_input_context], outputs=context, name='context_network')
            context_output = context_network([x_input, action_input])
        else:
            context_network = keras.models.Model(inputs=[x_input], outputs=context, name='context_network')
            context_output = context_network([x_input])
        context_network.summary()

        # =============================== for predicting next latent ==========================================

        preds = network_prediction(context_output, code_size, predict_terms)
        self.y_input_ph = tf.placeholder(dtype=tf.float32,
                                         shape=(None, predict_terms, negative_samples + 1) + image_shape)
        y_input = keras.layers.Input(tensor=self.y_input_ph, name='y_input')
        y_input_flat = keras.layers.Reshape((predict_terms * (negative_samples + 1), *image_shape))(y_input)
        y_encoded_flat = keras.layers.TimeDistributed(encoder_model)(y_input_flat)
        y_encoded = keras.layers.Reshape((predict_terms, (negative_samples + 1), code_size))(y_encoded_flat)

        logits = CPCLayer()([preds, y_encoded])

        # Model
        cpc_model = keras.models.Model(inputs=[x_input, action_input, y_input], outputs=logits)
        self.labels_ph = tf.placeholder(dtype=tf.float32, shape=(None, predict_terms, negative_samples + 1))
        self.state_loss = cross_entropy_loss(self.labels_ph, logits)

        correct_class = tf.argmax(logits, axis=-1)
        predicted_class = tf.argmax(self.labels_ph, axis=-1)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(correct_class, predicted_class), tf.int32)) / \
                        tf.size(correct_class)

        # ============================== for predicting reward =================================
        if rew_contrastive:
            rew_preds = network_prediction(context_output, 1, predict_terms)
        else:
            rew_preds = keras.layers.Dense(256, activation='relu')(context_output)
            rew_preds = keras.layers.Dense(256, activation='relu')(rew_preds)
            rew_preds = keras.layers.Dense(predict_terms, activation='linear')(rew_preds)
            rew_preds = keras.layers.Reshape((predict_terms, 1))(rew_preds)

        self.rew_input_ph = tf.placeholder(dtype=tf.float32,
                                            shape=(None, predict_terms, (negative_samples + 1), 1))
        rew_input = keras.layers.Input(tensor=self.rew_input_ph, name='rew_input')
        rew_logits = CPCLayer()([rew_preds, rew_input])

        if rew_contrastive:
            self.rew_loss = cross_entropy_loss(self.labels_ph, rew_logits)
        else:
            positive_ys = tf.reshape(self.rew_input_ph[:, :, 0, :], (-1, predict_terms, 1))
            self.rew_loss = tf.reduce_mean(tf.square(positive_ys - rew_preds))

        rew_correct_class = tf.argmax(rew_logits, axis=-1)
        rew_predicted_class = tf.argmax(self.labels_ph, axis=-1)
        self.rew_accuracy = tf.reduce_sum(tf.cast(tf.equal(rew_correct_class, rew_predicted_class), tf.int32)) / \
                        tf.size(rew_correct_class)

        # ============================== for predicting actions taken ====================================
        # only predict one action, one observation ahead
        predict_terms_temp = 1
        assert predict_terms_temp <= predict_terms
        action_pred_input = keras.layers.Lambda(lambda x: K.concatenate(x, axis=-1))\
            ([context_noaction, tf.reshape(y_encoded[:, :predict_terms_temp, 0], (-1, predict_terms_temp * code_size))])
        if action_contrastive:
            action_preds = network_prediction(action_pred_input, action_dim, predict_terms_temp)
        else:
            action_preds = keras.layers.Dense(256, activation='relu')(action_pred_input)
            action_preds = keras.layers.Dense(256, activation='relu')(action_preds)
            action_preds = keras.layers.Dense(predict_terms_temp * action_dim, activation='linear')(action_preds)
            action_preds = keras.layers.Reshape((predict_terms_temp, action_dim))(action_preds)

        action_logits = CPCLayer()([action_preds, action_input[:, :predict_terms_temp]])

        if action_contrastive:
            self.action_loss = cross_entropy_loss(self.labels_ph[:, :predict_terms_temp], action_logits)
        else:
            positive_ys = tf.reshape(self.action_input_ph[:, :predict_terms_temp, 0], (-1, predict_terms_temp, action_dim))
            self.action_loss = tf.reduce_mean(tf.square(positive_ys - action_preds))

        action_correct_class = tf.argmax(action_logits, axis=-1)
        action_predicted_class = tf.argmax(self.labels_ph[:, :predict_terms_temp], axis=-1)
        self.action_accuracy = tf.reduce_sum(tf.cast(tf.equal(action_correct_class, action_predicted_class), tf.int32)) / \
                            tf.size(action_correct_class)

        # ======================================== Loss ================================================
        self.loss = self.state_loss
        if rew_loss_weight > 0:
            self.loss += rew_loss_weight * self.rew_loss
        if action_loss_weight > 0:
            self.loss += action_loss_weight * self.action_loss

        # ==================================== Gradient Penalty ==========================================
        self.gpenalty = tf.constant(0, dtype=tf.float32)
        if grad_penalty:
            for i in range(predict_terms):
                for j in range(negative_samples + 1):
                    grad = tf.gradients(logits[:, i, j], [x_input, y_input])
                    grad_concat = tf.concat([tf.contrib.layers.flatten(grad[0]),
                                             tf.contrib.layers.flatten(grad[1][:, i, j])],
                                            axis=-1)
                    self.gpenalty += tf.reduce_mean(tf.pow(tf.norm(grad_concat, axis=-1) - 1, 2))

            self.loss += lambd * self.gpenalty

        # =================================== Train op and logging =======================================
        self.lr_ph = tf.placeholder(dtype=tf.float32)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_ph).minimize(self.loss)

        cpc_model.summary()

        self.encoder = encoder_model

    def encode(self, imgs):
        if imgs.ndim == len(self.image_shape):
            return self.encoder.predict(imgs[None, ...])[0]
        return self.encoder.predict(imgs)

    def train_step(self, inputs, labels, lr=1e-3, train=True):
        if train:
            loss, acc, rew_loss, rew_acc, action_loss, action_acc, _ = tf.get_default_session().run([self.loss, self.accuracy,
                                                                      self.rew_loss, self.rew_accuracy,
                                                                      self.action_loss, self.action_accuracy,
                                                                      self.train_op],
                                            feed_dict={self.x_input_ph: inputs[0],
                                                       self.action_input_ph:inputs[1],
                                                       self.y_input_ph:inputs[2],
                                                       self.rew_input_ph: inputs[3],
                                                       self.labels_ph: labels,
                                                       self.lr_ph: lr})
        else:
            loss, acc, rew_loss, rew_acc, action_loss, action_acc = tf.get_default_session().run([self.loss, self.accuracy,
                                                               self.rew_loss, self.rew_accuracy,
                                                               self.action_loss, self.action_accuracy,],
                                                    feed_dict={self.x_input_ph: inputs[0],
                                                               self.action_input_ph: inputs[1],
                                                               self.y_input_ph: inputs[2],
                                                               self.rew_input_ph: inputs[3],
                                                               self.labels_ph: labels})
        return loss, acc, rew_loss, rew_acc, action_loss, action_acc

    def fit_generator(self, generator, steps_per_epoch, validation_data, validation_steps, epochs, patience=3):
        lr = 1e-3
        non_decrease_count = 0
        target_loss = 100.
        for i in range(epochs):
            # logging
            logs = {'loss': [], 'val_loss': [],
                    'acc':[], 'val_acc': [],
                    'rew_loss':[], 'val_rew_loss': [],
                    'rew_acc':[], 'val_rew_acc': [],
                    'action_loss': [], 'val_action_loss': [],
                    'action_acc': [], 'val_action_acc': [],
                    }

            print("Epoch %d" % i)
            train_pb = Progbar(steps_per_epoch)
            for k in range(steps_per_epoch):
                loss, acc, rew_loss, rew_acc, action_loss, action_acc = self.train_step(*generator.next(), lr=lr)
                train_pb.add(1, values=[('loss', loss), ('acc', acc),
                                        ('rew_loss', rew_loss), ('rew_acc', rew_acc),
                                        ('action_loss', action_loss), ('action_acc', action_acc),])
                logs['loss'].append(loss); logs['acc'].append(acc);
                logs['rew_loss'].append(rew_loss); logs['rew_acc'].append(rew_acc)
                logs['action_loss'].append(action_loss);logs['action_acc'].append(action_acc)
            val_pb = Progbar(validation_steps)
            for k in range(validation_steps):
                loss, acc, rew_loss, rew_acc, action_loss, action_acc = self.train_step(*validation_data.next(), train=False)
                val_pb.add(1, values=[('val_loss', loss), ('val_acc', acc),
                                        ('val_rew_loss', rew_loss), ('val_rew_acc', rew_acc),
                                        ('val_action_loss', action_loss), ('val_action_acc', action_acc), ])
                logs['val_loss'].append(loss); logs['val_acc'].append(acc);
                logs['val_rew_loss'].append(rew_loss); logs['val_rew_acc'].append(rew_acc)
                logs['val_action_loss'].append(action_loss); logs['val_action_acc'].append(action_acc)

            cur_val_loss = np.mean(logs['val_loss'])
            logs = {k : np.mean(v) for k, v in logs.items()}
            # log to the file
            logger.logkvs(logs)
            logger.dump_tabular()

            # learning rate adjustment
            if cur_val_loss < target_loss:
                target_loss = cur_val_loss
                non_decrease_count = 0
            else:
                non_decrease_count += 1
                if non_decrease_count >= patience and lr > 1e-5:
                    lr /= 3
                    non_decrease_count = 0
                    print("reducing the lr to %f" % lr)

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