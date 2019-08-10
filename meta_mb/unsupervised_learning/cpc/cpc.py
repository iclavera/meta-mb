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
                learning_rate, encoder_arch='default', context_network='stack', context_size=32, predict_action=False,
                 grad_penalty=True, lambd=1., rew_loss_weight=0., rew_contrastive=True):

        self.image_shape = image_shape

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
            action_encoder_output = action_encoder_input
            action_encoder_model = keras.models.Model(action_encoder_input, action_encoder_output, name='action_encoder')

        # Define context network
        self.x_input_ph = tf.placeholder(dtype=tf.float32, shape=(None, terms, ) + image_shape)
        x_input = keras.layers.Input(tensor=self.x_input_ph, name='x_input')
        self.action_input_ph = tf.placeholder(dtype=tf.float32, shape=(None, predict_terms, action_dim))
        action_input = keras.layers.Input(tensor=self.action_input_ph, name='action_input')
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


        # ============================== for predicting reward =================================
        if rew_contrastive:
            rew_preds = network_prediction(context_output, 1, predict_terms)
        else:
            rew_preds = keras.layers.Dense(64, activation='relu')(context_output)
            rew_preds = keras.layers.Dense(16, activation='relu')(rew_preds)
            rew_preds = keras.layers.Dense(predict_terms, activation='linear')(rew_preds)

        self.rew_input_ph = tf.placeholder(dtype=tf.float32,
                                            shape=(None, predict_terms, (negative_samples + 1), 1))
        rew_input = keras.layers.Input(tensor=self.rew_input_ph, name='rew_input')
        rew_logits = CPCLayer()([rew_preds, rew_input])

        self.rew_labels_ph = tf.placeholder(dtype=tf.float32, shape=(None, predict_terms, negative_samples + 1))
        self.rew_loss = cross_entropy_loss(self.rew_labels_ph, rew_logits)

        rew_correct_class = tf.argmax(rew_logits, axis=-1)
        rew_predicted_class = tf.argmax(self.rew_labels_ph, axis=-1)
        self.rew_accuracy = tf.reduce_sum(tf.cast(tf.equal(rew_correct_class, rew_predicted_class), tf.int32)) / \
                        tf.size(rew_correct_class)

        # ============================== encoded future states/action ====================================

        if predict_action:
            preds = network_prediction(context_output, action_dim, predict_terms)
        else:
            preds = network_prediction(context_output, code_size, predict_terms)

        if predict_action:
            self.y_input_ph = tf.placeholder(dtype=tf.float32,
                                             shape=(None, predict_terms, (negative_samples + 1), action_dim))
            y_input = keras.layers.Input(tensor=self.y_input_ph, name='y_input')
            y_input_flat = keras.layers.Reshape((predict_terms * (negative_samples + 1), action_dim))(y_input)
            y_encoded_flat = keras.layers.TimeDistributed(action_encoder_model)(y_input_flat)
            y_encoded = keras.layers.Reshape((predict_terms, (negative_samples + 1), action_dim))(y_encoded_flat)

        else:
            self.y_input_ph = tf.placeholder(dtype=tf.float32, shape=(None, predict_terms, negative_samples+1) + image_shape)
            y_input = keras.layers.Input(tensor=self.y_input_ph, name = 'y_input')
            y_input_flat = keras.layers.Reshape((predict_terms * (negative_samples + 1), *image_shape))(y_input)
            y_encoded_flat = keras.layers.TimeDistributed(encoder_model)(y_input_flat)
            y_encoded = keras.layers.Reshape((predict_terms, (negative_samples + 1), code_size))(y_encoded_flat)

        # ======================================== Loss ================================================
        logits = CPCLayer()([preds, y_encoded])

        # Model
        cpc_model = keras.models.Model(inputs=[x_input, action_input, y_input], outputs=logits)
        self.labels_ph = tf.placeholder(dtype=tf.float32, shape=(None, predict_terms, negative_samples + 1))

        self.state_loss = cross_entropy_loss(self.labels_ph, logits)
        if rew_loss_weight > 0:
            self.loss = self.state_loss + rew_loss_weight * self.rew_loss
        else:
            self.loss = self.state_loss


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
        correct_class = tf.argmax(logits, axis=-1)
        predicted_class = tf.argmax(self.labels_ph, axis=-1)
        self.accuracy = tf.reduce_sum(tf.cast(tf.equal(correct_class, predicted_class), tf.int32)) / \
                        tf.size(correct_class)
        self.lr_ph = tf.placeholder(dtype=tf.float32)

        self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr_ph).minimize(self.loss)

        cpc_model.summary()

        self.model = cpc_model
        self.encoder = encoder_model

    def encode(self, imgs):
        if imgs.ndim == len(self.image_shape):
            return self.encoder.predict(imgs[None, ...])[0]
        return self.encoder.predict(imgs)

    def train_step(self, inputs, labels, lr=1e-3, train=True):
        if train:
            loss, acc, rew_acc, _ = tf.get_default_session().run([self.loss, self.accuracy, self.rew_accuracy, self.train_op],
                                            feed_dict={self.x_input_ph: inputs[0],
                                                       self.action_input_ph:inputs[1],
                                                       self.y_input_ph:inputs[2],
                                                       self.rew_input_ph: inputs[3],
                                                       self.rew_labels_ph: labels,
                                                       self.labels_ph: labels,
                                                       self.lr_ph: lr})
        else:
            loss, acc, rew_acc = tf.get_default_session().run([self.loss, self.accuracy, self.rew_accuracy],
                                                    feed_dict={self.x_input_ph: inputs[0],
                                                               self.action_input_ph: inputs[1],
                                                               self.y_input_ph: inputs[2],
                                                               self.rew_input_ph: inputs[3],
                                                               self.rew_labels_ph: labels,
                                                               self.labels_ph: labels})
        return loss, acc, rew_acc

    def fit_generator(self, generator, steps_per_epoch, validation_data, validation_steps, epochs, patience=3):
        lr = 1e-3
        non_decrease_count = 0
        target_loss = 100.
        for i in range(epochs):
            # logging
            logs = {'loss_cpc': [], 'val_loss_cpc': [], 'acc_cpc':[], 'val_acc_cpc': [], 'rew_acc_cpc':[], 'val_rew_acc_cpc': []}

            print("Epoch %d" % i)
            train_pb = Progbar(steps_per_epoch)
            for k in range(steps_per_epoch):
                loss, acc, rew_acc = self.train_step(*generator.next(), lr=lr)
                train_pb.add(1, values=[('loss_cpc', loss), ('acc_cpc', acc), ('rew_acc_cpc', rew_acc)])
                logs['loss_cpc'].append(loss); logs['acc_cpc'].append(acc); logs['rew_acc_cpc'].append(rew_acc)
            val_pb = Progbar(validation_steps)
            for k in range(validation_steps):
                val_loss, val_acc, val_rew_acc = self.train_step(*validation_data.next(), train=False)
                val_pb.add(1, values=[('val_loss_cpc', val_loss), ('val_acc_cpc', val_acc), ('val_rew_acc_cpc', val_rew_acc)])
                logs['val_loss_cpc'].append(val_loss); logs['val_acc_cpc'].append(val_acc); logs['val_rew_acc_cpc'].append(val_rew_acc)

            cur_val_loss = np.mean(logs['val_loss_cpc'])
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