import keras
import os
import tensorflow as tf

class SaveEncoder(keras.callbacks.Callback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_train_begin(self, logs={}):
        self.max_acc = -1.
        self.encoder = self.model.layers[1].layer

    def on_epoch_end(self, epoch, logs={}):
        cur_acc = logs.get('val_categorical_accuracy')
        if cur_acc > self.max_acc:
            print("saving model with accuracy %f" % cur_acc)
            self.max_acc = cur_acc
            self.encoder.save(os.path.join(self.output_dir, 'encoder.h5'))
            self.model.save(os.path.join(self.output_dir, 'cpc.h5'))

def make_periodic_lr(lr_schedule):
    def periodic_lr(epoch, lr):
        return lr_schedule[epoch % len(lr_schedule)]
    return periodic_lr


def res_block(input, input_channels):
    output = keras.layers.Conv2D(filters=input_channels // 2, kernel_size=3, strides=1, activation='relu', padding='same')(input)
    output = keras.layers.Conv2D(filters=input_channels, kernel_size=3, strides=1, activation='linear', padding='same')(output)
    output = keras.layers.Add()([output, input])
    output = keras.layers.ReLU()(output)

    return output

def cross_entropy_loss(y_true, y_pred):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred, dim=2)
    return tf.reduce_mean(loss)