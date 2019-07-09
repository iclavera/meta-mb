import keras
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

from meta_mb.envs.img_wrapper_env import ImgWrapperEnv
from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv
from meta_mb.logger import logger
from meta_mb.policies.base import Policy
from meta_mb.samplers.base import BaseSampler
from meta_mb.unsupervised_learning.cpc.cpc import network_cpc
from meta_mb.unsupervised_learning.cpc.data_utils import CPCDataGenerator, plot_seq
from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, make_periodic_lr, res_block, cross_entropy_loss
from meta_mb.unsupervised_learning.cpc.train_script.utils import RepeatRandom, init_uninit_vars
from meta_mb.unsupervised_learning.vae import VAE
from meta_mb.utils import Serializable


def collect_img_and_truestate(raw_env, policy, num_rollouts=1024, max_path_length=16, plot=False):
    env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=1)
    sampler = BaseSampler(env, policy, num_rollouts, max_path_length, ignore_done=True)

    # Sampling data from the given exploratory policy and create a data iterator
    env_paths = sampler.obtain_samples(log=True, log_prefix='Data-EnvSampler-', random=True, )
    img_seqs = np.concatenate([path['observations'] for path in env_paths])  # (N x T) x (img_shape)
    true_state_seqs = np.concatenate([path['env_infos']['true_state'] for path in env_paths]) # (N x T) x 2
    if plot:
        counter = 1
        plt.figure(figsize=(64, 16))
        for i in range(4):
            for img, state in zip(img_seqs[i * 16 : i * 16 + 16], true_state_seqs[i * 16 : i * 16 + 16]):
                ax = plt.subplot(4, 16, counter)
                plt.imshow(img)
                ax.set_title(state)
                counter += 1
                plt.savefig('img_truestate.png')

    return img_seqs, true_state_seqs

def build_model(encoder_path, img_shape, state_shape, lr, freeze_encoding=True):
    encoder = keras.models.load_model(encoder_path)

    # Freeze weights
    if freeze_encoding:
        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False

    x_input = keras.layers.Input(img_shape)
    x = encoder(x_input)
    x = keras.layers.Dense(units=128, activation='linear')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=state_shape, activation='linear')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    model.summary()

    return model

def build_model_vae(state_shape, lr, code_size=32):

    x_input = keras.layers.Input((code_size,))
    x = keras.layers.Dense(units=128, activation='linear')(x_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU()(x)
    x = keras.layers.Dense(units=state_shape, activation='linear')(x)

    # Model
    model = keras.models.Model(inputs=x_input, outputs=x)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )
    model.summary()

    return model


def train_regression(raw_env, policy, encoder_path, epochs, batch_size, lr, img_shape, state_shape,
                     freeze_encoding=True, num_rollouts=512, max_path_length=16, save_name='supervised', vae=False, code_size=32):

    x, y = collect_img_and_truestate(raw_env, policy, num_rollouts=num_rollouts, max_path_length=max_path_length)
    if vae:
        sess = tf.InteractiveSession()
        encoder = VAE(latent_dim=code_size, lr=lr)
        saver = tf.train.Saver()
        saver.restore(sess, os.path.join(encoder_path, 'vae'))
        x = encoder.encode(x)

        model = build_model_vae(state_shape, lr, code_size=32)
    else:
        model = build_model(os.path.join(encoder_path, 'encoder.h5'), img_shape, state_shape, lr,
                            freeze_encoding=freeze_encoding)

    callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1 / 3, patience=2, min_lr=1e-5, verbose=1, min_delta=0.01),
                 keras.callbacks.ModelCheckpoint(os.path.join(encoder_path, save_name + '.h5'), monitor='val_loss', verbose=1, save_best_only=True),
                 keras.callbacks.CSVLogger(os.path.join(encoder_path, save_name + '.log'))]

    # Trains the model
    model.fit(
        x, y,
        batch_size=batch_size,
        validation_split=0.2,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )

def print_prediction(model_path, raw_env, policy, num_rollouts=16, max_path_length=16):
    reg_model = keras.models.load_model(model_path)
    x, y = collect_img_and_truestate(raw_env, policy, num_rollouts=num_rollouts, max_path_length=max_path_length)
    y_pred = reg_model.predict(x)
    for pred, true in zip(y_pred, y):
        print(pred, true)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train CPC model')
    parser.add_argument('env_name', type=str)
    parser.add_argument('exp_name', type=str, help='name of the experiment: '
                                                   'data will be loaded from meta_mb/unsupervised_learning/cpc/data/$exp_name')
    parser.add_argument('num_rollout', type=int, help='number of rollouts to use during training')
    parser.add_argument('--epoch', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--vae', action='store_true', help='if a vae is used instead of CPC')
    parser.add_argument('--e2e', action='store_true', help='not freeze the encoding and train end to end')

    args = parser.parse_args()

    if args.env_name == 'ip':
        raw_env = InvertedPendulumEnv()
        state_shape = 4
    elif args.env_name == 'pt':
        raw_env = PointEnv()
        state_shape = 2

    policy = RepeatRandom(2, 2, repeat=3)
    encoder_path = os.path.join('meta_mb/unsupervised_learning/cpc/data', args.exp_name)

    #
    # collect_img_and_truestate(raw_env, policy, num_rollouts=8, plot=True)
    # import pdb; pdb.set_trace()
    freeze_encoding = not args.e2e
    if freeze_encoding:
        save_name='supervised%d' % args.num_rollout
    else:
        save_name = 'supervised_unfrozen%d' % args.num_rollout

    train_regression(raw_env, policy, encoder_path, epochs=args.epoch, batch_size=64, lr=1e-3, img_shape=(64, 64, 3),
                     state_shape=state_shape, num_rollouts=args.num_rollout, freeze_encoding=freeze_encoding, save_name=save_name,
                     vae=args.vae, code_size=32)

    print_prediction(os.path.join(encoder_path, save_name + '.h5'), raw_env, policy, num_rollouts=2, max_path_length=16)
