import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.envs_util import make_env
from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.vae import VAE
from meta_mb.unsupervised_learning.cpc.train_script.utils import collect_img, RepeatRandom, init_uninit_vars

INSTANCE_TYPE = 'c4.2xlarge'
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
EXP_NAME = 'envs_vae'


def train_with_exploratory_policy(**config):

    exp_name = 'vae-%s-code%d-%s' % (config['env'], config['code_size'], config['run_suffix'])

    logger.configure(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', EXP_NAME, exp_name))

    sess = tf.InteractiveSession()


    raw_env, max_path_length = make_env(config['env'])
    policy = RepeatRandom(2, 2, repeat=3)
    img_seqs, _ = collect_img(raw_env, policy, num_rollouts=config['num_rollouts'], max_path_length=max_path_length,
                           image_shape=config['image_shape'])
    imgs = np.concatenate(img_seqs)


    train_imgs, val_imgs = train_test_split(imgs)
    n_train = len(train_imgs)


    model = VAE(latent_dim=config['code_size'], lr=config['lr'], decoder_bernoulli=config['bnl_decoder'])
    init_uninit_vars(sess)

    for i in range(config['epochs'] * n_train // config['batch_size']):
        x_train = train_imgs[np.random.choice(n_train, size=config['batch_size'], replace=False)].astype(np.float32)
        model.train_step(x_train, beta=config['beta'])

    saver = tf.train.Saver()

    saver.save(sess, os.path.join(logger.get_dir(), 'vae'))

    sess.close()




if __name__ == "__main__":

    config = {
        'run_suffix': ['1'],

        # env config
        'env': ['cartpole_balance', 'cartpole_swingup', 'reacher_easy', 'cheetah'],
        'image_shape': [(64, 64, 3)],
        'num_rollouts': [64],

        # for point mass
        'ptsize': [2],
        'distractor': [False],

        # vae
        'bnl_decoder': [True],
        'beta': [1],
        'code_size': [8],


        # training config
        'epochs': [50],
        'lr': [1e-3],
        'batch_size': [32],

    }

    run_sweep(train_with_exploratory_policy, config, EXP_NAME, INSTANCE_TYPE)

