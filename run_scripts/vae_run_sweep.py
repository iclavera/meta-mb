import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.normalized_env import normalize
from meta_mb.logger import logger
from meta_mb.envs.pr2.pr2_env import PR2ReacherEnv
from meta_mb.unsupervised_learning.vae import VAE
from meta_mb.unsupervised_learning.utils import Random, collect_img

INSTANCE_TYPE = 'c4.2xlarge'
os.environ["CUDA_VISIBLE_DEVICES"]="1"
EXP_NAME = 'envs_vae'

def train_with_exploration(**kwargs):
    exp_name = 'vae-%s-%s' % (kwargs['env'], kwargs['seed'])

    logger.configure(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', EXP_NAME, exp_name))

    sess = tf.InteractiveSession()

    env = kwargs['env']()
    env.observation_space=14
    policy = Random(2, 2, repeat=2)
    img_seqs, _ = collect_img(env,
                              policy,
                              num_rollouts=kwargs['num_rollouts'],
                              max_path_length=kwargs['max_path_length'],
                              image_shape=kwargs['img_shape'])
    imgs = np.concatenate(img_seqs)

    train_imgs, val_imgs = train_test_split(imgs)
    n_train = len(train_imgs)

    model = VAE(latent_dim=kwargs['code_size'],
                lr=kwargs['lr'],
                decoder_bernoulli=kwargs['bnl_decoder'])

    vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(vars))

    for i in range(kwargs['epochs'] * n_train // kwargs['batch_size']):
        x_train = train_imgs[np.random.choice(n_train, size=kwargs['batch_size'], replace=False)].astype(np.float32)
        model.train_step(x_train, beta=kwargs['beta'])

    saver = tf.train.Saver()
    saver.save(sess, os.path.join(logger.get_dir(), 'vae'))
    sess.close()

if __name__ == "__main__":

    config = {
        'seed': ['1'],

        # env config
        'env': [PR2ReacherEnv],
        'img_shape': [(64, 64, 3)],
        'max_path_length': [200],
        'num_rollouts': [64],

        # for point mass
        'ptsize': [2],
        'distractor': [False],

        # vae
        'bnl_decoder': [True],
        'beta': [1],
        'code_size': [8],
        'latent_dim': [8],
        'time_steps': [2, 4],


        # training config
        'epochs': [50],
        'lr': [1e-3],
        'batch_size': [32],

    }

    run_sweep(train_with_exploration, config, EXP_NAME, INSTANCE_TYPE)   
