import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.vae import VAE
from meta_mb.unsupervised_learning.cpc.train_script.utils import collect_img, RepeatRandom, init_uninit_vars

os.environ["CUDA_VISIBLE_DEVICES"]="1"



def train_with_exploratory_policy(raw_env, policy, exp_name, batch_size=32, code_size=32, epochs=30,
                                  image_shape=(64, 64, 3),  num_rollouts=32, max_path_length=16,
                                  lr=1e-3, beta=1):

    logger.configure(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', exp_name))

    sess = tf.InteractiveSession()

    img_seqs = collect_img(raw_env, policy, num_rollouts=num_rollouts, max_path_length=max_path_length, image_shape=image_shape)
    imgs = np.concatenate(img_seqs)


    train_imgs, val_imgs = train_test_split(imgs)
    n_train = len(train_imgs)

    # sess = tf.Session()
    # with sess.as_default():
    #     pass

    model = VAE(latent_dim=code_size, lr=lr)
    init_uninit_vars(sess)

    for i in range(epochs * n_train // batch_size):
        x_train = train_imgs[np.random.choice(n_train, size=batch_size, replace=False)]
        model.train_step(x_train, beta=beta)

    saver = tf.train.Saver()

    saver.save(sess, os.path.join(logger.get_dir(), 'vae'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train the model for')
    parser.add_argument('--run_suffix', type=str, default='')
    # parser.add_argument('--encoder_arch', type=str, default='default')
    parser.add_argument('--context_network', type=str, default='stack')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1)

    args = parser.parse_args()

    raw_env = PointEnv()
    normalized_env = NormalizedEnv(raw_env)
    policy = RepeatRandom(2, 2, repeat=3)

    exp_name = 'vae_ballsize=0.4'

    train_with_exploratory_policy(raw_env, policy, exp_name, num_rollouts=1024, batch_size=32,
                                  epochs=args.epochs, lr=args.lr, beta=args.beta)



