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
from meta_mb.envs.mujoco.inverted_pendulum_env import  InvertedPendulumEnv
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.vae import VAE
from meta_mb.unsupervised_learning.cpc.train_script.utils import collect_img, RepeatRandom, init_uninit_vars

os.environ["CUDA_VISIBLE_DEVICES"]="1"



def train_with_exploratory_policy(raw_env, policy, exp_name, batch_size=32, code_size=32, epochs=30,
                                  image_shape=(64, 64, 3),  num_rollouts=32, max_path_length=16,
                                  lr=1e-3, beta=1, bnl_decoder=False):

    logger.configure(dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', exp_name))

    sess = tf.InteractiveSession()

    img_seqs = collect_img(raw_env, policy, num_rollouts=num_rollouts, max_path_length=max_path_length, image_shape=image_shape)
    imgs = np.concatenate(img_seqs)


    train_imgs, val_imgs = train_test_split(imgs)
    n_train = len(train_imgs)

    # sess = tf.Session()
    # with sess.as_default():
    #     pass

    model = VAE(latent_dim=code_size, lr=lr, decoder_bernoulli=bnl_decoder)
    init_uninit_vars(sess)

    for i in range(epochs * n_train // batch_size):
        x_train = train_imgs[np.random.choice(n_train, size=batch_size, replace=False)]
        model.train_step(x_train, beta=beta)

    saver = tf.train.Saver()

    saver.save(sess, os.path.join(logger.get_dir(), 'vae'))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train VAE')
    parser.add_argument('--env', type=str)
    parser.add_argument('--bnl_decoder', action='store_true')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train the model for')
    parser.add_argument('--run_suffix', type=str, default='')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--ptsize', type=int, default=2)
    parser.add_argument('--distractor', action='store_true')
    parser.add_argument('--code_size', type=int, default=32)


    args = parser.parse_args()

    if args.env == 'pt':
        if args.distractor:
            from meta_mb.envs.mujoco.point_pos_distractor import PointEnv
            raw_env = PointEnv()
        else:
            from meta_mb.envs.mujoco.point_pos import PointEnv
            raw_env = PointEnv(ptsize=args.ptsize)
    elif args.env == 'ip':
        raw_env = InvertedPendulumEnv()
    else:
        raise NotImplementedError

    normalized_env = NormalizedEnv(raw_env)
    policy = RepeatRandom(2, 2, repeat=3)

    # exp_name = 'vae-ptsize=%d-codesize=%d%s' % (args.ptsize, args.code_size, args.run_suffix) if not args.distractor else 'vae-distractor%s' % args.run_suffix
    exp_name = 'ip%s' % args.run_suffix

    train_with_exploratory_policy(raw_env, policy, exp_name, num_rollouts=1024, batch_size=32,
                                  epochs=args.epochs, lr=args.lr, beta=args.beta, bnl_decoder=args.bnl_decoder, code_size=args.code_size)


