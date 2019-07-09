import numpy as np
import keras
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.cpc.cpc import network_cpc
from meta_mb.unsupervised_learning.cpc.data_utils import CPCDataGenerator, plot_seq
from meta_mb.unsupervised_learning.cpc.train_script.utils import collect_img
from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, make_periodic_lr, res_block, cross_entropy_loss
from meta_mb.unsupervised_learning.cpc.train_script.utils import RepeatRandom

os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"



def train_with_exploratory_policy(raw_env, policy, exp_name, negative_samples, batch_size=32, code_size=32, epochs=30,
                                  image_shape=(64, 64, 3),  num_rollouts=32, max_path_length=16,
                                  encoder_arch='default', context_network='stack', lr=1e-3, terms=1):
    img_seqs = collect_img(raw_env, policy, num_rollouts=num_rollouts, max_path_length=max_path_length, image_shape=image_shape)
    train_seq, val_seq = train_test_split(img_seqs)
    train_data = CPCDataGenerator(train_seq, batch_size, terms=terms, negative_samples=negative_samples, predict_terms=1)
    validation_data = CPCDataGenerator(val_seq, batch_size, terms=terms, negative_samples=negative_samples, predict_terms=1)
    #
    #
    # for (x, y), labels in train_data:
    #     plot_seq(x, y, labels, name='ip-seq')
    #     break
    # import pdb; pdb.set_trace()

    # Create the model
    model = network_cpc(image_shape=image_shape, terms=terms, predict_terms=1, negative_samples=negative_samples,
                        code_size=code_size, learning_rate=lr, encoder_arch=encoder_arch, context_network=context_network)

    # Callbacks
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    callbacks = [#keras.callbacks.LearningRateScheduler(make_periodic_lr([5e-2, 5e-3, 5e-4]), verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-5, verbose=1, min_delta=0.001),
                 SaveEncoder(output_dir),
                 keras.callbacks.CSVLogger(os.path.join(output_dir, 'cpc.log'))]

    # Train the model
    model.fit_generator(
        generator=train_data,
        steps_per_epoch=len(train_data),
        validation_data=validation_data,
        validation_steps=len(validation_data),
        epochs=epochs,
        verbose=1,
        callbacks=callbacks
    )






if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train CPC model')
    parser.add_argument('negative_samples', type=int, help='number of negative samples')
    parser.add_argument('terms', type=int, help='number of time steps in the history')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train the model for')
    parser.add_argument('--run_suffix', type=str, default='')
    parser.add_argument('--context_network', type=str, default='stack')
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()
    from meta_mb.envs.mujoco.point_pos import PointEnv
    raw_env = PointEnv()
    # raw_env = InvertedPendulumEnv()

    normalized_env = NormalizedEnv(raw_env)
    policy = RepeatRandom(2, 2, repeat=3)

    exp_name = 'ballsize=0.2-neg-%d%s' % (args.negative_samples, args.run_suffix)

    train_with_exploratory_policy(raw_env, policy, exp_name, args.negative_samples, num_rollouts=512, batch_size=32,
                                  context_network=args.context_network, epochs=args.epochs, lr=args.lr, terms=args.terms)



