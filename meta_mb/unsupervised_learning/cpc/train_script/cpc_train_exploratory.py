import numpy as np
import json
import keras
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time

from experiment_utils.run_sweep import run_sweep
from  meta_mb.envs.envs_util import make_env
from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.logger import logger
from meta_mb.utils.utils import ClassEncoder
from meta_mb.unsupervised_learning.cpc.cpc import network_cpc
from meta_mb.unsupervised_learning.cpc.data_utils import CPCDataGenerator, plot_seq
from meta_mb.unsupervised_learning.cpc.train_script.utils import collect_img
from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, make_periodic_lr, res_block, cross_entropy_loss
from meta_mb.unsupervised_learning.cpc.train_script.utils import RepeatRandom

# os.environ["CUDA_VISIBLE_DEVICES"]="0, 1"
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'envs_action'


def train_with_exploratory_policy(**config):
    exp_name = '%s-neg%d-hist%d-fut%d-code%d-withaction%r-%s' % (config['env'], config['negative_samples'], config['terms'],
                                                                 config['predict_terms'], config['code_size'],
                                                                 config['include_action'], config['run_suffix'])

    raw_env, max_path_length = make_env(config['env'])
    policy = RepeatRandom(2, 2, repeat=3)
    negative_same_traj = 0
    if config['env'] == 'reacher_easy':
        negative_same_traj = config['negative_samples'] // 3

    img_seqs, action_seqs = collect_img(raw_env, policy, num_rollouts=config['num_rollouts'],
                                        max_path_length=max_path_length, image_shape=config['image_shape'])
    train_img, val_img, train_action, val_action = train_test_split(img_seqs, action_seqs)
    train_data = CPCDataGenerator(train_img, train_action, config['batch_size'], terms=config['terms'], negative_samples=config['negative_samples'],
                                  predict_terms=config['predict_terms'], negative_same_traj=negative_same_traj)
    validation_data = CPCDataGenerator(val_img, val_action, config['batch_size'], terms=config['terms'], negative_samples=config['negative_samples'],
                                       predict_terms=config['predict_terms'], negative_same_traj=negative_same_traj)

    # train_data.next()
    # for (x, y), labels in train_data:
    #     plot_seq(x[0], y, labels, name='reacher-seq')
    #     break
    # import pdb; pdb.set_trace()


    # Create the model
    model = network_cpc(image_shape=config['image_shape'], action_dim=raw_env.action_space.shape[0], include_action=config['include_action'],
                        terms=config['terms'], predict_terms=config['predict_terms'], negative_samples=config['negative_samples'],
                        code_size=config['code_size'], learning_rate=config['lr'], encoder_arch=config['encoder_arch'], context_network=config['context_network'])

    # Callbacks
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', EXP_NAME, exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json.dump(config, open(output_dir + '/cpc_params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
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
        epochs=config['epochs'],
        verbose=1,
        callbacks=callbacks
    )






if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('EXP_NAME', type=str)
    #
    # args = parser.parse_args()

    config = {
        'run_suffix': [1],

        # env config
        'env': ['cartpole_balance', 'cartpole_swingup', 'reacher_easy'],
        'image_shape': [(64, 64, 3)],
        'num_rollouts': [512],

        # for point mass
        'ptsize': [2],
        'distractor': [False],


        # cpc config
        'terms': [3],
        'predict_terms': [3],
        'encoder_arch': ['default'],
        'context_network': ['stack'],
        'code_size': [8],
        'negative_samples': [10],
        'include_action': [True],


        # training config
        'epochs': [30],
        'lr': [1e-3],
        'batch_size': [32],

    }

    run_sweep(train_with_exploratory_policy, config, EXP_NAME, INSTANCE_TYPE)





