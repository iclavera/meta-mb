import numpy as np
import keras
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time


from meta_mb.envs.img_wrapper_env import ImgWrapperEnv
from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.logger import logger
from meta_mb.policies.base import Policy
from meta_mb.samplers.base import BaseSampler
from meta_mb.unsupervised_learning.cpc.cpc import network_cpc
from meta_mb.unsupervised_learning.cpc.data_utils import CPCDataGenerator, plot_seq
from meta_mb.unsupervised_learning.cpc.training_utils import SaveEncoder, make_periodic_lr, res_block, cross_entropy_loss
from meta_mb.utils import Serializable



class RepeatRandom(Policy):
    def __init__(self,
                 *args,
                 repeat=4,
                 **kwargs):
        Serializable.quick_init(self, locals())
        Policy.__init__(self, *args, **kwargs)
        self.repeat = repeat
        self._current_repeat = 0
        self._previous_action = None

    def get_action(self, observation):
        if self._current_repeat == 0:
            self._previous_action =  np.random.uniform(-1, 1, self.action_dim)
        self._current_repeat = (self._current_repeat + 1) % self.repeat
        return self._previous_action


def train_with_exploratory_policy(raw_env, policy, exp_name, negative_samples, batch_size=32, code_size=32, epochs=30,
                                  image_shape=(64, 64, 3),  num_rollouts=32, max_path_length=16,
                                  encoder_arch='default', context_network='stack', lr=1e-3):
    # Create environment and sampler
    env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=1, img_size=image_shape)
    sampler = BaseSampler(env, policy, num_rollouts, max_path_length,)

    # Sampling data from the given exploratory policy and create a data iterator
    env_paths = sampler.obtain_samples(log=True, log_prefix='Data-EnvSampler-', random=True)
    img_seqs = np.stack([path['observations'] for path in env_paths])  # N x T x (img_shape)
    train_seq, val_seq = train_test_split(img_seqs)
    train_data = CPCDataGenerator(train_seq, batch_size, terms=1, negative_samples=negative_samples, predict_terms=1)
    validation_data = CPCDataGenerator(val_seq, batch_size, terms=1, negative_samples=negative_samples, predict_terms=1)

    # import pdb; pdb.set_trace()
    # for (x, y), labels in train_data:
    #     plot_seq(x, y, labels, name='point_mass_seq')
    #     break

    # Create the model
    model = network_cpc(image_shape=image_shape, terms=1, predict_terms=1, negative_samples=negative_samples,
                        code_size=code_size, learning_rate=lr, encoder_arch=encoder_arch, context_network=context_network)

    # Callbacks
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data', exp_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    callbacks = [#keras.callbacks.LearningRateScheduler(make_periodic_lr([5e-2, 5e-3, 5e-4]), verbose=1),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=1/3, patience=2, min_lr=1e-5, verbose=1, min_delta=0.001),
                 SaveEncoder(output_dir)]

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

def regression_to_pos()




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train CPC model')
    parser.add_argument('negative_samples', type=int, help='number of negative samples')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train the model for')
    parser.add_argument('--run_suffix', type=str, default='')
    # parser.add_argument('--encoder_arch', type=str, default='default')
    parser.add_argument('--context_network', type=str, default='stack')
    parser.add_argument('--lr', type=float, default=1e-3)

    args = parser.parse_args()

    raw_env = PointEnv()
    normalized_env = NormalizedEnv(raw_env)
    policy = RepeatRandom(2, 2, 4)

    exp_name = 'neg-%d_context-%s' % (args.negative_samples, args.context_network)

    train_with_exploratory_policy(raw_env, policy, exp_name, args.negative_samples, num_rollouts=512, batch_size=32,
                                  context_network=args.context_network, epochs=args.epochs, lr=args.lr)



