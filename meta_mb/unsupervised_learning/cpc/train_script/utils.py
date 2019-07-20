import numpy as np
import tensorflow as tf

from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.img_wrapper_env import ImgWrapperEnv
from meta_mb.policies.base import Policy
from meta_mb.samplers.base import BaseSampler
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
        return self._previous_action, {}


def collect_img(raw_env, policy, num_rollouts=32, max_path_length=16, image_shape=(64, 64, 3)):
    # Create environment and sampler
    env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=1, img_size=image_shape)
    sampler = BaseSampler(env, policy, num_rollouts, max_path_length, ignore_done=True)

    # Sampling data from the given exploratory policy and create a data iterator
    env_paths = sampler.obtain_samples(log=True, log_prefix='Data-EnvSampler-', random=True)
    img_seqs = np.stack([path['observations'] for path in env_paths])  # N x T x (img_shape)
    actions = np.stack([path['actions'] for path in env_paths])
    return img_seqs, actions


def init_uninit_vars(sess):
    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))