from meta_mb.samplers.base import BaseSampler
from meta_mb.optimizers.first_order_optimizer import FirstOrderOptimizer
from meta_mb.utils import utils
import numpy as np
import tensorflow as tf
from collections import OrderedDict

class MetaBPTTSampler(BaseSampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_mb.meta_envs.base.MetaEnv) : environment object
        policy (meta_mb.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of meta_envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            dynamics_model,
            meta_batch_size,
            rollouts_per_meta_task,
            max_path_length,
            parallel=False,
            deterministic_policy=False,
            optimize_actions=False,
            max_epochs=2,
            learning_rate=1e-4,
            **kwargs,
    ):
        super(MetaBPTTSampler, self).__init__(env, policy, rollouts_per_meta_task, max_path_length)
        assert not parallel

        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.max_path_length = max_path_length
        self.num_rollouts_per_meta_task = rollouts_per_meta_task
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * rollouts_per_meta_task * max_path_length
        self.num_rollouts = meta_batch_size * rollouts_per_meta_task
        self.total_timesteps_sampled = 0
        self.deterministic_policy = deterministic_policy
        self.num_models = getattr(dynamics_model, 'num_models', 1)

        self.optimizer = FirstOrderOptimizer(learning_rate=learning_rate, max_epochs=max_epochs)

        self.build_graph()

    def build_graph(self):
        self._initial_obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_rollouts,
                                                                       self.policy.obs_dim), name='init_obs')
        obses = []
        acts = []
        rewards = []
        means = []
        log_stds = []
        obs = self._initial_obs_ph
        for t in range(self.max_path_length):
            dist_policy = self.policy.distribution_info_sym(obs)
            act, dist_policy = self.policy.distribution.sample_sym(dist_policy)
            next_obs = self.dynamics_model.predict_batches_sym(obs, act)

            reward = self.env.tf_reward(obs, act, next_obs)
            obses.append(obs)
            acts.append(act)
            rewards.append(reward)
            means.append(dist_policy['mean'])
            log_stds.append(dist_policy['log_std'])
            # pre_tanhs.append(dist_policy['pre_tanh'])

            obs = next_obs

        # rewards = tf.stack(tf.split(tf.transpose(tf.stack(rewards, axis=0)), self.num_models))
        # random_weights = tf.random.uniform(shape=(self.num_models, self.num_rollouts, self.max_path_length))
        # rewards = rewards * random_weights / tf.reduce_sum(random_weights, axis=0)
        self._returns_var = tf.reduce_sum(rewards, axis=0)
        self._rewards_var = rewards
        self._actions_var = acts
        self._observations_var = obses
        self._means_var = means
        self._log_stds_var = log_stds

    def obtain_samples(self, log=False, log_prefix='', buffer=None):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        policy = self.policy
        policy.reset(dones=[True] * self.num_rollouts)

        # initial reset of meta_envs
        # init_obses = np.array([self.env.reset() for _ in range(self.num_rollouts)])

         #Peg
         #init_obs = np.array(
         #   [-0.07318277, -0.11606423, 1.54337565, -1.11586585, 1.4049322, - 1.69791869,
         #    -0.01773921, 0., 0., 0., 0., 0.,
         #    0., 0., -0.29127532, -0.0929903, 0.05638084, -0.2185227,
         #    -0.0253987, 0.0681465, -0.23636235, -0.09055804, 0.12782067])
        init_obs = [self.env.init_obs.copy() + np.random.uniform(-0.005, 0.005, size=23) for _ in range(self.num_rollouts)]

        sess = tf.get_default_session()
        observations, actions, means, log_stds, rewards = sess.run([self._observations_var,
                                                                    self._actions_var,
                                                                    self._means_var,
                                                                    self._log_stds_var,
                                                                    self._rewards_var
                                                                    ],
                                                                    feed_dict={self._initial_obs_ph: init_obs}
                                                                    )

        means = np.array(means).transpose((1, 0, 2))
        log_stds = np.array(log_stds).transpose((1, 0, 2))
        if log_stds.shape[0] == 1:
            log_stds = np.repeat(log_stds, self.num_rollouts, axis=0)
        agent_infos = [dict(mean=mean, log_std=log_std) for mean, log_std in zip(means, log_stds)]
        observations = np.array(observations).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        rewards = np.array(rewards).T
        dones = [[False for _ in range(self.max_path_length)] for _ in range(self.num_rollouts)]
        env_infos = [dict() for _ in range(self.num_rollouts)]
        paths = [dict(observations=obs, actions=act, rewards=rew,
                      dones=done, env_infos=env_info, agent_infos=agent_info) for
                 obs, act, rew, done, env_info, agent_info in
                 zip(observations, actions, rewards, dones, env_infos, agent_infos)]
        paths = dict((idx, paths[idx * self.num_rollouts_per_meta_task: (idx + 1) * self.num_rollouts_per_meta_task])
                     for idx in range(self.meta_batch_size))
        self.total_timesteps_sampled += self.total_samples

        return paths
