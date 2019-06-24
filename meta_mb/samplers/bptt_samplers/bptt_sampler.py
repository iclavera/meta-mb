from meta_mb.samplers.base import BaseSampler
from meta_mb.optimizers.first_order_optimizer import FirstOrderOptimizer
from meta_mb.utils import utils
import numpy as np
import tensorflow as tf
from collections import OrderedDict

class BPTTSampler(BaseSampler):
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
            num_rollouts,
            max_path_length,
            parallel=False,
            deterministic_policy=False,
            deterministic_dynamics=True,
            optimize_actions=False,
            max_epochs=2,
            learning_rate=1e-4,
            **kwargs,
    ):
        super(BPTTSampler, self).__init__(env, policy, num_rollouts, max_path_length)
        assert not parallel

        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.max_path_length = max_path_length
        self.total_samples = num_rollouts * max_path_length
        self.num_rollouts = num_rollouts
        self.total_timesteps_sampled = 0
        self.deterministic_policy = deterministic_policy
        self.deterministic_dynamics = deterministic_dynamics
        self.optimize_actions = optimize_actions
        self.num_models = getattr(dynamics_model, 'num_models', 1)

        self.optimizer = FirstOrderOptimizer(learning_rate=learning_rate, max_epochs=max_epochs)

        self.build_graph()

    def build_graph(self):
        self._initial_obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.num_rollouts * self.num_models, self.policy.obs_dim), name='init_obs')
        if self.optimize_actions:
            self.optimal_actions = tf.Variable(np.zeros(self.max_path_length, self.policy.action_dim),
                                      trainable=True,
                                      name='optimal_actions')
        obses = []
        acts = []
        rewards = []
        agent_infos = []
        obs = self._initial_obs_ph
        for t in range(self.max_path_length):
            dist_policy = self.policy.distribution_info_sym(obs)
            if self.optimize_actions:
                act = self.optimal_actions[t]
            elif self.deterministic_policy:
                act = dist_policy['mean']
            else:
                act = dist_policy['mean'] + tf.random.normal(shape=(self.num_rollouts * self.num_models, self.policy.action_dim)) * tf.exp(dist_policy['log_std'])
            dist_dynamics = self.dynamics_model.distribution_info_sym(obs, act)
            if self.deterministic_dynamics:
                next_obs = dist_dynamics['mean']
            else:
                next_obs = dist_dynamics['mean'] + tf.random.normal(shape=(self.num_rollouts * self.num_models, self.policy.obs_dim)) * tf.exp(dist_dynamics['log_std'])
            reward = self.env.tf_reward(obs, act, next_obs)
            # TODO: I think it's better to have random weighted rewards than random weighted states. So the way to
            # do it would be by predicting # num-ensemble trajectories, and then do a random weighted of the rewards.

            obses.append(obs)
            acts.append(act)
            rewards.append(reward)
            agent_infos.append(dist_policy)

            obs = next_obs

        # rewards = tf.stack(tf.split(tf.transpose(tf.stack(rewards, axis=0)), self.num_models))
        # random_weights = tf.random.uniform(shape=(self.num_models, self.num_rollouts, self.max_path_length))
        # rewards = rewards * random_weights / tf.reduce_sum(random_weights, axis=0)
        self._returns_var = tf.reduce_sum(rewards, axis=0)
        self._rewards_var = rewards
        self._actions_var = acts
        self._observations_var = obses
        self._agent_infos_var = agent_infos

        with tf.variable_scope('model_sampler'):
            self._loss = -tf.reduce_mean(self._returns_var)
            if self.optimize_actions:
                target = self.optimal_actions
            else:
                target = self.policy
            self.optimizer.build_graph(loss=self._loss,
                                       target=target,
                                       input_ph_dict=dict(initial_obs=self._initial_obs_ph)
                                       )

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
        init_obses = np.array([self.env.reset() for _ in range(self.num_rollouts)] * self.num_models)

        sess = tf.get_default_session()
        observations, actions, agent_infos, rewards = sess.run([self._observations_var,
                                                                self._actions_var,
                                                                self._agent_infos_var,
                                                                self._rewards_var,
                                                               ],
                                                               feed_dict={self._initial_obs_ph: init_obses}
                                                               )

        observations = np.array(observations).transpose((1, 0, 2))
        actions = np.array(actions).transpose((1, 0, 2))
        rewards = np.array(rewards).T
        agent_infos = transpose_list_dict(agent_infos)
        dones = [[False for _ in range(self.max_path_length)] for _ in range(self.num_rollouts)]
        env_infos = [dict() for _ in range(self.num_rollouts)]
        paths = [dict(observations=obs, actions=act, rewards=rew,
                      dones=done, env_infos=env_info, agent_infos=agent_info) for
                 obs, act, rew, done, env_info, agent_info in
                 zip(observations, actions, rewards, dones, env_infos, agent_infos)]
        self.total_timesteps_sampled += self.total_samples

        return paths

    def optimize_policy(self, log=True):
        init_obses = np.array([self.env.reset() for _ in range(self.num_rollouts)] * self.num_models)
        input_dict = dict(initial_obs=init_obses)
        self.optimizer.optimize(input_dict)


def transpose_list_dict(list_dict):
    t_mean = np.array([d['mean'] for d in list_dict])
    t_mean = t_mean.transpose((1, 0, 2)) # batch, time, obs_dim
    t_log_std = np.array([d['log_std'][0] for d in list_dict]) # time, obs_dim
    t_list_dict = [dict(mean=mean, log_std=t_log_std) for mean in t_mean]
    return t_list_dict
