from meta_mb.samplers.base import BaseSampler
from meta_mb.utils import utils
import numpy as np
import tensorflow as tf


class METRPOSampler(BaseSampler):
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
            deterministic=True,
    ):
        super(METRPOSampler, self).__init__(env, policy, num_rollouts, max_path_length)
        assert not parallel

        self.env = env
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.max_path_length = max_path_length
        self.total_samples = num_rollouts * max_path_length
        self.num_rollouts = num_rollouts
        self.total_timesteps_sampled = 0
        self.deterministic = deterministic

        self.build_graph()

    def build_graph(self):
        self._initial_obs_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.policy.obs_dim), name='init_obs')
        obses = []
        acts = []
        rewards = []
        agent_infos = []
        obs = self._initial_obs_ph
        for t in range(self.max_path_length):
            dist_policy = self.policy.distribution_info_sym(obs)
            if self.deterministic:
                act = dist_policy['mean']
            else:
                act = dist_policy['mean'] + tf.random.normal(shape=tf.shape(obs)[0]) * tf.exp(dist_policy['log_std'])
            next_obs = self.dynamics_model.distribution_info_sym(obs, act)
            reward = self.env.tf_reward(obs, act, next_obs)

            obses.append(obs)
            acts.append(act)
            rewards.append(reward)
            agent_infos.append(dist_policy)

            obs = next_obs

        self._returns_var = tf.reduce_sum(rewards, axis=0)
        self._rewards_var = rewards
        self._actions_var = acts
        self._observations_var = obses
        self._agent_infos_var = agent_infos

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
        init_obses = np.array([self.env.reset() for _ in range(self.num_rollouts)])

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
        agent_infos = utils.stack_tensor_dict_list(agent_infos)
        dones = [[False for _ in range(self.max_path_length)] for _ in range(self.num_rollouts)]
        env_infos = [dict() for _ in range(self.num_rollouts)]
        paths = [dict(observations=obs, actions=act, rewards=rew,
                      dones=done, env_infos=env_info, agent_infos=agent_info) for
                 obs, act, rew, done, env_info, agent_info in
                 zip(observations, actions, rewards, dones, env_infos, agent_infos)]
        self.total_timesteps_sampled += self.total_samples

        return paths
