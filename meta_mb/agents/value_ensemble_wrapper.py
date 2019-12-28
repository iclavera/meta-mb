import numpy as np
from meta_mb.logger import logger
from meta_mb.agents.ve_value_function import ValueFunction
from meta_mb.agents.ve_value_function_td_inf import ValueFunction as TDInfValueFunction
from meta_mb.agents.ve_value_function_supervised import ValueFunction as SupervisedValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.replay_buffers.gc_simple_replay_buffer import SimpleReplayBuffer
from meta_mb.samplers.gc_mb_sample_processor import ModelSampleProcessor
from meta_mb.samplers.gc_mb_sample_processor_traj import ModelSampleProcessor as TrajModelSampleProcessor

import pickle
import tensorflow as tf


class ValueEnsembleWrapper(object):
    def __init__(self, env_pickled, size, config, instance_kwargs, batch_size_fraction=0.8, update_str='td_1'):
        self.size = size
        self.env = env = pickle.loads(env_pickled)
        self.sess = sess = tf.Session(config=config)

        if size == 0:
            return

        self.update_str = update_str
        if update_str == 'td_1':
            vfun_factory = ValueFunction
            sample_processor_factory = ModelSampleProcessor
        elif update_str == 'td_inf':
            vfun_factory = TDInfValueFunction
            sample_processor_factory = ModelSampleProcessor
        elif update_str == 'supervised':
            vfun_factory = SupervisedValueFunction
            sample_processor_factory = TrajModelSampleProcessor
        else:
            print(update_str)
            raise NotImplementedError

        self.vfun_list = []
        with sess.as_default():
            for vfun_idx in range(size):
                vfun = vfun_factory(
                    env=self.env,
                    vfun_idx=vfun_idx,
                    reward_scale=instance_kwargs['reward_scale'],
                    discount=instance_kwargs['discount'],
                    learning_rate=instance_kwargs['learning_rate'],
                    hidden_sizes=instance_kwargs['vfun_hidden_sizes'],
                    hidden_nonlinearity=instance_kwargs['vfun_hidden_nonlinearity'],
                    output_nonlinearity=instance_kwargs['vfun_output_nonlinearity'],
                )
                self.vfun_list.append(vfun)

        self.batch_size_fraction = batch_size_fraction
        self.num_mc_goals = instance_kwargs['num_mc_goals']
        self.num_grad_steps = instance_kwargs['vfun_num_grad_steps']
        self.batch_size = instance_kwargs['vfun_batch_size']

        max_replay_buffer_size = instance_kwargs['vfun_max_replay_buffer_size']
        if max_replay_buffer_size > 0:
            self.replay_buffer = SimpleReplayBuffer(
                env_spec=env,
                max_replay_buffer_size=instance_kwargs['vfun_max_replay_buffer_size']
            )
        else:
            # no replay buffer, use on-policy data only
            self.replay_buffer = None

        baseline = LinearFeatureBaseline()
        self.sample_processor = sample_processor_factory(
            reward_fn=env.reward,
            achieved_goal_fn=env.get_achieved_goal,
            baseline=baseline,
            replay_k=-1,
            discount=instance_kwargs['discount'],
            gae_lambda=instance_kwargs['gae_lambda'],
            normalize_adv=instance_kwargs['normalize_adv'],
            positive_adv=instance_kwargs['positive_adv'],
        )

        sess.run(tf.initializers.global_variables())

    def sample_goals(self, init_obs_no, log_prefix='ve-', log=True):
        if self.size == 0:  # baseline
            return self.env.sample_goals(mode=None, num_samples=len(init_obs_no))

        mc_goals = self.env.sample_goals(mode=None, num_samples=self.num_mc_goals)
        if self.env.init_obs_noise:
            input_obs = np.repeat(init_obs_no, repeats=self.num_mc_goals, axis=0)
            input_goal = np.tile(mc_goals, [len(init_obs_no), 1])

            values = []
            with self.sess.as_default():
                for vfun in self.vfun_list:
                    # (num_envs * num_goals, 1) => (num_envs, num_goals)
                    # normalize first
                    _values = vfun.compute_values(input_obs, input_goal).flatten()
                    _values = (_values - np.mean(_values)) / np.std(_values)
                    values.append(_values.reshape((len(init_obs_no)), self.num_mc_goals))

            # (size, num_envs, num_goals) => (num_envs, num_goals)
            goal_distribution = np.var(values, axis=0)
            goal_distribution /= np.sum(goal_distribution, axis=-1, keepdims=True)
            indices = [np.random.choice(self.num_mc_goals, size=1, p=goal_distribution[row_idx, :])[0] for row_idx in range(len(init_obs_no))]
            samples = mc_goals[indices]

        else:
            input_obs = np.repeat(init_obs_no[0][np.newaxis, ...], repeats=self.num_mc_goals, axis=0)
            input_goal = mc_goals

            values = []
            with self.sess.as_default():
                for vfun in self.vfun_list:
                    # (num_goals, 1) => (num_goals,)
                    # normalize first
                    _values = vfun.compute_values(input_obs, input_goal).flatten()
                    _values = (_values - np.mean(_values)) / np.std(_values)
                    values.append(_values)

            # (size, num_goals) => (num_goals)
            goal_distribution = np.var(values, axis=0)
            goal_distribution /= np.sum(goal_distribution)
            indices = np.random.choice(self.num_mc_goals, size=len(init_obs_no), p=goal_distribution, replace=False)  # FIXME: replace = True?
            samples = mc_goals[indices]

        if log:
            logger.logkv(log_prefix + 'PMax', np.max(goal_distribution))
            logger.logkv(log_prefix + 'PMin', np.min(goal_distribution))
            logger.logkv(log_prefix + 'PStd', np.std(goal_distribution))

        return samples

    def train(self, paths, itr, log=True):
        if self.size == 0:
            return

        samples_data = self.sample_processor.process_samples(paths, eval=False, log='all', log_prefix='ve-train-')

        if self.replay_buffer is not None:
            if self.update_str == 'td_1':  # goals, observations, actions, rewards, terminals, next_observations
                self.replay_buffer.add_samples(
                    goals=samples_data['goals'],
                    observations=samples_data['observations'],
                    actions=samples_data['actions'],
                    rewards=samples_data['rewards'],
                    terminals=samples_data['dones'],
                    next_observations=samples_data['next_observations'],
                )
            elif self.update_str == 'td_inf':
                self.replay_buffer.add_samples(
                    goals=samples_data['goals'],
                    observations=samples_data['observations'],
                    actions=samples_data['actions'],
                    rewards=samples_data['rewards'],
                    terminals=samples_data['dones'],
                    next_observations=samples_data['next_observations'],
                    returns=samples_data['returns'],
                )
            elif self.update_str == 'supervised':
                num_samples = len(samples_data['goals'])
                self.replay_buffer.add_samples(
                    goals=samples_data['goals'],
                    observations=np.zeros((num_samples, self.env.obs_dim)),
                    actions=np.zeros((num_samples, self.env.act_dim)),
                    rewards=np.zeros((num_samples,)),
                    terminals=np.zeros((num_samples,)),
                    next_observations=np.zeros((num_samples, self.env.obs_dim)),
                    returns=samples_data['returns'],
                )

            with self.sess.as_default():
                for grad_step in range(self.num_grad_steps):
                    batch_indices = self.replay_buffer.random_batch_indices(int(self.batch_size/self.batch_size_fraction))
                    for vfun in self.vfun_list:
                        rand_indices = np.random.choice(len(batch_indices), size=self.batch_size)
                        batch = self.replay_buffer.get_batch_by_indices(rand_indices)
                        vfun.train(batch, itr=itr, log=log)
                        log = False

        else:  # train online
            with self.sess.as_default():
                for grad_step in range(self.num_grad_steps):
                    for vfun in self.vfun_list:
                        batch = self._get_batch_from_online_paths(samples_data)
                        vfun.train(batch, itr=itr, log=log)
                        log = False

    def reset(self):
        with self.sess.as_default():
            for vfun in self.vfun_list:
                vfun.reset_params()

    def _get_batch_from_online_paths(self, samples_data):
        indices = np.random.choice(len(samples_data['goals']), size=int(len(samples_data['goals']) * self.batch_size_fraction))
        batch = dict()
        batch['goals'] = samples_data['goals'][indices]
        if self.update_str == 'td_inf' or self.update_str == 'supervised':
            batch['returns'] = samples_data['returns'][indices]
        if self.update_str == 'supervised':
            return batch
        batch['observations'] = samples_data['observations'][indices]
        batch['actions'] = samples_data['actions'][indices]
        batch['rewards'] = samples_data['rewards'][indices]
        batch['dones'] = samples_data['dones'][indices]
        batch['next_observations'] = samples_data['next_observations'][indices]
        return batch

    def save_snapshot(self, itr):
        with self.sess.as_default():
            params = dict(itr=itr, vfun_tuple=tuple(self.vfun_list))
            logger.save_itr_params(itr, params, 've_')
