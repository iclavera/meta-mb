import numpy as np
from meta_mb.logger import logger
from meta_mb.agents.ve_value_function import ValueFunction
import pickle


class ValueEnsembleWrapper(object):
    def __init__(self, env_pickled, size, num_mc_goals, gpu_frac, instance_kwargs):
        self.env = pickle.loads(env_pickled)
        self.size = size
        self.num_mc_goals = num_mc_goals

        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        self.sess = sess = tf.Session(config=config)

        self.vfun_list = []

        with sess.as_default():
            for vfun_idx in range(size):
                vfun = ValueFunction(
                    env=self.env,
                    vfun_idx=vfun_idx,
                    reward_scale=instance_kwargs['reward_scale'],
                    discount=instance_kwargs['discount'],
                    learning_rate=instance_kwargs['learning_rate'],
                    gae_lambda=instance_kwargs['gae_lambda'],
                    normalize_adv=instance_kwargs['normalize_adv'],
                    positive_adv=instance_kwargs['positive_adv'],
                    hidden_sizes=instance_kwargs['vfun_hidden_sizes'],
                    hidden_nonlinearity=instance_kwargs['vfun_hidden_nonlinearity'],
                    output_nonlinearity=instance_kwargs['vfun_output_nonlinearity'],
                    batch_size=instance_kwargs['vfun_batch_size'],
                    num_grad_steps=instance_kwargs['vfun_num_grad_steps'],
                    max_replay_buffer_size=instance_kwargs['vfun_max_replay_buffer_size'],
                )
                self.vfun_list.append(vfun)

        sess.run(tf.initializers.global_variables())

    def sample_goals(self, init_obs_no, log_prefix='ve-', log=True):
        if self.size == 0:  # baseline
            return self.env.sample_goals(mode=None, num_samples=len(init_obs_no))

        mc_goals = self.env.sample_goals(mode=None, num_samples=self.num_mc_goals)
        input_obs = np.repeat(init_obs_no, repeats=self.num_mc_goals, axis=0)
        input_goal = np.tile(mc_goals, [len(init_obs_no), 1])
        values = []
        with self.sess.as_default():
            for vfun in self.vfun_list:
                # (num_envs * num_goals, 1) => (num_envs, num_goals)
                values.append(vfun.compute_values(input_obs, input_goal).reshape((len(init_obs_no), self.num_mc_goals)))

        # (size, num_envs, num_goals) => (num_envs, num_goals)
        goal_distribution = np.var(values, axis=0)
        goal_distribution /= np.sum(goal_distribution, axis=-1, keepdims=True)
        indices = [np.random.choice(self.num_mc_goals, size=1, p=goal_distribution[row_idx, :])[0] for row_idx in range(len(init_obs_no))]
        samples = mc_goals[indices]

        if log:
            logger.logkv(log_prefix + 'PMax', np.max(goal_distribution))
            logger.logkv(log_prefix + 'PMin', np.min(goal_distribution))
            logger.logkv(log_prefix + 'PStd', np.std(goal_distribution))

        return samples

    def train(self, paths, itr, log=True):
        with self.sess.as_default():
            for vfun in self.vfun_list:
                vfun.train(paths, itr=itr, log=log)  # TODO: sample 80%

    def save_snapshot(self, itr):
        with self.sess.as_default():
            params = dict(itr=itr, vfun_tuple=tuple(self.vfun_list))
            logger.save_itr_params(itr, params, 've_')
