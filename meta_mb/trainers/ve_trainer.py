from meta_mb.agents.value_ensemble_wrapper import ValueEnsembleWrapper
from meta_mb.logger import logger
from meta_mb.samplers.ve_gc_sampler import Sampler

import time
import pickle
import tensorflow as tf
import joblib


class Trainer(object):
    """
    Performs steps for MAML
    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) :
        sample_processor (SampleProcessor) :
        baseline (Baseline) :
        policy (Policy) :
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            saved_policy_path,
            size_value_ensemble,
            instance_kwargs,
            gpu_frac,
            n_itr,
            eval_interval,
            ve_update_str,
        ):

        self.eval_interval = eval_interval
        self.n_itr = n_itr

        self.sess = sess = tf.Session()

        with sess.as_default():
            data = joblib.load(saved_policy_path)

        self.policy = data['policy']
        self.env = data['env']
        self.Q_targets = data['Q_targets']

        env_pickled = pickle.dumps(self.env)

        self.value_ensemble = ValueEnsembleWrapper(
            size=size_value_ensemble,
            env_pickled=env_pickled,
            gpu_frac=gpu_frac,
            instance_kwargs=instance_kwargs,
            update_str=ve_update_str,
        )

        dummy_value_ensemble = ValueEnsembleWrapper(
            size=0,
            env_pickled=env_pickled,
            gpu_frac=gpu_frac,
            instance_kwargs=instance_kwargs,
        )

        self.sampler = Sampler(
            env=self.env,
            goal_sampler=dummy_value_ensemble,
            policy=self.policy,
            num_rollouts=instance_kwargs['num_rollouts'],
            max_path_length=instance_kwargs['max_path_length'],
            n_parallel=instance_kwargs['n_parallel'],
            action_noise_str='none',
        )

    def train(self):
        """
        Loop:
        1. feed goals to goal buffer of the agent
        2. call agent.train()
        3. use value ensemble to sample goals
        4. train value ensemble

        :return:
        """
        value_ensemble = self.value_ensemble
        sess = self.sess

        time_start = time.time()

        for itr in range(self.n_itr):

            t = time.time()

            """------------------------------- train agent -------------------------"""

            with sess.as_default():
                paths, goal_samples = self.sampler.collect_rollouts(greedy_eps=0, apply_action_noise=False,
                                                                         log=True, log_prefix='train-')
                params = dict(itr=itr, policy=self.policy, env=self.env, Q_targets=self.Q_targets,
                              goal_samples=goal_samples)
                logger.save_itr_params(itr, params, 'agent_')

            """-------------------------- train value ensemble ---------------------------"""

            value_ensemble.train(paths, itr=itr, log=True)
            value_ensemble.save_snapshot(itr=itr)

            if itr % self.eval_interval == 0:
                logger.logkv('TimeTotal', time.time() - time_start)
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
