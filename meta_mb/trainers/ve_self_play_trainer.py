from meta_mb.agents.ve_sac_agent import Agent
from meta_mb.agents.ve_value_function import ValueFunction
from meta_mb.agents.value_ensemble_wrapper import ValueEnsembleWrapper
from meta_mb.replay_buffers.gc_simple_replay_buffer import SimpleReplayBuffer
from meta_mb.logger import logger

import numpy as np
import time
import ray
import pickle


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
            size_value_ensemble,
            seed,
            instance_kwargs,
            gpu_frac,
            env,
            num_mc_goals,
            refresh_interval,
            alpha,
            n_itr,
            exp_dir,
            eval_interval,
            num_grad_steps,
            greedy_eps,
            snapshot_gap,
            n_initial_exploration_steps=1e3,
        ):

        self.size_value_ensemble = size_value_ensemble
        self.env = env
        self.alpha = alpha
        self.refresh_interval = refresh_interval
        self.eval_interval = eval_interval
        self.n_itr = n_itr

        # feed pickled env to all agents to guarantee identical environment (including env seed)
        env_pickled = pickle.dumps(env)

        """---------------- value ensemble and agent share replay buffer ------------------"""

        replay_buffer = SimpleReplayBuffer(self.env, instance_kwargs['max_replay_buffer_size'])

        """---------------- initiate value ensemble to compute intrinsic reward ------------"""

        self.value_ensemble = [ValueFunction(
            replay_buffer=replay_buffer,
            obs_dim=self.env.obs_dim,
            goal_dim=self.env.goal_dim,
            gpu_frac=gpu_frac,
            vfun_idx=vfun_idx,
            hidden_nonlinearity=instance_kwargs["vfun_hidden_nonlinearity"],
            output_nonlinearity=instance_kwargs["vfun_output_nonlinearity"],
            batch_size=instance_kwargs["vfun_batch_size"],
            reward_scale=instance_kwargs["reward_scale"],
            discount=instance_kwargs["discount"],
            learning_rate=instance_kwargs["learning_rate"],
        ) for vfun_idx in range(size_value_ensemble)]

        value_ensemble_wrapper = ValueEnsembleWrapper(
            size=size_value_ensemble,
            vfun_list=self.value_ensemble,
            env=env,
            num_mc_goals=num_mc_goals,
        )

        """------------------ initiate remote SAC agent ----------------------"""
        self.agent = Agent(
            exp_dir=exp_dir,
            snapshot_gap=snapshot_gap,
            gpu_frac=gpu_frac,
            seed=seed,
            env_pickled=env_pickled,
            value_ensemble=value_ensemble_wrapper,
            replay_buffer=replay_buffer,
            n_initial_exploration_steps=n_initial_exploration_steps,
            instance_kwargs=instance_kwargs,
            eval_interval=eval_interval,
            num_grad_steps=num_grad_steps,
            greedy_eps=greedy_eps,
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
        agent = self.agent
        value_ensemble = self.value_ensemble

        time_start = time.time()

        for itr in range(self.n_itr):

            t = time.time()
            _futures = []

            """------------------------------- train agent -------------------------"""

            agent.train(itr=itr)
            agent.save_snapshot(itr=itr)

            """-------------------------- train value ensemble ---------------------------"""

            for idx, vfun in enumerate(value_ensemble):
                vfun.train(itr=itr, log=True, log_prefix=f"vc-{idx}-")

            if itr == 0:
                agent.finalize_graph()

            if itr % self.eval_interval == 0:
                logger.logkv('TimeTotal', time.time() - time_start)
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
