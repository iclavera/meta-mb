from meta_mb.agents.ve_sac_agent import Agent
from meta_mb.agents.value_ensemble_wrapper import ValueEnsembleWrapper
from meta_mb.logger import logger

import time
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
            ve_reset_interval,
            seed,
            instance_kwargs,
            gpu_frac,
            env,
            n_itr,
            eval_interval,
            greedy_eps,
            n_initial_exploration_steps=1e3,
        ):

        self.eval_interval = eval_interval
        self.n_itr = n_itr

        # feed pickled env to all agents to guarantee identical environment (including env seed)
        env_pickled = pickle.dumps(env)

        """------------- value ensemble is related to the agent via this wrapper ------"""

        self.value_ensemble = ValueEnsembleWrapper(
            size=size_value_ensemble,
            env_pickled=env_pickled,
            gpu_frac=gpu_frac,
            instance_kwargs=instance_kwargs,
        )
        self.ve_reset_interval = ve_reset_interval

        """------------------ initiate remote SAC agent ----------------------"""
        self.agent = Agent(
            gpu_frac=gpu_frac,
            seed=seed,
            env_pickled=env_pickled,
            value_ensemble=self.value_ensemble,
            n_initial_exploration_steps=n_initial_exploration_steps,
            instance_kwargs=instance_kwargs,
            eval_interval=eval_interval,
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

            """------------------------------- train agent -------------------------"""

            on_policy_paths, goal_samples_snapshot = agent.train(itr=itr)
            agent.save_snapshot(itr=itr, goal_samples=goal_samples_snapshot)

            """-------------------------- train value ensemble ---------------------------"""

            value_ensemble.train(on_policy_paths, itr=itr, log=True)
            value_ensemble.save_snapshot(itr=itr)

            if itr == 0:
                agent.finalize_graph()

            if self.ve_reset_interval > 0 and itr % self.ve_reset_interval == 0:
                value_ensemble.reset()

            if itr % self.eval_interval == 0:
                logger.logkv('TimeTotal', time.time() - time_start)
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
