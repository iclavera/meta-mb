from meta_mb.agents.sac_agent import Agent
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
            num_agents,
            seeds,
            instance_kwargs,
            gpu_frac,
            env,
            num_sample_goals,
            alpha,
            n_itr,
            exp_dir,
            goal_update_interval,
            eval_interval,
            snapshot_gap=500,
            n_initial_exploration_steps=1e3,
        ):

        self.env = env
        self.num_sample_goals = num_sample_goals
        self.alpha = alpha
        self.goal_update_interval = goal_update_interval
        self.eval_interval = eval_interval
        self.n_itr = n_itr

        env_pickled = pickle.dumps(env)

        self.agents = [None] * num_agents

        for i in range(num_agents):
            agent = Agent.remote(
                i, exp_dir, snapshot_gap, gpu_frac, seeds[i],
                env_pickled, n_initial_exploration_steps, instance_kwargs, eval_interval,
            )
            self.agents[i] = agent

    def train(self):
        agents = self.agents

        for itr in range(self.n_itr):

            t = time.time()

            if itr % self.goal_update_interval == 0:
                if self.alpha == 1:
                    sample_goals = self.env.sample_goals(mode='target', num_samples=self.num_sample_goals)
                    # baseline: sample goals across all free area
                    # sample_goals = self.env.sample_goals(mode=None, num_samples=self.num_sample_goals)
                    q_list = None
                else:
                    sample_goals = self.env.sample_goals(mode=None, num_samples=self.num_sample_goals)
                    _t = time.time()
                    _futures = [agent.compute_q_values.remote(sample_goals) for agent in agents]
                    q_list = np.asarray(ray.get(_futures))
                    logger.logkv('TimeCompQ', time.time() - _t)
                futures = [agent.update_goal_buffer.remote(sample_goals, q_list) for agent in agents]
            else:
                futures = []

            for agent in agents:
                futures.extend([agent.update_replay_buffer.remote(), agent.update_policy.remote(), agent.save_snapshot.remote()])
            ray.get(futures)

            if itr == 0:
                ray.get([agent.finalize_graph.remote() for agent in agents])

            if itr % self.eval_interval == 0:
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
