from meta_mb.agents.sac_agents_v2 import Agent
from meta_mb.logger import logger

import numpy as np
import time


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
            instance_kwargs,
            config,
            env,
            num_mc_goals,
            update_expert_interval,
            n_itr,
            eval_interval,
            policy_greedy_eps,
            goal_greedy_eps,
            n_initial_exploration_steps=1e3,
        ):

        self.env = env
        self.num_mc_goals = num_mc_goals
        self.update_expert_interval = update_expert_interval
        self.eval_interval = eval_interval
        self.goal_greedy_eps = goal_greedy_eps
        self.n_itr = n_itr

        self.agents = [
            Agent(
                agent_idx=i,
                env=env,
                config=config,
                n_initial_exploration_steps=n_initial_exploration_steps,
                instance_kwargs=instance_kwargs,
                eval_interval=eval_interval,
                greedy_eps=policy_greedy_eps,
            ) for i in range(num_agents)
        ]

    def train(self):
        agents = self.agents
        env = self.env
        mc_goals, expert_q_values = None, None

        time_start = time.time()

        for itr in range(self.n_itr):

            t = time.time()

            """------------------- assign tasks to agents --------------------------"""

            if self.goal_greedy_eps < 1:
                if itr % self.update_expert_interval == 0:
                    # Every update_expert_interval, resample mc_goals and recompute expert value predictions
                    mc_goals = env.sample_goals(mode=None, num_samples=self.num_mc_goals)
                    q_values_list = [agent.compute_q_values(mc_goals) for agent in agents]
                    expert_q_values = np.max(q_values_list, axis=0)

                else:
                    q_values_list = [agent.compute_q_values(mc_goals) for agent in agents]

                for agent, agent_q_values in zip(agents, q_values_list):
                    # FIXME: normalize here? if so, along which axis?
                    agent.update_goal_dist(
                        mc_goals=mc_goals, expert_q_values=expert_q_values, agent_q_values=agent_q_values,
                    )
            else:
                # no need to compute q values for baseline
                if itr % self.update_expert_interval == 0:
                    mc_goals = env.sample_goals(mode=None, num_samples=self.num_mc_goals)
                    for agent in agents:
                        agent.goal_sampler.set_goal_dist(
                            mc_goals=mc_goals, goal_dist=None,
                        )
            time_goal_sampling = time.time() - t

            for agent in agents:
                _, goal_samples_snapshot = agent.train(itr)
                agent.save_snapshot(itr=itr, goal_samples=goal_samples_snapshot)

            if itr == 0:
                for agent in agents:
                    agent.finalize_graph()

            if itr % self.eval_interval == 0:
                eval_paths = sum([agent.collect_eval_paths() for agent in agents], [])
                _ = agents[0].sample_processor.process_samples(eval_paths, eval=True, log='all', log_prefix='eval-')

                logger.logkv('TimeTotal', time.time() - time_start)
                logger.logkv('TimeGoalSampling', time_goal_sampling)
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
