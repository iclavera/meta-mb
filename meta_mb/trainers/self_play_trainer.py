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

        # feed pickled env to all agents to guarantee identical environment (including env seed)
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
        proposed_goals_list = [[] for _ in range(len(agents))]  # updated with returned values from update_goal_buffer

        for itr in range(self.n_itr):

            t = time.time()

            """------------------- assign tasks to agents --------------------------"""

            if itr % self.goal_update_interval == 0:
                if self.alpha == 1:
                    mc_goals = self.env.sample_goals(mode='target', num_samples=self.num_sample_goals)
                    # baseline: sample goals across all free area
                    # sample_goals = self.env.sample_goals(mode=None, num_samples=self.num_sample_goals)
                    q_list = None
                else:
                    mc_goals = self.env.sample_goals(mode=None, num_samples=self.num_sample_goals)
                    _t = time.time()
                    _futures = [agent.compute_q_values.remote(mc_goals) for agent in agents]
                    q_list = np.asarray(ray.get(_futures))
                    logger.logkv('TimeCompQ', time.time() - _t)

                _futures = [agent.update_goal_buffer.remote(mc_goals, proposed_goals, q_list) \
                           for agent, proposed_goals in zip(agents, proposed_goals_list)]

            futures = []
            for agent in agents:
                futures.extend([agent.update_replay_buffer.remote(), agent.update_policy.remote(), agent.save_snapshot.remote()])

            """------------------- collect future objects ---------------------"""

            if itr % self.goal_update_interval == 0 and self.alpha > 1:
                # update proposed_goals_list
                # If an agent successfully proposes a goal at current iteration,
                # the goal will be appended to its goal buffer for the next iteration.
                proposed_goals_list = [[] for _ in range(len(agents))]
                proposed_goals_indices = ray.get(_futures)
                proposed_goals_indices = np.unique(np.concatenate(proposed_goals_indices))
                proposed_goals = mc_goals[proposed_goals_indices]
                proposer_indices = np.argmax(q_list, axis=0)[proposed_goals_indices]
                for goal, proposer_index in zip(proposed_goals, proposer_indices):
                    proposed_goals_list[proposer_index].append(goal)

            ray.get(futures)

            if itr == 0:
                ray.get([agent.finalize_graph.remote() for agent in agents])

            if itr % self.eval_interval == 0:
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
