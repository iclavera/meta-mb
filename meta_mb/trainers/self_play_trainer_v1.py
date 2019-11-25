from meta_mb.agents.sac_agents_v1 import AgentV1
from meta_mb.logger import logger

import numpy as np
import time
import ray
import pickle


class TrainerV1(object):
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

        self.env = env
        self.alpha = alpha
        self.num_mc_goals = num_mc_goals
        self.refresh_interval = refresh_interval
        self.eval_interval = eval_interval
        self.n_itr = n_itr

        # feed pickled env to all agents to guarantee identical environment (including env seed)
        env_pickled = pickle.dumps(env)

        self.agents = [None] * num_agents

        for i in range(num_agents):
            agent = AgentV1.remote(
                agent_idx=i, 
                exp_dir=exp_dir, 
                snapshot_gap=snapshot_gap, 
                gpu_frac=gpu_frac, 
                seed=seeds[i], 
                env_pickled=env_pickled,
                n_initial_exploration_steps=n_initial_exploration_steps, 
                instance_kwargs=instance_kwargs, 
                eval_interval=eval_interval, 
                num_grad_steps=num_grad_steps,
                greedy_eps=greedy_eps,
            )
            self.agents[i] = agent

    def train(self):
        agents = self.agents
        proposed_goals_list = [[] for _ in range(len(agents))]  # updated with returned values from update_goal_buffer
        mc_goals = None

        time_start = time.time()

        for itr in range(self.n_itr):

            t = time.time()

            """------------------- assign tasks to agents --------------------------"""

            if itr % self.refresh_interval == 0:
                if self.alpha == 1:  # baseline
                    mc_goals = self.env.sample_goals(mode='target', num_samples=self.num_mc_goals)
                    q_list = None
                elif self.alpha == -1:  # baseline
                    mc_goals = self.env.sample_goals(mode=None, num_samples=self.num_mc_goals)
                    q_list = None
                else:
                    # Every refresh_interval, resample mc_goals and recompute Q-value predictions for all agents
                    mc_goals = self.env.sample_goals(mode=None, num_samples=self.num_mc_goals)
                    _tmp = [agent.compute_q_values.remote(mc_goals) for agent in agents]
                    q_list = np.asarray(ray.get(_tmp))

                _futures_update_info = [agent.update_info.remote(mc_goals=mc_goals, q_list=q_list) \
                            for agent, proposed_goals in zip(agents, proposed_goals_list)]
            else:
                _futures_update_info = []

            # Every iteration, resample goals to generate new goal batches
            _futures_update_buffer = [agent.update_buffer.remote(proposed_goals=proposed_goals) \
                             for agent, proposed_goals in zip(agents, proposed_goals_list)]
            _futures_train = [agent.train.remote(itr) for agent in agents]
            _futures_train.extend([agent.save_snapshot.remote(itr) for agent in agents])

            """------------------- collect future objects ---------------------"""

            _ = ray.get(_futures_update_info)
            proposed_goals_indices = ray.get(_futures_update_buffer)

            if 0 <= self.alpha < 1:
                # update proposed_goals_list
                # If an agent successfully proposes a goal at current iteration,
                # the goal will be appended to its goal buffer for the next iteration.
                proposed_goals_list = [[] for _ in range(len(agents))]
                proposed_goals_indices = np.unique(proposed_goals_indices)
                proposed_goals = mc_goals[proposed_goals_indices]
                proposer_indices = np.argmax(q_list, axis=0)[proposed_goals_indices]
                for goal, proposer_index in zip(proposed_goals, proposer_indices):
                    proposed_goals_list[proposer_index].append(goal)
            else:
                assert proposed_goals_indices[0] is None, proposed_goals_indices

            _ = ray.get(_futures_train)

            if itr == 0:
                ray.get([agent.finalize_graph.remote() for agent in agents])

            if itr % self.eval_interval == 0:
                logger.logkv('TimeTotal', time.time() - time_start)
                logger.logkv('TimeItr', time.time() - t)
                logger.logkv('Itr', itr)
                logger.dumpkvs()
