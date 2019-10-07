from meta_mb.agents.sac_agents import Agent
import numpy as np
import ray


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
            agent_kwargs,
            env,
            num_target_goals,
            n_itr,
            n_initial_exploration_steps=1e3,
            ):
        self.num_agents = num_agents
        self.env = env
        self.num_target_goals = num_target_goals
        self.n_itr = n_itr
        self.prepare_start_info = dict(seeds=seeds, agent_kwargs=agent_kwargs, n_initial_exploration_steps=n_initial_exploration_steps)

    def train(self):
        """
        Trains policy on env using algo
        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """

        agents = [Agent.remote() for _ in range(self.num_agents)]
        futures = [agent.prepare_start(
            seed,
            self.prepare_start_info['n_initial_exploration_steps'],
            self.prepare_start_info['agent_kwargs'],
        ) for agent, seed in zip(agents, self.prepare_start_info['seeds'])]
        ray.get(futures)

        for itr in range(self.n_itr):

            """----------------------- Compute q values to approximate goal distribution ------------------------"""

            target_goals = self.env.sample_goals(self.num_target_goals)
            futures = [agent.prepare_sample_collection(target_goals) for agent in agents]
            init_obs_list, log_q_list = list(zip(ray.get(futures)))
            max_log_q = np.max(log_q_list, axis=0)
            futures = [
                agent.update_replay_buffer(init_obs, agent_log_q, max_log_q, target_goals) for init_obs, agent_log_q, agent in zip(init_obs_list, log_q_list, agents)
            ]
            samples = [agent.collect_samples(agent_q_value, agent_q_value, target_goals) for agent, agent_q_value in zip(agents, q_values)]
            futures = [(agent.collect_samples(), agent.update_policy()) for agent in agents]
            ray.get(futures)

            if itr == 0:
                ray.get([agent.finalize_graph() for agent in agents])
