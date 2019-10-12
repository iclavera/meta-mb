from meta_mb.agents.sac_agent import Agent
import numpy as np
import ray


print(ray.init())


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
            exp_dir,
            n_initial_exploration_steps=1e3,
            ):
        self.num_agents = num_agents
        self.env = env
        self.num_target_goals = num_target_goals
        self.n_itr = n_itr
        self.prepare_start_info = dict(seeds=seeds,
                                       agent_kwargs=agent_kwargs,
                                       n_initial_exploration_steps=n_initial_exploration_steps)
        self.agents = [Agent.remote(i, exp_dir) for i in range(self.num_agents)]

    def train(self):
        agents = self.agents
        futures = [agent.prepare_start.remote(
            seed,
            self.prepare_start_info['n_initial_exploration_steps'],
            self.prepare_start_info['agent_kwargs'],
        ) for agent, seed in zip(agents, self.prepare_start_info['seeds'])]
        ray.get(futures)

        for itr in range(self.n_itr):

            """----------------------- Compute q values to approximate goal distribution ------------------------"""

            target_goals = self.env.sample_goals(self.num_target_goals)
            futures = [agent.prepare_sample_collection.remote(target_goals) for agent in agents]
            q_list = ray.get(futures)
            max_q = np.max(q_list, axis=0)

            futures = []
            for agent_q, agent in zip(q_list, agents):
                futures.extend([agent.update_replay_buffer.remote(agent_q, max_q, target_goals), agent.update_policy.remote()])
            ray.get(futures)

            if itr == 0:
                ray.get([agent.finalize_graph.remote() for agent in agents])
