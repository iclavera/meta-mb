from meta_mb.agents.sac_agent import Agent
import numpy as np
import ray
import pickle


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
            instance_kwargs,
            env,
            num_target_goals,
            num_eval_goals_sqrt,
            n_itr,
            exp_dir,
            goal_update_interval,
            snapshot_gap=500,
            n_initial_exploration_steps=1e3,
        ):
        self.env = env
        self.num_target_goals = num_target_goals
        self.goal_update_interval = goal_update_interval
        self.n_itr = n_itr

        env_pickled = pickle.dumps(env)
        eval_goals = self.env.sample_grid_goals(num_eval_goals_sqrt)
        self.agents = [Agent.remote(i, exp_dir, snapshot_gap) for i in range(num_agents)]

        futures = [agent.prepare_start.remote(
            seed,
            env_pickled,
            n_initial_exploration_steps,
            eval_goals,
            instance_kwargs,
            instance_kwargs.get('gpu_frac', 0.95)
        ) for agent, seed in zip(self.agents, seeds)]

        ray.get(futures)

    def train(self):
        agents = self.agents

        ray.get([agent.print_fields.remote() for agent in agents])

        for itr in range(self.n_itr):

            """----------------------- Compute q values to approximate goal distribution ------------------------"""

            target_goals = self.env.sample_goals(self.num_target_goals)
            futures = [agent.compute_q_values.remote(target_goals) for agent in agents]
            q_list = ray.get(futures)
            max_q = np.max(q_list, axis=0)

            if itr % self.goal_update_interval == 0:
                futures = [agent.update_goal_buffer.remote(target_goals, agent_q, max_q, q_list) for agent_q, agent in zip(q_list, agents)]
            else:
                futures = []

            for agent_q, agent in zip(q_list, agents):
                futures.extend([agent.update_replay_buffer.remote(), agent.update_policy.remote(), agent.save_snapshot.remote()])
            ray.get(futures)

            if itr == 0:
                ray.get([agent.finalize_graph.remote() for agent in agents])
