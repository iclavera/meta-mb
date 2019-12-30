from meta_mb.logger import logger
from meta_mb.agents.goal_buffer_v2 import GoalSampler
from meta_mb.algos.gc_sac import SAC
from meta_mb.samplers.ve_gc_sampler import Sampler
from meta_mb.samplers.gc_mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gc_gaussian_mlp_policy import GCGaussianMLPPolicy
from meta_mb.value_functions.gc_value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.replay_buffers.gc_simple_replay_buffer import SimpleReplayBuffer

import time
import numpy as np


class Agent():
    """
    Agent wrapper for self-play.
    Args:

    """
    def __init__(
            self,
            agent_idx,
            config,
            env,
            n_initial_exploration_steps,
            instance_kwargs,
            eval_interval,
            greedy_eps,
    ):
        import tensorflow as tf
        self.agent_idx = agent_idx
        self.sess = sess = tf.Session(config=config)
        self.log_prefix = f"{agent_idx}-"
        self.env = env

        with sess.as_default():

            """----------------- Construct instances -------------------"""

            baseline = LinearFeatureBaseline()

            _eval_goals = env._sample_eval_goals()
            num_eval_goals = len(_eval_goals) // instance_kwargs['num_rollouts'] * instance_kwargs['num_rollouts']
            indices = np.random.choice(len(_eval_goals), size=num_eval_goals, replace=False)
            self.eval_goals = _eval_goals[indices]

            self.Qs = [ValueFunction(
                name=f"q_fun_{i}_{agent_idx}",
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                goal_dim=env.goal_dim,
                hidden_nonlinearity=instance_kwargs['vfun_hidden_nonlinearity'],
                output_nonlinearity=instance_kwargs['vfun_output_nonlinearity'],
                hidden_sizes=instance_kwargs['vfun_hidden_sizes'],
            ) for i in range(2)]

            self.Q_targets = [ValueFunction(
                name=f"q_fun_target_{i}_{agent_idx}",
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                goal_dim=env.goal_dim,
                hidden_nonlinearity=instance_kwargs['vfun_hidden_nonlinearity'],
                output_nonlinearity=instance_kwargs['vfun_output_nonlinearity'],
                hidden_sizes=instance_kwargs['vfun_hidden_sizes'],
            ) for i in range(2)]

            self.policy = GCGaussianMLPPolicy(
                goal_dim=env.goal_dim,
                name=f"policy_{agent_idx}",
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                hidden_sizes=instance_kwargs['policy_hidden_sizes'],
                output_nonlinearity=instance_kwargs['policy_output_nonlinearity'],
                hidden_nonlinearity=instance_kwargs['policy_hidden_nonlinearity'],
                max_std=instance_kwargs['policy_max_std'],
                min_std=instance_kwargs['policy_min_std'],
                squashed=True,
            )

            self.goal_sampler = GoalSampler(
                env=env,
                greedy_eps=instance_kwargs['goal_greedy_eps'],
            )

            self.sampler = Sampler(
                env=env,
                policy=self.policy,
                goal_sampler=self.goal_sampler,
                num_rollouts=instance_kwargs['num_rollouts'],
                max_path_length=instance_kwargs['max_path_length'],
                n_parallel=instance_kwargs['n_parallel'],
                action_noise_str=instance_kwargs['action_noise_str'],
            )

            self.sample_processor = ModelSampleProcessor(
                reward_fn=env.reward,
                achieved_goal_fn=env.get_achieved_goal,
                baseline=baseline,
                replay_k=instance_kwargs['replay_k'],
                discount=instance_kwargs['discount'],
                gae_lambda=instance_kwargs['gae_lambda'],
                normalize_adv=instance_kwargs['normalize_adv'],
                positive_adv=instance_kwargs['positive_adv'],
            )

            self.replay_buffer = SimpleReplayBuffer(
                env_spec=self.env,
                max_replay_buffer_size=instance_kwargs['policy_max_replay_buffer_size'],
            )

            self.algo = SAC(
                name=f"agent_{agent_idx}",
                replay_buffer=self.replay_buffer,
                policy=self.policy,
                discount=instance_kwargs['discount'],
                learning_rate=instance_kwargs['learning_rate'],
                env=env,
                Qs=self.Qs,
                Q_targets=self.Q_targets,
                reward_scale=instance_kwargs['reward_scale'],
                target_update_interval = instance_kwargs['target_update_interval'],
            )

            self.eval_interval = eval_interval
            self.greedy_eps = greedy_eps
            self.num_grad_steps = instance_kwargs["policy_num_grad_steps"]

            sess.run(tf.initializers.global_variables())

            """------------------------- initial exploration steps ----------------------------"""

            self.policy_itr = -1

            self.algo._update_target(tau=1.0)
            self.goal_sampler.set_goal_dist(mc_goals=env.sample_goals(mode=None, num_samples=self.sampler.num_rollouts),
                                            goal_dist=None)  # errors without this line
            if n_initial_exploration_steps > 0:
                while self.replay_buffer._size < n_initial_exploration_steps:
                    paths, _ = self.sampler.collect_rollouts(goals=env.sample_goals(mode=None, num_samples=self.sampler.num_rollouts),
                                                          greedy_eps=1, log=True, log_prefix=f'{agent_idx}-train-')
                    samples_data = self.sample_processor.process_samples(paths, eval=False, log='all', log_prefix=f'{agent_idx}-train-')
                    self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                                   samples_data['rewards'], samples_data['dones'], samples_data['next_observations'])

    def compute_q_values(self, goals):
        """

        Args:
            goals: (np.array) shape (num_goals, goal_dim)

        Returns:
            q_values: (np.array) shape (num_qs, num_goals)  # FIXME: take min? Epsilon greedy policy?
        """
        with self.sess.as_default():
            input_obs = np.tile(self.env.init_obs[np.newaxis, ...], (len(goals), 1))
            actions, _ = self.policy.get_actions(input_obs, goals)
            min_q_values = np.min([qfun.compute_values(input_obs, actions, goals) for qfun in self.Qs], axis=0)
            return min_q_values

    def update_goal_dist(self, mc_goals, expert_q_values, agent_q_values, log=True):
        diff = expert_q_values - agent_q_values
        if log:  # expert_q_values may be updated less frequently than agnet_q_values
            logger.logkv(self.log_prefix+'PChangePct', np.sum(diff < 0) / len(diff))

        diff = np.maximum(diff, 0)  # non-negative

        if log:  # log relative magnitude
            logger.logkv(self.log_prefix+'DiffOverQVal', np.mean(diff) / np.mean(expert_q_values))

        if np.sum(diff) == 0:
            goal_dist = np.ones((len(mc_goals),)) / len(mc_goals)
        else:
            goal_dist = diff / np.sum(diff)

        self.goal_sampler.set_goal_dist(mc_goals=mc_goals, goal_dist=goal_dist)

        if log:
            logger.logkv(self.log_prefix+'PMax', np.max(goal_dist))
            logger.logkv(self.log_prefix+'PMin', np.min(goal_dist))
            logger.logkv(self.log_prefix+'PStd', np.std(goal_dist))
            logger.logkv(self.log_prefix+'PMean', np.mean(goal_dist))

    def train(self, itr):
        with self.sess.as_default():

            """------------------- collect training samples with goal batch -------------"""

            t = time.time()
            paths, goal_samples_snapshot = self.sampler.collect_rollouts(greedy_eps=self.greedy_eps, apply_action_noise=True,
                                                                         log=True, log_prefix=self.log_prefix, count_timesteps=True)
            logger.logkv(self.log_prefix+'TimeSampling', time.time() - t)

            samples_data = self.sample_processor.process_samples(paths, eval=False, log='all', log_prefix=self.log_prefix+'train-')

            self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                           samples_data['rewards'], samples_data['dones'], samples_data['next_observations'])

            """------------------------ train policy for one iteration ------------------"""

            t = time.time()
            self.policy_itr += 1
            self.algo.optimize_policy(itr=self.policy_itr, grad_steps=self.num_grad_steps, log_prefix=self.log_prefix+'algo-')

            logger.logkv(self.log_prefix+'TimeTrainPolicy', time.time() - t)

            return paths, goal_samples_snapshot

    def collect_eval_paths(self):
        with self.sess.as_default():
            eval_paths = []
            for batch in np.split(self.eval_goals, len(self.eval_goals) // self.sampler.num_rollouts):
                paths, _ = self.sampler.collect_rollouts(goals=batch, greedy_eps=0, apply_action_noise=False, log=False,
                                                         count_timesteps=False)
                eval_paths.extend(paths)
            return eval_paths

    def finalize_graph(self):
        self.sess.graph.finalize()

    def save_snapshot(self, itr, goal_samples):
        with self.sess.as_default():
            params = dict(itr=itr, policy=self.policy, env=self.env, Q_targets=tuple(self.Q_targets), goal_samples=goal_samples)
            logger.save_itr_params(itr, params, f"agent_{self.agent_idx}_")
