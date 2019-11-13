from meta_mb.logger import logger
from meta_mb.agents.goal_buffer import GoalBuffer
from meta_mb.algos.gc_sac import SAC
from meta_mb.utils.utils import set_seed
from meta_mb.samplers.gc_sampler import GCSampler
from meta_mb.samplers.gc_mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gc_tanh_mlp_gaussian_policy import TanhGaussianMLPPolicy
from meta_mb.value_functions.gc_value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.replay_buffers.gc_simple_replay_buffer import SimpleReplayBuffer

import ray
import pickle
import numpy as np
import time


@ray.remote
class Agent(object):
    """
    Agent wrapper for self-play.
    Args:

    """
    def __init__(
            self,
            agent_idx,
            exp_dir,
            snapshot_gap,
            gpu_frac,
            seed,
            env_pickled,
            n_initial_exploration_steps,
            instance_kwargs,
            eval_interval,
            num_grad_step=None,
    ):

        self.agent_index = agent_idx
        logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'], snapshot_mode='gap', snapshot_gap=snapshot_gap, log_suffix=f"_agent_{agent_idx}")

        import tensorflow as tf

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_frac
        self.sess = sess = tf.Session(config=config)

        with sess.as_default():

            """----------------- Construct instances -------------------"""

            set_seed(seed)

            baseline = LinearFeatureBaseline()

            self.env = env = pickle.loads(env_pickled)

            Qs = [ValueFunction(name="q_fun_%d" % i,
                                obs_dim=env.obs_dim,
                                action_dim=env.act_dim,
                                goal_dim=env.goal_dim,
                                ) for i in range(2)]

            self.Q_targets = Q_targets = [ValueFunction(name="q_fun_target_%d" % i,
                                       obs_dim=env.obs_dim,
                                       action_dim=env.act_dim,
                                       goal_dim=env.goal_dim,
                                       ) for i in range(2)]

            self.policy = policy = TanhGaussianMLPPolicy(
                goal_dim=env.goal_dim,
                name="policy",
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                hidden_sizes=instance_kwargs['policy_hidden_sizes'],
                learn_std=instance_kwargs['policy_learn_std'],
                output_nonlinearity=instance_kwargs['policy_output_nonlinearity'],
            )

            self.goal_buffer = GoalBuffer(
                env=env,
                agent_index=self.agent_index,
                policy=policy,
                q_ensemble=Q_targets,
                max_buffer_size=instance_kwargs['max_goal_buffer_size'],
                alpha=instance_kwargs['goal_buffer_alpha'],
                sample_rule=instance_kwargs['sample_rule'],
                curiosity_percentage=instance_kwargs['curiosity_percentage'],
            )

            self.sampler = GCSampler(
                env=env,
                policy=policy,
                goal_buffer=self.goal_buffer,
                num_rollouts=instance_kwargs['num_rollouts'],
                max_path_length=instance_kwargs['max_path_length'],
                n_parallel=instance_kwargs['n_parallel'],
            )

            self.sample_processor = ModelSampleProcessor(
                reward_fn=env.reward,
                achieved_goal_fn=env.get_achieved_goal,
                baseline=baseline,
                discount=instance_kwargs['discount'],
                gae_lambda=instance_kwargs['gae_lambda'],
                normalize_adv=instance_kwargs['normalize_adv'],
                positive_adv=instance_kwargs['positive_adv'],
            )

            self.algo = SAC(
                policy=policy,
                discount=instance_kwargs['discount'],
                learning_rate=instance_kwargs['learning_rate'],
                env=env,
                Qs=Qs,
                Q_targets=Q_targets,
                reward_scale=instance_kwargs['reward_scale']
            )

            self.eval_interval = eval_interval
            self.num_grad_steps = self.sampler.total_samples if num_grad_step is None else num_grad_step
            self.replay_buffer = SimpleReplayBuffer(self.env, instance_kwargs['max_replay_buffer_size'])

            sess.run(tf.initializers.global_variables())

            """------------------------- initial exploration steps ----------------------------"""

            self.itr = -1

            self.algo._update_target(tau=1.0)
            if n_initial_exploration_steps > 0:
                while self.replay_buffer._size < n_initial_exploration_steps:
                    paths = self.sampler.collect_rollouts(log=True, log_prefix='train-', random=True)
                    samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                    self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                                   samples_data['rewards'], samples_data['dones'], samples_data['next_observations'])

    def compute_q_values(self, sample_goals):
        with self.sess.as_default():
            min_q = self.goal_buffer.compute_min_q(sample_goals)

        return min_q

    def update_goal_buffer(self, sample_goals, q_list):
        t = time.time()
        self.goal_buffer.refresh(sample_goals, q_list, log=True)
        logger.logkv('TimeGoalSampling', time.time() - t)

    def update_replay_buffer(self):
        with self.sess.as_default():

            t = time.time()
            paths = self.sampler.collect_rollouts(log=True, log_prefix='train-')
            samples_data = self.sample_processor.process_samples(paths, replay_strategy='future', log='all', log_prefix='train-')
            self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                           samples_data['rewards'], samples_data['dones'], samples_data['next_observations'])
            t = time.time() - t
            logger.logkv('TimeSampling', time.time() - t)

            """-------------------------- Evaluation ------------------"""

            if self.itr % self.eval_interval == 0:
                eval_paths = self.sampler.collect_rollouts(eval=True, log=True, log_prefix='eval-')
                _ = self.sample_processor.process_samples(eval_paths, replay_strategy=None, log='all', log_prefix='eval-', log_all=False)

                logger.dumpkvs()

    def update_policy(self):
        t = time.time()
        self.itr += 1
        with self.sess.as_default():
            self.algo.optimize_policy(self.replay_buffer, self.itr, self.num_grad_steps)

        logger.logkv('TimeTrainPolicy', time.time() - t)

    def save_snapshot(self):
        with self.sess.as_default():
            params = self._get_itr_snapshot(self.itr)
            logger.save_itr_params(self.itr, params)

    def _get_itr_snapshot(self, itr):
        return dict(itr=itr, policy=self.policy, env=self.env, Q_targets=tuple(self.Q_targets))

    def finalize_graph(self):
        self.sess.graph.finalize()

