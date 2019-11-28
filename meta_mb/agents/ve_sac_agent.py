from meta_mb.logger import logger
from meta_mb.algos.gc_sac import SAC
from meta_mb.utils.utils import set_seed
from meta_mb.samplers.ve_gc_sampler import Sampler
from meta_mb.samplers.gc_mb_sample_processor import ModelSampleProcessor
from meta_mb.replay_buffers.gc_simple_replay_buffer import SimpleReplayBuffer
from meta_mb.policies.gc_gaussian_mlp_policy import GCGaussianMLPPolicy
from meta_mb.value_functions.gc_value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline

import pickle
import time
import numpy as np


class Agent(object):
    """
    Agent wrapper for self-play.
    Args:

    """
    def __init__(
            self,
            exp_dir,
            snapshot_gap,
            gpu_frac,
            seed,
            env_pickled,
            value_ensemble,
            n_initial_exploration_steps,
            instance_kwargs,
            eval_interval,
            greedy_eps,
    ):

        logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'],
            snapshot_mode='gap', snapshot_gap=snapshot_gap, log_suffix=f"_agent")

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
            self.eval_goals = env.eval_goals
            assert len(self.eval_goals) % instance_kwargs["num_rollouts"] == 0

            Qs = [ValueFunction(
                name="q_fun_%d" % i,
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                goal_dim=env.goal_dim,
                hidden_nonlinearity=instance_kwargs['vfun_hidden_nonlinearity'],
            ) for i in range(2)]

            self.Q_targets = [ValueFunction(
                name="q_fun_target_%d" % i,
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                goal_dim=env.goal_dim,
                hidden_nonlinearity=instance_kwargs['vfun_hidden_nonlinearity'],
            ) for i in range(2)]

            self.policy = GCGaussianMLPPolicy(
                goal_dim=env.goal_dim,
                name="policy",
                obs_dim=env.obs_dim,
                action_dim=env.act_dim,
                hidden_sizes=instance_kwargs['policy_hidden_sizes'],
                output_nonlinearity=instance_kwargs['policy_output_nonlinearity'],
                hidden_nonlinearity=instance_kwargs['policy_hidden_nonlinearity'],
                max_std=instance_kwargs['policy_max_std'],
                min_std=instance_kwargs['policy_min_std'],
                squashed=True,
            )

            self.sampler = Sampler(
                env_pickled=env_pickled,
                value_ensemble=value_ensemble,
                policy=self.policy,
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
                env_spec=env,
                max_replay_buffer_size=instance_kwargs['policy_max_replay_buffer_size'],
            )

            self.algo = SAC(
                replay_buffer=self.replay_buffer,
                policy=self.policy,
                discount=instance_kwargs['discount'],
                learning_rate=instance_kwargs['learning_rate'],
                env=env,
                Qs=Qs,
                Q_targets=self.Q_targets,
                reward_scale=instance_kwargs['reward_scale']
            )

            self.eval_interval = eval_interval
            self.greedy_eps = greedy_eps
            if instance_kwargs["policy_num_grad_steps"] == -1:
                self.num_grad_steps = self.sampler._timesteps_sampled_per_itr
            else:
                self.num_grad_steps = instance_kwargs["policy_num_grad_steps"]

            sess.run(tf.initializers.global_variables())

            """------------------------- initial exploration steps ----------------------------"""

            self.policy_itr = -1

            self.algo._update_target(tau=1.0)
            if n_initial_exploration_steps > 0:
                while self.replay_buffer._size < n_initial_exploration_steps:
                    paths = self.sampler.collect_rollouts(goals=env.sample_goals(mode=None, num_samples=self.sampler.num_rollouts),
                                                          greedy_eps=1, log=True, log_prefix='train-')
                    samples_data = self.sample_processor.process_samples(paths, eval=False, log='all', log_prefix='train-')
                    self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                                   samples_data['rewards'], samples_data['dones'], samples_data['next_observations'])

    def train(self, itr):
        with self.sess.as_default():

            """------------------- collect training samples with goal batch -------------"""

            t = time.time()
            paths = self.sampler.collect_rollouts(greedy_eps=self.greedy_eps, apply_action_noise=True,
                                                  log=True, log_prefix='train-')
            logger.logkv('TimeSampling', time.time() - t)

            samples_data = self.sample_processor.process_samples(paths, eval=False, log='all', log_prefix='train-')

            self.replay_buffer.add_samples(samples_data['goals'], samples_data['observations'], samples_data['actions'],
                                           samples_data['rewards'], samples_data['dones'], samples_data['next_observations'])

            """------------------------ train policy for one iteration ------------------"""

            t = time.time()
            self.policy_itr += 1
            self.algo.optimize_policy(itr=self.policy_itr, grad_steps=self.num_grad_steps)

            logger.logkv('TimeTrainPolicy', time.time() - t)

            if itr % self.eval_interval == 0:

                """-------------------------- Evaluation ------------------"""

                eval_paths = []
                for batch in np.split(self.eval_goals, len(self.eval_goals)//self.sampler.num_rollouts):
                    eval_paths.extend(self.sampler.collect_rollouts(goals=batch, greedy_eps=0, apply_action_noise=False,
                                                                    log=True, log_prefix='eval-'))
                _ = self.sample_processor.process_samples(eval_paths, eval=True, log='all', log_prefix='eval-')

            return paths

    def save_snapshot(self, itr):
        with self.sess.as_default():
            params = self._get_itr_snapshot(itr)
            logger.save_itr_params(itr, params)

    def _get_itr_snapshot(self, itr):
        return dict(itr=itr, policy=self.policy, env=self.env, Q_targets=tuple(self.Q_targets))

    def finalize_graph(self):
        self.sess.graph.finalize()

