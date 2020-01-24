import os
import json
import tensorflow as tf
import numpy as np
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = "script4"

from pdb import set_trace as st
from meta_mb.algos.maac import MAAC
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.mb_envs import *
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.maac_trainer import Trainer
from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.logger import logger
# from meta_mb.policies.mpc_controller import MPCController
from meta_mb.value_functions.value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline

from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='gap', snapshot_gap=5)
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    if isinstance(kwargs['env'](), HopperEnv):
        if kwargs['rollout_length_params'] == 'default':
            rollout_length_params = [20, 100, 1, 15]
        else:
            rollout_length_params = [20, 100, 1, 1]
    elif isinstance(kwargs['env'](), AntEnv):
        if kwargs['rollout_length_params'] == 'default':
            rollout_length_params = [20, 100, 1, 25]
        else:
            rollout_length_params = [20, 100, 1, 1]
    else:
        rollout_length_params = [20, 100, 1, 1]

    if isinstance(kwargs['env'](), HopperEnv) or isinstance(kwargs['env'](), Walker2dEnv):
        vfun_hidden_nonlineariy = 'tanh'
    elif isinstance(kwargs['env'](), HalfCheetahEnv) or isinstance(kwargs['env'](), AntEnv):
        vfun_hidden_nonlineariy = 'relu'

    if isinstance(kwargs['env'](), HalfCheetahEnv):
        policy_hidden_nonlinearity = 'relu'
    else:
        policy_hidden_nonlinearity = 'tanh'


    if isinstance(kwargs['env'](), HopperEnv) or isinstance(kwargs['env'](), HalfCheetahEnv) or isinstance(kwargs['env'](), AntEnv):
        T = 3
    elif isinstance(kwargs['env'](), Walker2dEnv) :
        T = 1


    with sess.as_default() as sess:

        # Instantiate classes
        set_seed(kwargs['seed'])

        baseline = kwargs['baseline']()

        env = normalize(kwargs['env']())
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))


        Qs = [ValueFunction(name="q_fun_%d" % i,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            hidden_nonlinearity=vfun_hidden_nonlineariy,) for i in range(2)]

        Q_targets = [ValueFunction(name="q_fun_target_%d" % i,
                                   obs_dim=obs_dim,
                                   action_dim=action_dim,
                                   hidden_nonlinearity=vfun_hidden_nonlineariy,) for i in range(2)]

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
            hidden_nonlinearity=policy_hidden_nonlinearity,
            squashed=True
        )


        train_env_sampler = BaseSampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],

        )

        train_env_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        eval_env_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        dynamics_model = ProbMLPDynamicsEnsemble('dynamics-ensemble',
                                                env=env,
                                                num_models=kwargs['num_models'],
                                                hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                                batch_size=kwargs['dynamics_batch_size'],
                                                learning_rate=kwargs['model_learning_rate'],
                                                buffer_size=kwargs['dynamics_buffer_size'],
                                                rolling_average_persitency=kwargs['rolling_average_persitency'],
                                                )
        algo = MAAC(
            policy=policy,
            discount=kwargs['discount'],
            learning_rate=kwargs['learning_rate'],
            target_entropy=kwargs['target_entropy'],
            env=env,
            dynamics_model=dynamics_model,
            obs_dim=obs_dim,
            action_dim=action_dim,
            Qs=Qs,
            Q_targets=Q_targets,
            num_actions_per_next_observation=kwargs['num_actions_per_next_observation'],
            prediction_type=kwargs['prediction_type'],
            T=T,
            actor_type=kwargs['actor_type'],
            critic_type=kwargs['critic_type'],
            H=kwargs['H'],
            model_used_ratio=kwargs['model_used_ratio'],
            experiment_name=EXP_NAME,
            exp_dir=exp_dir,
            target_update_interval=kwargs['n_train_repeats'],
        )

        # eval_policy = MPCController(
        #     name="mpc",
        #     env=env,
        #     dynamics_model=dynamics_model,
        #     discount=kwargs['discount'],
        #     n_candidates=kwargs['n_candidates'],
        #     horizon=kwargs['horizon'],
        #     use_cem=kwargs['use_cem'],
        #     num_cem_iters=kwargs['num_cem_iters'],
        #     Qs=Qs,
        # )
        #
        # eval_env_sampler = BaseSampler(
        #     env=env,
        #     policy=policy,
        #     num_rollouts=kwargs['num_rollouts'],
        #     max_path_length=kwargs['max_path_length'],
        #     mpc=eval_policy,
        #
        # )

        trainer = Trainer(
            algo=algo,
            env=env,
            train_env_sampler=train_env_sampler,
            # eval_env_sampler=eval_env_sampler,
            train_env_sample_processor=train_env_sample_processor,
            # eval_env_sample_processor=eval_env_sample_processor,
            dynamics_model=dynamics_model,
            policy=policy,
            # eval_policy=eval_policy,
            n_itr=kwargs['n_itr'],
            sess=sess,
            n_initial_exploration_steps=kwargs['n_initial_exploration_steps'],
            env_max_replay_buffer_size=kwargs['env_replay_buffer_max_size'],
            model_max_replay_buffer_size=kwargs['model_replay_buffer_max_size'],
            rollout_length_params=rollout_length_params,
            rollout_batch_size=kwargs['rollout_batch_size'],
            model_train_freq=kwargs['model_train_freq'],
            n_train_repeats=kwargs['n_train_repeats'],
            real_ratio=kwargs['real_ratio'],
            dynamics_model_max_epochs=kwargs['dynamics_model_max_epochs'],
            sampler_batch_size=kwargs['sampler_batch_size'],
            T=T,
        )

        trainer.train()
    sess.__exit__()


if __name__ == '__main__':
    sweep_params = {
        'seed': [1, 2, 3, 4],
        'baseline': [LinearFeatureBaseline],
        'env': [HalfCheetahEnv, AntEnv, Walker2dEnv, HopperEnv],
        'n_itr': [200],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],

        # Env Sampling
        'n_initial_exploration_steps': [5e3],
        'max_path_length': [1001],
        'num_rollouts': [1],

        # replay_buffer
        'env_replay_buffer_max_size': [1e6],
        'model_replay_buffer_max_size': [2e6],

        # Training
        'n_train_repeats': [8],
        'rollout_length_params': ['default'],
        'model_train_freq': [250],
        'rollout_batch_size': [100e3],
        'num_actions_per_next_observation': [5],
        'H': [2],  # Critic
        'target_entropy': [1],
        # 'num_eval_trajectories': [5],

        # STEVE： used in MAAC
        # None: used in MBPO
        'critic_type': ['STEVE'],
        # MAAC
        # None: used in MBPO
        'actor_type': ['MAAC'],
        'model_used_ratio': [0.5],

        # Problem Conf
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
        'prediction_type': ['none'],

        # Dynamics Model
        'num_models': [8],
        'dynamics_buffer_size': [1e6],
        'dynamics_model_max_epochs': [200],
        'rolling_average_persitency': [0.9],
        'sampler_batch_size': [256],
        'real_ratio': [.05],
        'dynamics_hidden_sizes': [(200, 200, 200, 200)],
        'model_learning_rate': [1e-3],
        'dyanmics_hidden_nonlinearity': ['swish', 'relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_batch_size': [256],
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
