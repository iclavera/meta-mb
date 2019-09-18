from meta_mb.trainers.ilqr_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.algos.ilqr import iLQR
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.dynamics.mlp_dynamics_ensemble_refactor import MLPDynamicsEnsemble
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.utils.utils import ClassEncoder
import json
import joblib
import os
import tensorflow as tf
import numpy as np


EXP_NAME = 'bptt-mb-ilqr'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    repr = "mb-ilqr-"
    if config['env'] is HalfCheetahEnv:
        repr += 'hc'
        config['max_path_length'] = 100
    elif config['env'] is InvertedPendulumEnv:
        repr += 'ip'
        config['max_path_length'] = 100
        config['policy_damping_factor'] = 5e-1
    elif config['env'] is InvertedPendulumSwingUpEnv:
        repr += 'ipup'
        config['max_path_length'] = 100
    elif config['env'] is ReacherEnv:
        repr += 'reacher'
        config['max_path_length'] = 50
        config['policy_damping_factor'] = 1e1
        config['alpha_init'] = 1e-2

    repr += f"-{config['horizon']}-{config['num_ilqr_iters']}-{config['c_1']}-{config['num_models']}"

    if config.get('model_path', None) is not None:
        repr += '-pretrain'
        # config['fit_model'] = False
        config['initializer_str'] = 'zeros'
        config['cem_num_rollouts'] = config['num_rollouts']
        config['initial_random_samples'] = False

    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '') + repr
    print(f'===================================== exp_dir = {exp_dir} =====================')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:

        env = config['env']()

        if config.get('model_path', None) is not None:
            data = joblib.load(config['model_path'])
            dynamics_model = data['dynamics_model']
            assert dynamics_model.obs_space_dims == env.observation_space.shape[0]
            assert dynamics_model.action_space_dims == env.action_space.shape[0]

        else:
            dynamics_model = MLPDynamicsEnsemble(
                name="dyn_model",
                env=env,
                learning_rate=config['learning_rate'],
                hidden_sizes=config['hidden_sizes_model'],
                weight_normalization=config['weight_normalization_model'],
                num_models=config['num_models'],
                valid_split_ratio=config['valid_split_ratio'],
                rolling_average_persitency=config['rolling_average_persitency'],
                hidden_nonlinearity=config['hidden_nonlinearity_model'],
                output_nonlinearity=config['output_nonlinearity_model'],
                batch_size=config['batch_size_model'],
            )

        sample_processor = ModelSampleProcessor()

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=config['policy_hidden_sizes'],
            learn_std=config['policy_learn_std'],
            hidden_nonlinearity=config['policy_hidden_nonlinearity'],
            output_nonlinearity=config['policy_output_nonlinearity'],
        )

        algo = iLQR(
            env=env,
            dynamics_model=dynamics_model,
            policy=policy,
            horizon=config['horizon'],
            initializer_str=config['initializer_str'],
            use_hessian_f=config['use_hessian_f'],
            num_ilqr_iters=config['num_ilqr_iters'],
            discount=config['discount'],
            mu_min=config['mu_min'],
            mu_max=config['mu_max'],
            mu_init=config['mu_init'],
            policy_damping_factor=config['policy_damping_factor'],
            delta_0=config['delta_0'],
            delta_init=config['delta_init'],
            alpha_init=config['alpha_init'],
            alpha_decay_factor=config['alpha_decay_factor'],
            c_1=config['c_1'],
            max_forward_iters=config['max_forward_iters'],
            max_backward_iters=config['max_backward_iters'],
            policy_buffer_size=config['policy_buffer_size'],
        )

        # if config['on_policy_freq'] > 1:
        #     cem_policy = MPCController(
        #         env=env,
        #         dynamics_model=dynamics_model,
        #         method_str='cem',
        #         num_rollouts=config['cem_num_rollouts'],
        #         discount=config['discount'],
        #         n_candidates=config['n_candidates'],
        #         horizon=config['horizon'],
        #         num_cem_iters=config['num_cem_iters'],
        #         deterministic_policy=config['cem_deterministic_policy'],
        #     )
        #
        #     cem_sampler = Sampler(
        #         env=env,
        #         policy=cem_policy,
        #         num_rollouts=config['cem_num_rollouts'],
        #         max_path_length=config['max_path_length'],
        #     )
        # else:
        #     cem_sampler = None

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
        )

        trainer = Trainer(
            algo=algo,
            env=env,
            env_sampler=sampler,
            policy=policy,
            dynamics_model=dynamics_model,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            steps_per_iter=config['steps_per_iter'],
            initial_random_samples=config['initial_random_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
        )
        trainer.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1, 2],
        'fit_model': [True],
        'on_policy_freq': [1],

        # Problem
        'env': [InvertedPendulumEnv,], #ReacherEnv, InvertedPendulumEnv, InvertedPendulumSwingUpEnv], #[ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        # HalfCheetah
        # 'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrain-mb-ppo/hc-1002019_09_04_21_10_23_0/params.pkl'],
        'n_itr': [101],
        'discount': [1],  # FIXME: does not support discount < 1!! need to modify J_val_1, J_val_2
        'horizon': [5],  # FIXME: 15

        # iLQR
        'initializer_str': ['zeros',],
        'num_ilqr_iters': [5], #[5, 10],
        'mu_min': [1e-6],
        'mu_max': [1e10],
        'mu_init': [1e-5],
        'delta_0': [2],
        'delta_init': [1.0],
        'alpha_init': [1e-1],
        'alpha_decay_factor': [3.0],
        'c_1': [1e-1],  # 1e-3
        'max_forward_iters': [10],
        'max_backward_iters': [10],
        'use_hessian_f': [False],
        'num_cem_iters_for_init': [5],

        # # CEM
        # 'n_candidates': [100],
        # 'num_cem_iters': [5],
        # 'cem_deterministic_policy': [True],
        # 'cem_num_rollouts': [20],

        # Training
        'num_rollouts': [5],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [True],
        'steps_per_iter': [1],

        # Policy
        'policy_hidden_sizes': [(32,)], #[tuple()], #[(64, 64)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tf.tanh],
        'policy_output_nonlinearity': [None],
        'policy_buffer_size': [10,],  # FIXME

        # Dynamics Model
        'num_models': [5],
        'hidden_nonlinearity_model': ['swish',],
        'output_nonlinearity_model': [None],
        'dynamic_model_epochs': [50],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'hidden_sizes_model': [(512, 512),],
        'batch_size_model': [64],
        'learning_rate': [0.001],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
