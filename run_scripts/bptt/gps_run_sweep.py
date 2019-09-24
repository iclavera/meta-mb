from meta_mb.trainers.ilqr_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.algos.gps import GPS
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.dynamics.mlp_dynamics_ensemble_refactor import MLPDynamicsEnsemble
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.envs.mb_envs.half_cheetah_quad_reward import HalfCheetahEnvQuadReward
from meta_mb.utils.utils import ClassEncoder
import json
import joblib
import os
import tensorflow as tf
import numpy as np


EXP_NAME = 'baseline-gps'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    repr = ""
    if config['env'] is HalfCheetahEnv:
        repr += 'hc'
        config['max_path_length'] = 100
    elif config['env'] is HalfCheetahEnvQuadReward:
        repr += 'hcq'
        config['max_path_length'] = 100
        # config['policy_damping_factor'] = 5e0
    elif config['env'] is InvertedPendulumEnv:
        repr += 'ip'
        config['max_path_length'] = 100
        # config['policy_damping_factor'] = 2e-1
    elif config['env'] is InvertedPendulumSwingUpEnv:
        repr += 'ipup'
        config['max_path_length'] = 100
    elif config['env'] is ReacherEnv:
        repr += 'reacher'
        config['max_path_length'] = 50
        # config['policy_damping_factor'] = 1e2
    elif config['env'] is HopperEnv:
        repr += 'hopper'
        config['max_path_length'] = 100
    elif config['env'] is AntEnv:
        repr += 'ant'
        config['max_path_length'] = 100

    if config['use_saved_params']:
        params_path = f'/home/yunzhi/mb/meta-mb/data/bptt-mb-ilqr-policy/params/{repr}.json'
        with open(params_path) as f:
            saved_config = json.load(f)

        saved_config['env'] = config['env']
        saved_config['policy_hidden_nonlinearity'] = config['policy_hidden_nonlinearity']
        config = saved_config

    repr += f"-{config['horizon']}-{config['num_ilqr_iters']}-{config['c_1']}-{config['num_models']}"

    if config.get('model_path', None) is not None:
        repr += '-pretrain'
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

        algo = GPS(
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
            damping_str=config['damping_str'],
            delta_0=config['delta_0'],
            delta_init=config['delta_init'],
            alpha_init=config['alpha_init'],
            alpha_decay_factor=config['alpha_decay_factor'],
            c_1=config['c_1'],
            max_forward_iters=config['max_forward_iters'],
            max_backward_iters=config['max_backward_iters'],
            policy_buffer_size=config['policy_buffer_size'],
            use_hessian_policy=config['use_hessian_policy'],
            learning_rate=config['policy_learning_rate'],
            batch_size=config['batch_size'],
            num_gradient_steps=config['num_gradient_steps'],
            verbose=config['verbose'],
        )

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
        'env': [InvertedPendulumEnv], #ReacherEnv, InvertedPendulumEnv, InvertedPendulumSwingUpEnv], #[ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        'verbose': [True],  # FIXME: TURN OFF BEFORE SENDING TO EC2!!!
        'use_saved_params': [False],
        'n_itr': [301],

        # Problem
        'seed': [1,],
        'discount': [1],  # FIXME: does not support discount < 1!! need to modify J_val_1, J_val_2
        'horizon': [5, 15,],

        # iLQR
        'initializer_str': ['zeros',],
        'num_ilqr_iters': [1],
        'mu_min': [1e-6],  # UNUSED
        'mu_max': [1e10],  # UNUSED
        'mu_init': [1e-5,], # 1e1
        'delta_0': [2],  # UNUSED
        'delta_init': [1.0],  # UNUSED
        'alpha_init': [1e-2, 1e-3],
        'alpha_decay_factor': [3.0],
        'c_1': [1e-1, 1e-3],
        'max_forward_iters': [10],
        'max_backward_iters': [5],
        'use_hessian_f': [False],  # False
        'use_hessian_policy': [False],
        'policy_damping_factor': [1e-4,],  # UNUSED if not use_hessian_policy
        'damping_str': ['Q'],

        # Training
        'num_rollouts': [5],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [True],
        'steps_per_iter': [1,],
        'batch_size': [200],

        # Policy
        'policy_hidden_sizes': [tuple(), (16,),],# [tuple()], #[(64, 64)],[(32,)], #
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tf.tanh],
        'policy_output_nonlinearity': [None],
        'policy_buffer_size': [200],
        'policy_learning_rate': [1e-3,],
        'num_gradient_steps': [10],

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
