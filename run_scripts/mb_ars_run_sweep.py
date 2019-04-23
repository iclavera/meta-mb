import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from meta_mb.envs.mujoco.ant_env import AntEnv
from meta_mb.envs.mujoco.humanoid_env import HumanoidEnv
from meta_mb.envs.mujoco.walker2d_env import Walker2DEnv
from meta_mb.envs.mujoco.hopper_env import HopperEnv
from meta_mb.envs.mujoco.swimmer_env import SwimmerEnv
from meta_mb.envs.blue.blue_env import BlueReacherEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.envs.img_wrapper_env import image_wrapper
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv
from meta_mb.optimizers.random_search_optimizer import RandomSearchOptimizer
from meta_mb.trainers.ars_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.ars_sampler.ars_sample_processor import ARSSamplerProcessor
from meta_mb.samplers.mbmpo_samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.samplers.ars_sampler.ars_sampler import ARSSampler
from meta_mb.policies.np_linear_policy import LinearPolicy
from meta_mb.policies.np_nn_policy import NNPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.unsupervised_learning.vae import VAE

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'ars-32'


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)
    with sess.as_default() as sess:

        # Instantiate classes
        set_seed(kwargs['seed'])

        baseline = kwargs['baseline']()

        if not kwargs['use_images']:
            env = normalize(kwargs['env']())

        else:
            vae = VAE(latent_dim=8)
            env = image_wrapper(normalize(kwargs['env']()), vae=vae, latent_dim=32)

        policy = NNPolicy(name="policy",
                          obs_dim=np.prod(env.observation_space.shape),
                          action_dim=np.prod(env.action_space.shape),
                          hidden_sizes=kwargs['hidden_sizes']
                          )

        if kwargs['deterministic']:
            dynamics_model = MLPDynamicsEnsemble('dynamics-ensemble',
                                                 env=env,
                                                 num_models=kwargs['num_models'],
                                                 hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                 hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                 output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                                 learning_rate=kwargs['dynamics_learning_rate'],
                                                 batch_size=kwargs['dynamics_batch_size'],
                                                 buffer_size=kwargs['dynamics_buffer_size'],
                                                 )

        else:
            dynamics_model = ProbMLPDynamicsEnsemble('dynamics-ensemble',
                                                 env=env,
                                                 num_models=kwargs['num_models'],
                                                 hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                 hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                 output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                                 learning_rate=kwargs['dynamics_learning_rate'],
                                                 batch_size=kwargs['dynamics_batch_size'],
                                                 buffer_size=kwargs['dynamics_buffer_size'],
                                                 )

        dynamics_model = None
        # assert kwargs['rollouts_per_policy'] % kwargs['num_models'] == 0

        env_sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            # n_parallel=kwargs['n_parallel'],
        )

        # TODO: I'm not sure if it works with more than one rollout per model

        model_sampler = ARSSampler(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            rollouts_per_policy=kwargs['rollouts_per_policy'],
            max_path_length=kwargs['max_path_length'],
            num_deltas=kwargs['num_deltas'],
            n_parallel=2,
        )

        dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        ars_sample_processor = ARSSamplerProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = RandomSearchOptimizer(
            policy=policy,
            learning_rate=kwargs['learning_rate'],
            num_deltas=kwargs['num_deltas'],
            percentile=kwargs['percentile']
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            model_sampler=model_sampler,
            env_sampler=env_sampler,
            ars_sample_processor=ars_sample_processor,
            dynamics_sample_processor=dynamics_sample_processor,
            dynamics_model=dynamics_model,
            num_deltas=kwargs['num_deltas'],
            n_itr=kwargs['n_itr'],
            dynamics_model_max_epochs=kwargs['dynamics_max_epochs'],
            log_real_performance=kwargs['log_real_performance'],
            steps_per_iter=kwargs['steps_per_iter'],
            delta_std=kwargs['delta_std'],
            sess=sess
        )

        trainer.train()


if __name__ == '__main__':

    sweep_params = {
        'seed': [1, 2],

        'algo': ['ars'],
        'baseline': [LinearFeatureBaseline],
        'env': [InvertedPendulumEnv, HalfCheetahEnv, Walker2DEnv, HopperEnv, HumanoidEnv],
        'use_images': [False],

        # Problem Conf
        'n_itr': [200],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],
        'steps_per_iter': [(10, 10)],

        # Real Env Sampling
        'num_rollouts': [2],
        'parallel': [True],

        # Dynamics Model
        'num_models': [1],
        'dynamics_hidden_sizes': [(500, 500)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [25],
        'dynamics_learning_rate': [1e-3],
        'dynamics_batch_size': [64],
        'dynamics_buffer_size': [10000],
        'deterministic': [True],

        # Meta-Algo
        'learning_rate': [0.02, 0.01, 0.05, 0.005],
        'num_deltas': [4, 16, 32],
        'rollouts_per_policy': [1],
        'percentile': [0.5],
        'delta_std': [0.01, 0.03, 0.05],
        'hidden_sizes': [(128, 128)],

        'scope': [None],
        'exp_tag': [''], # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

