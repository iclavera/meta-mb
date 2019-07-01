import os
import json
import tensorflow as tf
import numpy as np
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'sac-q-3'


from meta_mb.algos.sac_edit import SAC_MB
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.mb_envs import *
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.sac_edit_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.logger import logger
from meta_mb.value_functions.value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
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

        env = normalize(kwargs['env']())

        Qs = [ValueFunction(name="q_fun_%d" % i,
                            obs_dim=int(np.prod(env.observation_space.shape)),
                            action_dim=int(np.prod(env.action_space.shape))
                            ) for i in range(2)]

        Q_targets = [ValueFunction(name="q_fun_target_%d" % i,
                                   obs_dim=int(np.prod(env.observation_space.shape)),
                                   action_dim=int(np.prod(env.action_space.shape))
                                   ) for i in range(2)]

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
            squashed=True
        )

        env_sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            n_parallel=kwargs['n_parallel'],
        )

        env_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        # dynamics_model = MLPDynamicsEnsemble('dynamics-ensemble',
        #                                      env=env,
        #                                      num_models=kwargs['num_models'],
        #                                      hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
        #                                      hidden_sizes=kwargs['dynamics_hidden_sizes'],
        #                                      output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
        #                                      learning_rate=kwargs['model_learning_rate'],
        #                                      batch_size=kwargs['dynamics_batch_size'],
        #                                      buffer_size=kwargs['dynamics_buffer_size'],
        #                                      )
        dynamics_model = ProbMLPDynamicsEnsemble('dynamics-ensemble',
                                                env=env,
                                                num_models=kwargs['num_models'],
                                                hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                                batch_size=kwargs['dynamics_batch_size'],
                                                learning_rate=kwargs['model_learning_rate'],
                                                buffer_size=kwargs['dynamics_buffer_size'],
                                                )

        dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = SAC_MB(
            policy=policy,
            discount=kwargs['discount'],
            learning_rate=kwargs['learning_rate'],
            env=env,
            dynamics_model=dynamics_model,
            Qs=Qs,
            Q_targets=Q_targets,
            reward_scale=kwargs['reward_scale'],
            sampler_batch_size=kwargs['sampler_batch_size']
        )

        trainer = Trainer(
            algo=algo,
            env=env,
            env_sampler=env_sampler,
            env_sample_processor=env_sample_processor,
            model_sample_processor = dynamics_sample_processor,
            dynamics_model=dynamics_model,
            policy=policy,
            n_itr=kwargs['n_itr'],
            dynamics_model_max_epochs=kwargs['dynamics_model_max_epochs'],
            policy_update_per_iteration=kwargs['policy_update_per_iteration'],
            num_model_rollouts=kwargs['num_model_rollouts'],
            sess=sess,
            env_max_replay_buffer_size=kwargs['env_replay_buffer_max_size'],
            model_max_replay_buffer_size=kwargs['model_replay_buffer_max_size'],
            speed_up_factor=kwargs['speed_up_factor']
        )

        trainer.train()
    sess.__exit__()


if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        """=============hyperparameter search start=============="""
        # 'seed': [3, 36, 369],
        # 'baseline': [LinearFeatureBaseline],
        # 'env': [HalfCheetahEnv],
        #
        # # Policy
        # 'policy_hidden_sizes': [(256, 256)],
        # 'policy_learn_std': [True],
        # 'policy_output_nonlinearity': [None],
        # 'policy_update_per_iteration': [40],
        # 'speed_up_factor': [100],
        # 'num_model_rollouts': [20],
        #
        # # Env Sampling
        # 'num_rollouts': [1],
        # 'n_parallel': [1],
        #
        # # replay_buffer
        # 'env_replay_buffer_max_size': [1e6],
        # 'model_replay_buffer_max_size': [4*1e7],
        #
        # # Problem Conf
        # 'n_itr': [400],
        # 'max_path_length': [1000],
        # 'discount': [0.99],
        # 'gae_lambda': [1.],
        # 'normalize_adv': [True],
        # 'positive_adv': [False],
        # 'learning_rate': [5e-5, 1e-4, 5e-4],
        # 'reward_scale': [1],
        # 'sampler_batch_size': [256],
        #
        # # Dynamics Model
        # 'num_models': [4],
        # 'model_learning_rate': [1e-3, 5e-4],
        # 'dynamics_hidden_sizes': [(200, 200, 200, 200)],
        # 'dyanmics_hidden_nonlinearity': ['relu'],
        # 'dyanmics_output_nonlinearity': [None],
        # 'dynamics_batch_size': [512, 1024],
        # 'dynamics_buffer_size': [10000],
        # 'dynamics_model_max_epochs': [10, 50],
        """=============hyperparameter search end=============="""
        'seed': [3],
        'baseline': [LinearFeatureBaseline],
        'env': [HalfCheetahEnv],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],
        'policy_update_per_iteration': [40],
        'speed_up_factor': [100],
        'num_model_rollouts': [20],

        # Env Sampling
        'num_rollouts': [1],
        'n_parallel': [1],

        # replay_buffer
        'env_replay_buffer_max_size': [1e6],
        'model_replay_buffer_max_size': [4*1e7],

        # Problem Conf
        'n_itr': [400],
        'max_path_length': [1000],
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
        'reward_scale': [1],
        'sampler_batch_size': [256],

        # Dynamics Model
        'num_models': [4],
        'model_learning_rate': [1e-3],
        'dynamics_hidden_sizes': [(200, 200, 200, 200)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_batch_size': [512],
        'dynamics_buffer_size': [10000],
        'dynamics_model_max_epochs': [50]


        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
