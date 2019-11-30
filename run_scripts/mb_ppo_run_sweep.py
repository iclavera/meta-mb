import os
import json
import tensorflow as tf
import numpy as np
from hw5.utils.utils import set_seed, ClassEncoder
from hw5.baselines.linear_baseline import LinearFeatureBaseline
from hw5.envs.mb_envs import *
from hw5.envs.normalized_env import normalize
from hw5.algos.ppo import PPO

from hw5.trainers.mbmf_trainer import Trainer

from hw5.samplers.base import Sampler
from hw5.samplers.base import SampleProcessor
from hw5.samplers.mb_sampler import MBSampler
from hw5.policies.gaussian_mlp_policy import GaussianMLPPolicy
from hw5.dynamics.mlp_dynamics import MLPDynamicsModel
from hw5.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from hw5.logger import logger
from hw5.samplers.mb_sample_processor import ModelSampleProcessor
import argparse


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/parallel_mb_ppo/' + kwargs['name'] + '/' + kwargs.get('exp_name', '')
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

        env = normalize(kwargs['env']()) # Wrappers?

        policy = GaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            hidden_nonlinearity=kwargs['policy_hidden_nonlinearity'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
        )
        if kwargs['ensemble']:
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
            dynamics_model = MLPDynamicsModel('dynamics-vanilla',
                                              env=env,
                                              hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                              hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                              output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                              batch_size=kwargs['dynamics_batch_size'],
                                              learning_rate=kwargs['dynamics_learning_rate'],
                                              weight_normalization=False,
                                              normalize_input=True,
                                              buffer_size=kwargs['dynamics_buffer_size'],
                                              )

        env_sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
        )

        model_sampler = MBSampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['imagined_num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            dynamics_model=dynamics_model,
            deterministic=kwargs['deterministic'],
        )

        dynamics_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        model_sample_processor = SampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = PPO(
            policy=policy,
            learning_rate=kwargs['learning_rate'],
            clip_eps=kwargs['clip_eps'],
            max_epochs=kwargs['num_ppo_steps'],
        )

        trainer = Trainer(
                algo=algo,
            policy=policy,
            env=env,
            model_sampler=model_sampler,
            env_sampler=env_sampler,
            model_sample_processor=model_sample_processor,
            dynamics_sample_processor=dynamics_sample_processor,
            dynamics_model=dynamics_model,
            n_itr=kwargs['n_itr'],
            dynamics_model_max_epochs=kwargs['dynamics_max_epochs'],
            log_real_performance=kwargs['log_real_performance'],
            steps_per_iter=kwargs['steps_per_iter'],
            sample_from_buffer=True,
            sess=sess,
        )

        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--exp_num', type=int, default=0)
    parser.add_argument('--ensemble', type=int, default=0)

    args = parser.parse_args()
    env_dict = {'HalfCheetah': HalfCheetahEnv, 'Swimmer': SwimmerEnv, 'Hopper': HopperEnv}
    env = env_dict[args.env_name]
    params = {
            'name': args.exp_name,
            'seed': args.exp_num,
            'exp_num': args.exp_num,
            'algo': 'meppo',
            'baseline': LinearFeatureBaseline,
            'env': env,
            'ensemble': args.ensemble,

            # Problem Conf
            'n_itr': 51,
            'max_path_length': 200,
            'discount': 0.99,
            'gae_lambda': 1,
            'normalize_adv': True,
            'positive_adv': False,
            'log_real_performance': True,
            'steps_per_iter': (10, 10),     #(50, 50)],

            # Real Env Sampling
            'num_rollouts': 5,

            # Dynamics Model
            'num_models': 5,
            'dynamics_hidden_sizes': (512, 512),
            'dyanmics_hidden_nonlinearity': 'relu',
            'dyanmics_output_nonlinearity': None,
            'dynamics_max_epochs': 50,
            'dynamics_learning_rate': 1e-3,
            'dynamics_batch_size': 256,
            'dynamics_buffer_size': 10000,
            'deterministic': True,

            # Policy
            'policy_hidden_sizes': (64, 64),
            'policy_learn_std': True,
            'policy_hidden_nonlinearity': tf.tanh,
            'policy_output_nonlinearity': None,

            # Algo
            'clip_eps': 0.2,                # 0.3, 0.1],
            'learning_rate': 1e-3,          # 5e-4],
            'num_ppo_steps': 5,
            'imagined_num_rollouts': 20,    #50],
            'scope': None,
            'exp_tag': 'mb_ppo_all',        # For changes besides hyperparams
            'recurrent': False,
        }

    run_experiment(**params)

