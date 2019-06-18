import os
import json
import tensorflow as tf
import numpy as np
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'sac-running'


from meta_mb.algos.sac import SAC
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.mb_envs import *
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.sac_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics import MLPDynamicsModel
from meta_mb.baselines.nn_basline import NNValueFun
from meta_mb.logger import logger
from meta_mb.value_functions.utils import get_Q_function_from_variant
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline



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


        variant = {
        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': None,
                'observation_preprocessors_params': {}
            }
        },
        }
        Qs = get_Q_function_from_variant(variant, env)

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
        )
        # Load policy here

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            n_parallel=kwargs['n_parallel'],
        )

        sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        algo = SAC(
            policy = policy,
            discount=kwargs['discount'],
            learning_rate=kwargs['learning_rate'],
            training_environment=env,
            Qs=Qs,
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            n_itr=kwargs['n_itr'],
            sess=sess,
        )

        trainer.train()
    sess.__exit__()

if __name__ == '__main__':
    M = 256
    sweep_params = {
        'algo': ['sac'],
        'seed': [1, 2],
        'baseline': [LinearFeatureBaseline],
        'env': [HalfCheetahEnv],

        # Policy
        'policy_hidden_sizes': [(100, 100)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],

        # Env Sampling
        'num_rollouts': [10],
        'n_parallel': [5],

        # Problem Conf
        'n_itr': [1000],
        'max_path_length': [1000],
        # 'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        #
        # 'max_path_length': [500],
        # 'n_parallel': [10],
        #
        # 'learn_std': [True],
        # 'output_nonlinearity': [None],
        # 'init_std': [1.],
        #
        # 'learning_rate': [3e-4, 1e-2],
        'learning_rate' : [3e-4],
        'reward_scale': [1.0],
        'sampler_batch_size': [256],

        # 'num_minibatches': [1],
        # 'clip_eps': [.3],
        #
        # 'n_itr': [5000],
        # 'scope': [None],
        #
        # 'exp_tag': ['v0'],

        'Q_params': {
            'type': 'double_feedforward_Q_function',
            'kwargs': {
                'hidden_layer_sizes': (M, M),
                'observation_keys': ('observation'),
                'observation_preprocessors_params': {}
            }
        }

    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
