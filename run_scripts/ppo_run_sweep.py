import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.cassie.cassie_env import CassieEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo import PPO
from meta_mb.envs.blue.full_blue_env import FullBlueEnv
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.single_sample_processor import SingleSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'cassie-balancing-entropy'


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

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['hidden_sizes'],
            learn_std=kwargs['learn_std'],
            hidden_nonlinearity=kwargs['hidden_nonlinearity'],
            output_nonlinearity=kwargs['output_nonlinearity'],
            init_std=kwargs['init_std'],
        )

        # Load policy here

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            n_parallel=kwargs['n_parallel'],
        )

        sample_processor = SingleSampleProcessor(
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
            entropy_bonus=kwargs['entropy_bonus'],
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

if __name__ == '__main__':
    sweep_params = {
        'algo': ['ppo'],
        'seed': [1, 2],

        'baseline': [LinearFeatureBaseline],

        'env': [An],

        'num_rollouts': [100],
        'max_path_length': [200, 300],
        'n_parallel': [10],

        'discount': [0.99],
        'gae_lambda': [.975, 0.5],
        'normalize_adv': [True],
        'positive_adv': [False],

        'hidden_sizes': [(256, 256)],
        'learn_std': [True],
        'hidden_nonlinearity': [tf.nn.tanh],
        'output_nonlinearity': [None],
        'init_std': [1.],

        'learning_rate': [1e-3],
        'num_ppo_steps': [2, 5],
        'num_minibatches': [1],
        'clip_eps': [.3],
        'entropy_bonus': [1e-2, 1e-3, 1e-4, 1e-5],

        'n_itr': [5000],
        'scope': [None],

        'exp_tag': ['v0']
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
