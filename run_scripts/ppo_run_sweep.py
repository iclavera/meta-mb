import os
import json
import tensorflow as tf
import numpy as np

from hw5.utils.utils import set_seed, ClassEncoder

from hw5.baselines.linear_baseline import LinearFeatureBaseline
from hw5.baselines.zero_baseline import ZeroBaseline

from hw5.envs.normalized_env import normalize
from hw5.algos.ppo import PPO
from hw5.trainers.mf_trainer import Trainer
from hw5.samplers.base import Sampler
from hw5.samplers.base import SampleProcessor
from hw5.policies.gaussian_mlp_policy import GaussianMLPPolicy
from hw5.logger import logger
from hw5.envs.mb_envs import *

import argparse


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + kwargs['name'] + f"/{str(kwargs['exp_num']).zfill(2)}"
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


        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
        )

        sample_processor = SampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
            use_gae=kwargs['use_gae']
        )


        algo = PPO(
            policy=policy,
            learning_rate=kwargs['learning_rate'],
            clip_eps=kwargs['clip_eps'],
            max_epochs=kwargs['num_ppo_steps'],
            entropy_bonus=kwargs['entropy_bonus'],
            use_clipper=kwargs['use_clipper'],
            use_entropy=kwargs['use_entropy'],
            use_ppo_obj=kwargs['use_ppo_obj']
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='HalfCheetah')
    parser.add_argument('--exp_name', type=str, default='test')
    parser.add_argument('--exp_num', type=int, default=0)
    parser.add_argument('--use_baseline', type=int, default=0)
    parser.add_argument('--use_ppo_obj', type=int, default=0)
    parser.add_argument('--use_clipper', type=int, default=0)
    parser.add_argument('--use_entropy', type=int, default=0)
    parser.add_argument('--use_gae', type=int, default=0)
    parser.add_argument('--discount', type=float, default=1)

    args = parser.parse_args()


    # baseline options: LinearFeatureBaseline, ZeroBaseline
    # discount factor: 0.99
    env_dict = {'HalfCheetah': HalfCheetahEnv, 'Swimmer': SwimmerEnv, 'Hopper': HopperEnv}

    if args.use_baseline:
        bsl = LinearFeatureBaseline
    else:
        bsl = ZeroBaseline

    name = args.exp_name + '_' + args.env_name
    env = env_dict[args.env_name]
    params = {
        'name': name,
        'exp_num': args.exp_num,
        'algo': 'ppo',
        'seed': args.exp_num,

        'baseline': bsl,

        'env': env,

        'num_rollouts': 100,        # this number can be smaller to make it faster
        'max_path_length': 100,

        'discount': 0.99,
        'gae_lambda': .975,
        'normalize_adv': True,
        'positive_adv': False,

        'hidden_sizes': (256, 256),
        'learn_std': True,
        'hidden_nonlinearity': tf.nn.tanh,
        'output_nonlinearity': None,
        'init_std': 1.,

        'learning_rate': 1e-3,
        'num_ppo_steps': 5,
        'num_minibatches': 1,
        'clip_eps': .2,
        'entropy_bonus': 1e-5,

        'n_itr': 300,
        'scope': None,
        'exp_tag': 'v0',
        'use_gae': args.use_gae,
        'use_entropy': args.use_entropy,
        'use_clipper': args.use_clipper,
        'use_ppo_obj': args.use_ppo_obj,
    }
    run_experiment(**params)
