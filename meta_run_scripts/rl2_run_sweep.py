from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.meta_envs.mujoco.ant_rand_direc import AntRandDirecEnv
from meta_mb.meta_envs.mujoco.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_mb.meta_envs.mujoco.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from meta_mb.meta_envs.mujoco.humanoid_rand_direc import HumanoidRandDirecEnv
from meta_mb.meta_envs.mujoco.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from meta_mb.meta_envs.mujoco.ant_rand_goal import AntRandGoalEnv

from meta_mb.envs.pr2.pr2_env import PR2ReacherEnv
from meta_mb.meta_envs.rl2_env import rl2env
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo import PPO
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers.meta_sampler import MetaSampler
from meta_mb.samplers.meta_samplers.rl2_sample_processor import RL2SampleProcessor
from meta_mb.policies.gaussian_rnn_policy import GaussianRNNPolicy
import os
import random
from meta_mb.logger import logger
import json
import numpy as np
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
import tensorflow as tf

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'rl2-pr2-reach-dr'

def run_experiment(**config):
    exp_name = EXP_NAME + '-' + str(random.randint(0, 1e9))
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + exp_name
    print("\n---------- experiment with dir {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    set_seed(config['seed'])
    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:

        baseline = config['baseline']()
        #timeskip = config['timeskip']
        #log_rand = config['log_rand']
        if config['env'] == 'PR2ReacherEnv':
            exp_type = config['exp_type']
            env = rl2env(normalize(config['env'](exp_type=exp_type)))
        else:
            env = rl2env(normalize(config['env']()))#log_rand=log_rand)))#timeskip=timeskip)))
        obs_dim = np.prod(env.observation_space.shape) + np.prod(env.action_space.shape) + 1 + 1 # obs + act + rew + done
        policy = GaussianRNNPolicy(
                name="meta-policy",
                obs_dim=obs_dim,
                action_dim=np.prod(env.action_space.shape),
                meta_batch_size=config['meta_batch_size'],
                hidden_sizes=config['hidden_sizes'],
                cell_type=config['cell_type']
            )

        sampler = MetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=config['rollouts_per_meta_task'],
            meta_batch_size=config['meta_batch_size'],
            max_path_length=config['max_path_length'],
            parallel=config['parallel'],
            envs_per_task=1,
        )

        sample_processor = RL2SampleProcessor(
            baseline=baseline,
            discount=config['discount'],
            gae_lambda=config['gae_lambda'],
            normalize_adv=config['normalize_adv'],
            positive_adv=config['positive_adv'],
        )

        algo = PPO(
            policy=policy,
            learning_rate=config['learning_rate'],
            max_epochs=config['max_epochs'],
            backprop_steps=config['backprop_steps'],
        )

        trainer = Trainer(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            n_itr=config['n_itr'],
            sess=sess,
        )
        trainer.train()


if __name__ == '__main__':

    sweep_params = {
        'algo': ['rl2'],
        'seed': [1, 2, 3],

        'baseline': [LinearFeatureBaseline],
        'env': [PR2ReacherEnv],
        'exp_type': ['reach'],
        'meta_batch_size': [100],
        "hidden_sizes": [(64,), (128,)],
        'backprop_steps': [50, 100, 200],
        "rollouts_per_meta_task": [10],
        "parallel": [True],
        "max_path_length": [30],
        "discount": [0.99],
        "gae_lambda": [1.0],
        "normalize_adv": [True],
        "positive_adv": [False],
        "learning_rate": [1e-3],
        "max_epochs": [5],
        "cell_type": ["lstm"],
        "num_minibatches": [1],
        "n_itr": [100],
        'exp_tag': ['v0'],
        'log_rand': [0, 1, 2]
        #'timeskip': [1, 2, 3, 4]
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
