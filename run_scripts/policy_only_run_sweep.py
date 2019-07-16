from meta_mb.trainers.policy_only_trainer import PolicyOnlyTrainer
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import HalfCheetahEnv, HopperEnv, AntEnv, Walker2dEnv
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
# from meta_mb.envs.blue.full_blue_env import FullBlueEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf
import joblib


EXP_NAME = 'mb-mpc'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    print(f'=====================================exp_dir = {exp_dir}=====================')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:

        env = config['env']()

        data = joblib.load(config['model_path'])
        dynamics_model = data['dynamics_model']

        policy = MPCController(
            name="policy",
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            use_cem=config['use_cem'],
            use_opt=config['use_opt'],
            num_cem_iters=config['num_cem_iters'],
            num_opt_iters=config['num_opt_iters'],
            opt_learning_rate=config['opt_learning_rate'],
            num_rollouts=config['num_rollouts'],
        )

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
            n_parallel=config['n_parallel'],
        )

        sample_processor = ModelSampleProcessor()

        algo = PolicyOnlyTrainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            sampler=sampler,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            initial_random_samples=config['initial_random_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1],
        'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrain-model-me-ppo/2019_07_15_18_34_37_0/params.pkl'],

        # Problem
        'env': [HalfCheetahEnv],
        'max_path_length': [100],
        'normalize': [False],
         'n_itr': [50],
        'discount': [1.],

        # Policy
        'n_candidates': [1000], # K
        'horizon': [5], # Tau
        'use_cem': [False],
        'num_cem_iters': [5],
        'use_opt': [True, False],
        'num_opt_iters': [10,],
        'opt_learning_rate': [1e-3],

        # Training
        'num_rollouts': [20],
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],

        # Dynamics Model
        'dynamic_model_epochs': [15],

        #  Other
        'n_parallel': [1],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
