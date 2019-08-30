from meta_mb.trainers.bptt_trainer import BPTTTrainer
from meta_mb.policies.bptt_controllers.gt_ipopt_controller import GTIpoptController
from meta_mb.samplers.gt_sampler import GTSampler
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.envs.mb_envs.toy import FirstOrderEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf


EXP_NAME = 'bptt-gt-ipopt'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    repr = f"gt-{config['method_str']}-{config['initializer_str']}-"
    if config['env'] is HalfCheetahEnv:
        repr += 'hc'
        config['max_path_length'] = 100
    elif config['env'] is InvertedPendulumEnv:
        repr += 'ip'
        config['max_path_length'] = 100
    elif config['env'] is InvertedPendulumSwingUpEnv:
        repr += 'ipup'
        config['max_path_length'] = 100
    elif config['env'] is ReacherEnv:
        repr += 'reacher'
        config['max_path_length'] = 50
    elif config['env'] is FirstOrderEnv:
        repr += 'fo'
        config['max_path_length'] = 100

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

        policy = GTIpoptController(
            env=env,
            eps=config['eps'],
            discount=config['discount'],
            horizon=config['horizon'],
            method_str=config['method_str'],
            initializer_str=config['initializer_str'],
        )
        sampler = GTSampler(
            env=env,
            policy=policy,
            max_path_length=config['max_path_length'],
        )
        algo = BPTTTrainer(
            env=env,
            policy=policy,
            dynamics_model=None,
            sampler=sampler,
            dynamics_sample_processor=None,
            n_itr=1,
            initial_random_samples=False,
            initial_sinusoid_samples=False,
            dynamics_model_max_epochs=None,
            sess=sess,
            fit_model=False,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1],

        # Problem
        'env': [HalfCheetahEnv], #[ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        'discount': [1.0,],

        # Policy
        'initializer_str': ['zeros'], #['zeros', 'uniform'],
        'method_str': ['collocation'],
        'horizon': [20],
        'eps': [1e-6], #[1e-6, 1e-4, 1e-3],
        'deterministic_policy': [True],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
