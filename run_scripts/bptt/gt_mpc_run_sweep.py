from meta_mb.trainers.bptt_trainer import BPTTTrainer
from meta_mb.policies.bptt_controllers.gt_mpc_controller import GTMPCController
from meta_mb.samplers.gt_sampler import GTSampler
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf


EXP_NAME = 'bptt-gt-mpc'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    print(config['opt_learning_rate'], config['eps'], config['lmbda'])
    if config['method_str'] in ['cem', 'collocation', 'ddp']:
        config['n_itr'] = 1
        config['num_rollouts'] = 1
        logger.log(f'n_itr = 1, num_rollouts = 1!!!')
    if 'policy' in config['method_str']:
        if config['policy_filter'] is True:
            repr = f"{config['controller_str']}-{config['method_str']}_w_filter"
        else:
            repr = f"{config['controller_str']}-{config['method_str']}_wo_filter"
    else:
        repr = f"{config['controller_str']}-{config['method_str']}-"

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
        dynamics_model = None
        sample_processor = None

        policy = GTMPCController(
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            method_str=config['method_str'],
            num_cem_iters=config['num_cem_iters'],
            num_rollouts=config['num_rollouts'],
            alpha=config['alpha'],
            percent_elites=config['percent_elites'],
            deterministic_policy=config['deterministic_policy'],
        )

        sampler = GTSampler(
            env=env,
            policy=policy,
            max_path_length=config['max_path_length'],
        )

        algo = BPTTTrainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            sampler=sampler,
            dynamics_sample_processor=sample_processor,
            n_itr=1,
            initial_random_samples=config['initial_random_samples'],
            initial_sinusoid_samples=config['initial_sinusoid_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
            fit_model=False,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1],

        # Problem
        'env': [InvertedPendulumEnv], #[ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        'normalize': [False],
        'discount': [1.0,],

        # Policy
        'method_str': ['cem', 'rs'],
        'deterministic_policy': [True],

        # cem
        'horizon': [30],
        'n_candidates': [1000],
        'num_cem_iters': [50],
        'alpha': [0.15],
        'percent_elites': [0.1],

        # Training
        'num_rollouts': [1],  # number of experts
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [False],
        'initial_sinusoid_samples': [False],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
    # print('================ runnning toy example ================')
    #run_sweep(run_experiment, config_debug, EXP_NAME, INSTANCE_TYPE)
