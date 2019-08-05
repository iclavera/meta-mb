from meta_mb.trainers.policy_only_trainer import PolicyOnlyTrainer
from meta_mb.policies.gt_mpc_controller import GTMPCController
from meta_mb.samplers.gt_sampler import GTSampler
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf


EXP_NAME = 'gt-mpc'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    print(config['opt_learning_rate'], config['eps'], config['lmbda'])
    if config['method_str'] in ['cem', 'collocation', 'ddp']:
        config['n_itr'] = 1
        config['num_rollouts'] = 1
        logger.log(f'n_itr = 1, num_rollouts = 1!!!')
    repr = f"{config['controller_str']}-{config['method_str']}-"
    if config['env'] is HalfCheetahEnv:
        repr += 'hc'
        config['max_path_length'] = max_path_length = 1000
    elif config['env'] is InvertedPendulumEnv:
        repr += 'ip'
        config['max_path_length'] = max_path_length = 100
    elif config['env'] is InvertedPendulumSwingUpEnv:
        repr += 'ipup'
        config['max_path_length'] = max_path_length = 100
    elif config['env'] is ReacherEnv:
        repr += 'reacher'
        config['max_path_length'] = max_path_length = 50
    # repr += '-reg-' + str(config['reg_coef'])
    # if config['reg_str'] is not None:
    #     repr += config['reg_str']

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

        # dynamics_model = GTDynamics(
        #     name="dyn_model",
        #     env=env,
        #     num_rollouts=config['num_rollouts'],
        #     horizon=config['horizon'],
        #     max_path_length=config['max_path_length'],
        #     discount=config['discount'],
        #     n_parallel=config['n_parallel'],
        # )
        dynamics_model = None
        sample_processor = None

        policy = GTMPCController(
            name="policy",
            env=env,
            dynamics_model=dynamics_model,
            eps=config['eps'],
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            max_path_length= max_path_length, # config['max_path_length'],
            method_str=config['method_str'],
            n_parallel=config['n_parallel'],
            dyn_pred_str=config['dyn_pred_str'],
            reg_coef=config['reg_coef'],
            reg_str=config['reg_str'],
            initializer_str=config['initializer_str'],
            num_cem_iters=config['num_cem_iters'],
            num_opt_iters=config['num_opt_iters'],
            num_collocation_iters=config['num_collocation_iters'],
            num_ddp_iters=config['num_ddp_iters'],
            opt_learning_rate=config['opt_learning_rate'],
            persistency=config['persistency'],
            num_rollouts=config['num_rollouts'],
            alpha=config['alpha'],
            percent_elites=config['percent_elites'],
            lmbda=config['lmbda'],
        )

        sampler = GTSampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=max_path_length, # config['max_path_length'],
            dyn_pred_str=config['dyn_pred_str'],
        )

        algo = PolicyOnlyTrainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            sampler=sampler,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            initial_random_samples=config['initial_random_samples'],
            initial_sinusoid_samples=config['initial_sinusoid_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
            fit_model=config['fit_model'],
            plot_freq=config['plot_freq'],
            deterministic_policy=config['deterministic_policy'],
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1],
        'fit_model': [False],
        'plot_freq': [-1],

        # Problem
        'env': [HalfCheetahEnv], #[ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        # 'max_path_length': [50],  # [40, 80, 200]  # hardcoded in run_experiments
        'normalize': [False],
        'n_itr': [1],  # only matters for opt_policy
        'discount': [1.0,],
        'controller_str': ['gt'],

        # Policy
        'initializer_str': ['zeros'], #['zeros', 'uniform'],  # only matters for opt_act
        'reg_coef': [0], #[0.05, 0.1, 0.2], #[1, 0],
        'reg_str': ['tanh'], #['scale', 'poly', 'tanh'],
        'method_str': ['ipopt_shooting_w_policy'], #['opt_policy', 'opt_act'],  # ['opt_policy', 'opt_act', 'cem', 'rs']
        'dyn_pred_str': ['all'],  # UNUSED

        'num_opt_iters': [50,], #20, 40,],
        'opt_learning_rate': [1e-3], #[1e-3, 1e-2], #[1e-5, 1e-4, 1e-3], #1e-3,], #1e-2],
        'clip_norm': [-1], # UNUSED
        'eps': [1e-5], #[1e-6, 1e-4, 1e-3],
        'deterministic_policy': [True],

        # cem
        'horizon': [30],  # only matters for cem/rs/collocation
        'n_candidates': [1000],
        'num_cem_iters': [50],
        'alpha': [0.15],
        'percent_elites': [0.1],

        # collocation
        'lmbda': [1e0],
        'num_collocation_iters': [500*30],
        'persistency': [0.9],

        # DDP
        'num_ddp_iters': [5],

        # Training
        'num_rollouts': [1],  # number of experts
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [False],
        'initial_sinusoid_samples': [False],

        # Dynamics Model  # UNUSED
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'dynamic_model_epochs': [15],
        'weight_normalization_model': [False],  # FIXME: Doesn't work

        # MLP  # UNSUED
        'hidden_sizes_model': [(512, 512)],  # (500, 500)
        'batch_size_model': [64],
        'learning_rate': [0.001],

        # Recurrent  # UNUSED
        'hidden_sizes_model_rec': [(128,)],  # (500, 500)
        'batch_size_model_rec': [10],
        'backprop_steps': [100],
        'cell_type': ['lstm'],  # ['lstm', 'gru', 'rnn']
        'learning_rate_rec': [0.01],

        #  Other
        'n_parallel': [5],
    }

    # assert config['horizon'] == config['max_path_length']

    config_debug = config.copy()
    config_debug['max_path_length'] = [7]
    config_debug['num_opt_iters'] = [1]
    config_debug['horizon'] = [4]
    config_debug['num_models'] = [3]
    config_debug['num_rollouts'] = [2]
    config_debug['plot_freq'] = [1]
    config_debug['n_parallel'] = 4

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
    # print('================ runnning toy example ================')
    #run_sweep(run_experiment, config_debug, EXP_NAME, INSTANCE_TYPE)
