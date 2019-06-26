import os
import json
import pickle
import numpy as np
from tensorflow import tanh, ConfigProto
from multiprocessing import Process, Pipe
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.mujoco.walker2d_env import Walker2DEnv
from meta_mb.envs.mb_envs import AntEnv, Walker2dEnv, HalfCheetahEnv
from meta_mb.envs.mujoco.hopper_env import HopperEnv
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.trainers.mbmpo_trainer import Trainer
from meta_mb.trainers.parallel_mbmpo_trainer import ParallelTrainer
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'timing-parallel-mbmpo'


def init_vars(sender, config, policy, dynamics_model):
    import tensorflow as tf

    with tf.Session(config=config).as_default() as sess:

        # initialize uninitialized vars  (only initialize vars that were not loaded)
        uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
        sess.run(tf.variables_initializer(uninit_vars))

        policy_pickle = pickle.dumps(policy)
        dynamics_model_pickle = pickle.dumps(dynamics_model)

    sender.send((policy_pickle, dynamics_model_pickle))
    sender.close()


def run_experiment(**kwargs):
    # exp_dir = os.getcwd() + '/data/' + EXP_NAME
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + kwargs.get('exp_name', '')
    print("\n---------- experiment with dir {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    os.mkdir(exp_dir + '/Data/')
    os.mkdir(exp_dir + '/Model/')
    os.mkdir(exp_dir + '/Policy/')
    json.dump(kwargs, open(exp_dir + '/Data/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Model/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Policy/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)

    # Instantiate classes
    set_seed(kwargs['seed'])

    baseline = kwargs['baseline']()

    env = kwargs['env']() # Wrappers?

    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=kwargs['meta_batch_size'],
        hidden_sizes=kwargs['policy_hidden_sizes'],
        learn_std=kwargs['policy_learn_std'],
        hidden_nonlinearity=kwargs['policy_hidden_nonlinearity'],
        output_nonlinearity=kwargs['policy_output_nonlinearity'],
    )

    dynamics_model = MLPDynamicsEnsemble(
        'dynamics-ensemble',
         env=env,
         num_models=kwargs['num_models'],
         hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
         hidden_sizes=kwargs['dynamics_hidden_sizes'],
         output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
         learning_rate=kwargs['dynamics_learning_rate'],
         batch_size=kwargs['dynamics_batch_size'],
         buffer_size=kwargs['dynamics_buffer_size'],
         loss_str=kwargs['loss_str'],
    )

    '''-------- dumps and reloads -----------------'''

    baseline_pickle = pickle.dumps(baseline)
    env_pickle = pickle.dumps(env)

    receiver, sender = Pipe()
    p = Process(
        target=init_vars,
        name="init_vars",
        args=(sender, config, policy, dynamics_model),
        daemon=False,
    )
    p.start()
    policy_pickle, dynamics_model_pickle = receiver.recv()
    receiver.close()

    '''-------- following classes depend on baseline, env, policy, dynamics_model -----------'''

    worker_data_feed_dict = {
        'env_sampler': {
            'rollouts_per_meta_task': kwargs['real_env_rollouts_per_meta_task'],
            'meta_batch_size': kwargs['meta_batch_size'],
            'max_path_length': kwargs['max_path_length'],
            'parallel': kwargs['parallel'],
        },
        'dynamics_sample_processor': {
            'discount': kwargs['discount'],
            'gae_lambda': kwargs['gae_lambda'],
            'normalize_adv': kwargs['normalize_adv'],
            'positive_adv': kwargs['positive_adv'],
        },
    }

    worker_model_feed_dict = {}

    worker_policy_feed_dict = {
        'model_sampler': {
            'rollouts_per_meta_task': kwargs['rollouts_per_meta_task'],
            'meta_batch_size': kwargs['meta_batch_size'],
            'max_path_length': kwargs['max_path_length'],
            'dynamics_model': dynamics_model,
            'deterministic': kwargs['deterministic'],
        },
        'model_sample_processor': {
            'discount': kwargs['discount'],
            'gae_lambda': kwargs['gae_lambda'],
            'normalize_adv': kwargs['normalize_adv'],
            'positive_adv': kwargs['positive_adv'],
        },
        'algo': {
            'step_size': kwargs['step_size'],
            'inner_type': kwargs['inner_type'],
            'inner_lr': kwargs['inner_lr'],
            'meta_batch_size': kwargs['meta_batch_size'],
            'num_inner_grad_steps': kwargs['num_inner_grad_steps'],
            'exploration': kwargs['exploration'],
        }
    }

    trainer = ParallelTrainer(
        exp_dir=exp_dir,
        policy_pickle=policy_pickle,
        env_pickle=env_pickle,
        baseline_pickle=baseline_pickle,
        dynamics_model_pickle=dynamics_model_pickle,
        feed_dicts=[worker_data_feed_dict, worker_model_feed_dict, worker_policy_feed_dict],
        n_itr=kwargs['n_itr'],
        num_inner_grad_steps=kwargs['num_inner_grad_steps'],
        meta_steps_per_iter=kwargs['meta_steps_per_iter'],
        log_real_performance=kwargs['log_real_performance'],
        flags_need_query=kwargs['flags_need_query'],
        sample_from_buffer=kwargs['sample_from_buffer'],
        fraction_meta_batch_size=kwargs['fraction_meta_batch_size'],
        config=config,
        simulation_sleep=kwargs['simulation_sleep'],
    )

    trainer.train()


if __name__ == '__main__':

    sweep_params = {

        'flags_need_query': [
            [False, False, False],
            # [True, True, True],
        ],

        'seed': [1, 2,],

        'algo': ['mbmpo'],
        'baseline': [LinearFeatureBaseline],
        'env': [AntEnv, Walker2dEnv, HalfCheetahEnv],

        # Problem Conf
        'n_itr': [21],#[501],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],  # not implemented
        'meta_steps_per_iter': [1],  # 30 # Get rid of outer loop in effect

        # Real Env Sampling
        'real_env_rollouts_per_meta_task': [1],
        'parallel': [False],
        'fraction_meta_batch_size': [1.],
        'simulation_sleep': [50, 200],

        # Dynamics Model
        'num_models': [5],
        'dynamics_hidden_sizes': [(500, 500, 500)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [50,],  # UNUSED
        'dynamics_learning_rate': [1e-3],
        'dynamics_batch_size': [128],
        'dynamics_buffer_size': [10000],
        'deterministic': [False],
        'loss_str': ['L2'],


        # Policy
        'policy_hidden_sizes': [(64, 64)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tanh],
        'policy_output_nonlinearity': [None],

        # Meta-Algo
        'meta_batch_size': [20],  # Note: It has to be multiple of num_models
        'rollouts_per_meta_task': [20,],
        'num_inner_grad_steps': [1],
        'inner_lr': [0.001],
        'inner_type': ['log_likelihood'],
        'step_size': [0.01],
        'exploration': [False],
        'sample_from_buffer': [False],  #[True, False],# not implemented

        'scope': [None],
        'exp_tag': ['timing-parallel-mbmpo'],  # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

