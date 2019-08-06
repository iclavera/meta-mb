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
from meta_mb.envs.mb_envs import AntEnv, Walker2dEnv, HalfCheetahEnv, HopperEnv
from meta_mb.envs.mujoco.hopper_env import HopperEnv
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.envs.pr2.real_pr2_reach_env import PR2ReachEnv
from meta_mb.envs.pr2.real_pr2_peg_env import PR2PegEnv
from meta_mb.envs.pr2.real_pr2_water_bottle import PR2BottleEnv
from meta_mb.trainers.mbmpo_trainer import Trainer
from meta_mb.trainers.parallel_mbmpo_trainer import ParallelTrainer
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.envs.normalized_env import normalize

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'shape-mod-demo'


import random

    
def init_vars(sender, config, policy, dynamics_model):
    import tensorflow as tf

    with tf.Session(config=config).as_default() as sess:

        # initialize uninitialized vars  (only initialize vars that were not loaded)
        # uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
        # sess.run(tf.variables_initializer(uninit_vars))
        sess.run(tf.global_variables_initializer())

        policy_pickle = pickle.dumps(policy)
        dynamics_model_pickle = pickle.dumps(dynamics_model)

    sender.send((policy_pickle, dynamics_model_pickle))
    sender.close()

def run_experiment(**kwargs):
    exp_name = EXP_NAME + '-' + str(random.randint(0, 1e9))
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + exp_name
    print("\n---------- experiment with dir {} ---------------------------".format(exp_dir))
    logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    os.makedirs(exp_dir + '/Data/', exist_ok=True)
    os.makedirs(exp_dir + '/Model/', exist_ok=True)
    os.makedirs(exp_dir + '/Policy/', exist_ok=True)
    json.dump(kwargs, open(exp_dir + '/Data/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Model/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    json.dump(kwargs, open(exp_dir + '/Policy/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)
    run_base(exp_dir, **kwargs)

def run_base(exp_dir, **kwargs):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)

    # Instantiate classes
    set_seed(kwargs['seed'])

    baseline = kwargs['baseline']()
    if kwargs['env'] == 'PR2ReachEnv':
        env = normalize(PR2ReachEnv(exp_type='reach_joint',
                                    vel_penalty=1.25e-3,#kwargs['vel_penalty'],
                                    torque_penalty=1.25e-3,#kwargs['torque_penalty'],
                                    max_torques=kwargs['max_torques']))
        simulation_sleep = 0
    elif kwargs['env'] == 'PR2PegEnv':
        env = normalize(PR2PegEnv())
        simulation_sleep = 0
    elif kwargs['env'] == 'PR2BottleEnv':
        env = normalize(PR2BottleEnv())
        simulation_sleep = 0
    else:
        raise NotImplementedError

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
        rolling_average_persitency=kwargs['rolling_average_persitency'],
    )


    '''-------- dumps and reloads -----------------'''

    baseline_pickle = pickle.dumps(baseline)
    env_pickle = pickle.dumps(env)

    receiver, sender = Pipe()
    p = Process(
        target=init_vars,
        name="init_vars",
        args=(sender, config, policy, dynamics_model),
        daemon=True,
    )
    p.start()
    policy_pickle, dynamics_model_pickle = receiver.recv()
    receiver.close()

    '''-------- following classes depend on baseline, env, policy, dynamics_model -----------'''

    worker_data_feed_dict = {
        'env_sampler': {
            #'rollouts_per_meta_task': kwargs['real_env_rollouts_per_meta_task'],
            'num_rollouts': kwargs['num_rollouts'],
            'max_path_length': kwargs['max_path_length'],
            # 'parallel': kwargs['parallel'],
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
        initial_random_samples=kwargs['initial_random_samples'],
        flags_need_query=kwargs['flags_need_query'],
        num_rollouts_per_iter=int(kwargs['meta_batch_size'] * kwargs['fraction_meta_batch_size']),
        config=config,
        simulation_sleep=simulation_sleep,
        sampler_str='bptt'
    )

    trainer.train()


if __name__ == '__main__':

    sweep_params = {

        'flags_need_query': [
            [False, False, False],
            # [True, True, True],
        ],
        'rolling_average_persitency': [0.1],

        'seed': [2],

        'n_itr': [200], #600 whooooo
        'num_rollouts': [1],
        'simulation_sleep_frac': [1],
        'env': ['PR2ReachEnv'],
        'exp_type': ['reach'],
        'torque_penalty': [1.25e-2],
        'vel_penalty': [1.25e-1],
        'max_torques': [(3, 3, 2, 2, 1, 0.5, 1)],

        # Problem Conf
        'num_models': [5],
        'algo': ['mbmpo'],
        'baseline': [LinearFeatureBaseline],
        'max_path_length': [30],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],  # not implemented
        'meta_steps_per_iter': [1],  # UNUSED

        # Real Env Sampling
        'real_env_rollouts_per_meta_task': [1],
        'parallel': [False],
        'fraction_meta_batch_size': [.2],
         'meta_batch_size': [5],  # Note: It has to be multiple of num_models

        # Dynamics Model
        'dynamics_hidden_sizes': [(512, 512)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [200,],  # UNUSED
        'dynamics_learning_rate': [5e-4],
        'dynamics_batch_size': [256],
        'dynamics_buffer_size': [5000],
        'deterministic': [True],
        'loss_str': ['MSE'],
        'initial_random_samples': [True],

        # Policy
        'policy_hidden_sizes': [(32, 32)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tanh],
        'policy_output_nonlinearity': [None],

        # Meta-Algo
        'rollouts_per_meta_task': [50,],
        'num_inner_grad_steps': [1],
        'inner_lr': [0.001],
        'inner_type': ['log_likelihood'],
        'step_size': [0.05],
        'exploration': [False],
        'sample_from_buffer': [True],  # not implemented

        'scope': [None],
        'exp_tag': ['low_torques_wrist_and_elbow'],  # For changes besides hyperparams
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

