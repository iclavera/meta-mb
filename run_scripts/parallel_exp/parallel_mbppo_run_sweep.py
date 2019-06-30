import os
import json
import pickle
import numpy as np
from tensorflow import tanh, ConfigProto
from multiprocessing import Process, Pipe
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.mb_envs import HalfCheetahEnv, Walker2dEnv, AntEnv, HopperEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.parallel_metrpo_trainer import ParallelTrainer
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.logger import logger

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'local-g1e1-revert-20-parallel-mbppo-partial-buffer'


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
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
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

    if kwargs['env'] == 'Ant':
        env = normalize(AntEnv())
        simulation_sleep = 0.05 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    elif kwargs['env'] == 'HalfCheetah':
        env = normalize(HalfCheetahEnv())
        simulation_sleep = 0.05 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    elif kwargs['env'] == 'Hopper':
        env = normalize(HopperEnv())
        simulation_sleep = 0.008 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    elif kwargs['env'] == 'Walker2d':
        env = normalize(Walker2dEnv())
        simulation_sleep = 0.008 * kwargs['num_rollouts'] * kwargs['max_path_length'] * kwargs['simulation_sleep_frac']
    else:
        raise NotImplementedError

    policy = GaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
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
            'num_rollouts': kwargs['num_rollouts'],
            'max_path_length': kwargs['max_path_length'],
            'n_parallel': kwargs['n_parallel'],
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
            'num_rollouts': kwargs['imagined_num_rollouts'],
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
            'learning_rate': kwargs['learning_rate'],
            'clip_eps': kwargs['clip_eps'],
            'max_epochs': kwargs['num_ppo_steps'],
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
        log_real_performance=kwargs['log_real_performance'],
        steps_per_iter=kwargs['steps_per_iter'],
        flags_need_query=kwargs['flags_need_query'],
        config=config,
        simulation_sleep=simulation_sleep,
    )

    trainer.train()


if __name__ == '__main__':

    sweep_params = {

        'flags_need_query': [
            [False, False, False],
        ],
        'rolling_average_persitency': [
            0.99,
        ],

        'seed': [1, 2,],

        'algo': ['meppo'],
        'baseline': [LinearFeatureBaseline],
        'env': ['HalfCheetah', 'Ant', 'Walker2d', 'Hopper'],
        'simulation_sleep_frac': [1, 0.5, 0.1],

        # Problem Conf
        'n_itr': [401],
        'max_path_length': [200],
        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],
        'log_real_performance': [True],
        'steps_per_iter': [1], # No outer loop in effect

        # Real Env Sampling
        'num_rollouts': [20],
        'n_parallel': [1],

        # Dynamics Model
        'num_models': [5],
        'dynamics_hidden_sizes': [(512, 512, 512)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_max_epochs': [50],  # UNUSED
        'dynamics_learning_rate': [1e-3],
        'dynamics_batch_size': [256,],
        'dynamics_buffer_size': [25000],
        'deterministic': [False],
        'loss_str': ['MSE'],

        # Policy
        'policy_hidden_sizes': [(64, 64)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': [tanh],
        'policy_output_nonlinearity': [None],

        # Algo
        'clip_eps': [0.3],
        'learning_rate': [1e-3],
        'num_ppo_steps': [5],
        'imagined_num_rollouts': [50,],
        'scope': [None],
        'exp_tag': ['timing-parallel-mbppo'],  # For changes besides hyperparams

    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)

