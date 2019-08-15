from meta_mb.trainers.policy_only_trainer import PolicyOnlyTrainer
from meta_mb.policies.dyn_ilqr_controller import DyniLQRController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.dynamics.mlp_dynamics import MLPDynamicsModel
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics import ProbMLPDynamics
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.utils.utils import ClassEncoder
import json
import joblib
import os
import tensorflow as tf


EXP_NAME = 'dyn-ilqr'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
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
        config['max_path_length'] = max_path_length = 100
    elif config['env'] is InvertedPendulumEnv:
        repr += 'ip'
        config['max_path_length'] = max_path_length = 100
    elif config['env'] is InvertedPendulumSwingUpEnv:
        repr += 'ipup'
        config['max_path_length'] = max_path_length = 100
    elif config['env'] is ReacherEnv:
        repr += 'reacher'
        config['max_path_length'] = max_path_length = 50

    if not config['fit_model']:
        config['n_itr'] = 1

    if config.get('model_path', None) is not None:
        # config['fit_model'] = False
        # config['max_path_length'] = 30
        config['initial_random_samples'] = False
        config['initial_sinusoid_samples'] = False

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

        if config.get('model_path', None) is not None:
            data = joblib.load(config['model_path'])
            dynamics_model = data['dynamics_model']
            assert dynamics_model.obs_space_dims == env.observation_space.shape[0]
            assert dynamics_model.action_space_dims == env.action_space.shape[0]

        else:
            if config['dyn_str'] == 'model':
                dynamics_model = MLPDynamicsModel(
                    name="dyn_model",
                    env=env,
                    learning_rate=config['learning_rate'],
                    hidden_sizes=config['hidden_sizes_model'],
                    weight_normalization=config['weight_normalization_model'],
                    valid_split_ratio=config['valid_split_ratio'],
                    rolling_average_persitency=config['rolling_average_persitency'],
                    hidden_nonlinearity=config['hidden_nonlinearity_model'],
                    batch_size=config['batch_size_model'],
                )

            elif config['dyn_str'] == 'ensemble':
                dynamics_model = MLPDynamicsEnsemble(
                    name="dyn_model",
                    env=env,
                    learning_rate=config['learning_rate'],
                    hidden_sizes=config['hidden_sizes_model'],
                    weight_normalization=config['weight_normalization_model'],
                    num_models=config['num_models'],
                    valid_split_ratio=config['valid_split_ratio'],
                    rolling_average_persitency=config['rolling_average_persitency'],
                    hidden_nonlinearity=config['hidden_nonlinearity_model'],
                    batch_size=config['batch_size_model'],
                )
            elif config['dyn_str'] == 'prob_model':
                dynamics_model = ProbMLPDynamics(
                    name="dyn_model",
                    env=env,
                    learning_rate=config['learning_rate'],
                    hidden_sizes=config['hidden_sizes_model'],
                    weight_normalization=config['weight_normalization_model'],
                    valid_split_ratio=config['valid_split_ratio'],
                    rolling_average_persitency=config['rolling_average_persitency'],
                    hidden_nonlinearity=config['hidden_nonlinearity_model'],
                    batch_size=config['batch_size_model'],
                )
            elif config['dyn_str'] == 'prob_ensemble':
                dynamics_model = ProbMLPDynamicsEnsemble(
                    name="dyn_model",
                    env=env,
                    learning_rate=config['learning_rate'],
                    hidden_sizes=config['hidden_sizes_model'],
                    weight_normalization=config['weight_normalization_model'],
                    num_models=config['num_models'],
                    valid_split_ratio=config['valid_split_ratio'],
                    rolling_average_persitency=config['rolling_average_persitency'],
                    hidden_nonlinearity=config['hidden_nonlinearity_model'],
                    batch_size=config['batch_size_model'],
                )
            else:
                raise NotImplementedError

        sample_processor = ModelSampleProcessor()

        policy = DyniLQRController(
            name="policy",
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            max_path_length= max_path_length, # config['max_path_length'],
            n_parallel=config['n_parallel'],
            initializer_str=config['initializer_str'],
            num_ddp_iters=config['num_ddp_iters'],
            num_rollouts=config['num_rollouts'],
            alpha=config['alpha'],
            percent_elites=config['percent_elites'],
        )

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=max_path_length, # config['max_path_length'],
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
        'fit_model': [True],
        'plot_freq': [-1],

        # Problem
        'env': [HalfCheetahEnv], #[ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        # HalfCheetah
        # 'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrain-model-me-ppo/2019_07_15_18_34_37_0/params.pkl'],
        # HalfCheetah
        # 'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrained-mb-ppo/hc-100/params.pkl'],
        'normalize': [False],
        'n_itr': [50],  # only matters for opt_policy
        'discount': [1.0,],
        'controller_str': ['dyn'],

        # Policy
        'initializer_str': ['zeros'], #['zeros', 'uniform'],
        'method_str': ['ilqr'], #'ipopt_shooting'], #['opt_policy', 'opt_act'],  # ['opt_policy', 'opt_act', 'cem', 'rs']
        'policy_filter': [False,],

        'deterministic_policy': [True],

        # cem
        'horizon': [10],  # only matters for cem/rs/collocation
        'n_candidates': [1000],
        'num_cem_iters': [50],
        'alpha': [0.15],
        'percent_elites': [0.1],

        # collocation
        'lmbda': [1e0],
        'num_collocation_iters': [500*30],
        'persistency': [0.9],

        # DDP
        'num_ddp_iters': [20],

        # Training
        'num_rollouts': [1],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],

        # Dynamics Model
        'dyn_str': ['model'], #['prob_ensemble', 'ensemble', 'prob_model', 'model'],
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'dynamic_model_epochs': [15],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'hidden_sizes_model': [(512, 512)],
        'batch_size_model': [64],
        'learning_rate': [0.001],

        #  Other
        'n_parallel': [1],
    }

    # assert config['horizon'] == config['max_path_length']

    config_debug = config.copy()
    config_debug['max_path_length'] = [7]
    config_debug['horizon'] = [4]
    config_debug['num_models'] = [3]
    config_debug['num_rollouts'] = [2]
    config_debug['plot_freq'] = [1]
    config_debug['n_parallel'] = 4

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
    # print('================ runnning toy example ================')
    #run_sweep(run_experiment, config_debug, EXP_NAME, INSTANCE_TYPE)
