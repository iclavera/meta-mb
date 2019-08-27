from meta_mb.trainers.bptt_trainer import BPTTTrainer
from meta_mb.policies.bptt_controllers.ipopt_controller import IpoptController
from meta_mb.policies.bptt_controllers.mpc_controller import MPCController
from meta_mb.samplers.ipopt_sampler import IpoptSampler
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.dynamics.mlp_dynamics_ensemble_refactor import MLPDynamicsEnsemble
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mb_envs import *
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.envs.mb_envs.toy import FirstOrderEnv
from meta_mb.utils.utils import ClassEncoder
import json
import joblib
import os
import tensorflow as tf


EXP_NAME = 'dyn-ipopt'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    repr = f"dyn-{config['method_str']}-"
    if config['env'] is HalfCheetahEnv:
        repr += 'hc'
        config['max_path_length'] = 100  # 50
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
        config['max_path_length'] = 30

    if config['use_pretrained_model']:
        repr += '-pretrain'

    print(f"horizon, max_path_length, max_path_length_eval = {config['horizon']}, {config['max_path_length']}, {config['max_path_length_eval']}")

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

        if config['use_pretrained_model']:
            data = joblib.load(config['model_path'])
            dynamics_model = data['dynamics_model']
            assert dynamics_model.obs_space_dims == env.observation_space.shape[0]
            assert dynamics_model.action_space_dims == env.action_space.shape[0]

        else:
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

        sample_processor = ModelSampleProcessor()

        cem_policy = MPCController(
            env=env,
            dynamics_model=dynamics_model,
            method_str='cem',
            num_rollouts=config['cem_num_rollouts'],
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            num_cem_iters=config['num_cem_iters'],
            deterministic_policy=config['cem_deterministic_policy'],
        )

        cem_sampler = Sampler(
            env=env,
            policy=cem_policy,
            num_rollouts=config['cem_num_rollouts'],
            max_path_length=config['max_path_length'],
        )

        policy = IpoptController(
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            horizon=config['horizon'],
            initializer_str=config['initializer_str'],
            method_str=config['method_str'],
            num_rollouts=config['num_rollouts'],
        )

        sampler = IpoptSampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
        )

        algo = BPTTTrainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            sampler=sampler,
            cem_sampler=cem_sampler,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            initial_random_samples=config['initial_random_samples'],
            initial_sinusoid_samples=config['initial_sinusoid_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
            fit_model=config['fit_model'],
            on_policy_freq=config['on_policy_freq'],
            use_pretrained_model=config['use_pretrained_model'],
            num_random_iters=config['num_random_iters'],
        )

        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1],
        'fit_model': [True],

        # Problem
        'env': [HalfCheetahEnv], #[FirstOrderEnv, ReacherEnv, InvertedPendulumEnv,], #[HalfCheetahEnv],
        'use_pretrained_model': [False],
        # HalfCheetah
        'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrain-mb-ppo/hc-100/params.pkl'],
        'n_itr': [50],
        'discount': [1.0,],
        'max_path_length_eval': [20],  # FIXME
        'horizon': [20],
        'method_str': ['collocation'],

        # Policy
        'initializer_str': ['uniform'], #['zeros', 'uniform'],

        # CEM
        'n_candidates': [100],
        'num_cem_iters': [10],
        'cem_deterministic_policy': [True],
        'cem_num_rollouts': [20],

        # Training
        'num_rollouts': [1],
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],
        'num_random_iters': [1],
        'on_policy_freq': [5],

        # Dynamics Model
        'num_models': [1],
        'hidden_nonlinearity_model': ['relu'],
        'dynamic_model_epochs': [15],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'hidden_sizes_model': [(512,)],  # FIXME: need to agree with pretrained model
        'batch_size_model': [64],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],

        #  Other
        'n_parallel': [1],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
