from meta_mb.trainers.bptt_trainer import BPTTTrainer
from meta_mb.policies.bptt_controllers.mpc_controller import MPCController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.dynamics.mlp_dynamics_ensemble_refactor import MLPDynamicsEnsemble
from meta_mb.envs.mb_envs import InvertedPendulumEnv, HalfCheetahEnv, ReacherEnv
from meta_mb.envs.mb_envs.inverted_pendulum import InvertedPendulumSwingUpEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf


EXP_NAME = 'bptt-mb-mpc'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    repr = f"mb-{config['method_str']}-"

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

        policy = MPCController(
            env=env,
            dynamics_model=dynamics_model,
            method_str=config['method_str'],
            num_rollouts=config['num_rollouts'],
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            num_cem_iters=config['num_cem_iters'],
            deterministic_policy=config['deterministic_policy'],
        )

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
            n_parallel=1,
        )

        algo = BPTTTrainer(
            env=env,
            sampler=sampler,
            policy=policy,
            dynamics_model=dynamics_model,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            initial_random_samples=config['initial_random_samples'],
            initial_sinusoid_samples=config['initial_sinusoid_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            sess=sess,
            fit_model=True,
            num_random_iters=1,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'env': [InvertedPendulumSwingUpEnv],
        'horizon': [20,],
        'num_rollouts': [20],
        'method_str': ['rs', 'cem'],

        # Problem
        'seed': [1],
        'normalize': [False],
         'n_itr': [201],
        'discount': [1.0, 0.99],

        # Policy
        'n_candidates': [1000],
        'num_cem_iters': [10],
        'deterministic_policy': [True],

        # Training
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],

        # Dynamics Model
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(512, 512)],
        'dynamic_model_epochs': [15],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
