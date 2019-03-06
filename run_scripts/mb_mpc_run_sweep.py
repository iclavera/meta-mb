from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.rnn_dynamics_ensemble import RNNDynamicsEnsemble
from meta_mb.trainers.mb_trainer import Trainer
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mbmpo_samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.logger import logger
from meta_mb.envs.normalized_env import normalize
from experiment_utils.run_sweep import run_sweep
from meta_mb.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from meta_mb.envs.blue.blue_env import BlueReacherEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os

EXP_NAME = 'mb-mpc'

INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(config['env']())


    if config['recurrent']:
        dynamics_model = RNNDynamicsEnsemble(
            name="dyn_model",
            env=env,
            hidden_sizes=config['hidden_sizes_model'],
            learning_rate=config['learning_rate'],
            backprop_steps=config['backprop_steps'],
            cell_type=config['cell_type'],
            num_models=config['num_models'],
            batch_size=config['batch_size_model'],
            normalize_input=True,
        )

        policy = RNNMPCController(
            name="policy",
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            use_cem=config['use_cem'],
            num_cem_iters=config['num_cem_iters'],
            use_reward_model=config['use_reward_model']
        )

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

        policy = MPCController(
            name="policy",
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            use_cem=config['use_cem'],
            num_cem_iters=config['num_cem_iters']
        )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
    )

    sample_processor = ModelSampleProcessor()

    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        dynamics_sample_processor=sample_processor,
        n_itr=config['n_itr'],
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
    )
    algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
                'seed': [11, 22],

                # Problem
                'env': [BlueReacherEnv],  # 'HalfCheetahEnv'
                'max_path_length': [50],
                'normalize': [False],
                 'n_itr': [30],
                'discount': [0.99],

                # Policy
                'n_candidates': [256],
                'horizon': [5],
                'use_cem': [False],
                'num_cem_iters': [4],

                # Training
                'num_rollouts': [20],
                'learning_rate': [0.001],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.99],
                'initial_random_samples': [True],

                # Dynamics Model
                'recurrent': [False],
                'num_models': [5],
                'hidden_nonlinearity_model': ['swish'],
                'hidden_sizes_model': [(256, 256)],
                'dynamic_model_epochs': [200],
                'backprop_steps': [100],
                'weight_normalization_model': [False],  # FIXME: Doesn't work
                'batch_size_model': [256],
                'cell_type': ['lstm'],

                #  Other
                'n_parallel': [5],

    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
