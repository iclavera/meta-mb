from meta_mb.trainers.policy_only_trainer import PolicyOnlyTrainer
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.envs.mb_envs import InvertedPendulumEnv, HalfCheetahEnv, HopperEnv, AntEnv, Walker2dEnv
# from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
# from meta_mb.envs.blue.full_blue_env import FullBlueEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf
import joblib


EXP_NAME = 'mb-mpc-vis-hc'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    print(f'=====================================exp_dir = {exp_dir}=====================')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:

        env = config['env']()

        if config.get('model_path', None) is None:
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
            data = joblib.load(config['model_path'])
            dynamics_model = data['dynamics_model']

        policy = MPCController(
            name="policy",
            env=env,
            dynamics_model=dynamics_model,
            discount=config['discount'],
            n_candidates=config['n_candidates'],
            horizon=config['horizon'],
            use_cem=config['use_cem'],
            use_opt=config['use_opt'],
            use_opt_w_policy=config['use_opt_w_policy'],
            num_cem_iters=config['num_cem_iters'],
            num_opt_iters=config['num_opt_iters'],
            opt_learning_rate=config['opt_learning_rate'],
            num_rollouts=config['num_rollouts'],
        )

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
            n_parallel=config['n_parallel'],
        )

        sample_processor = ModelSampleProcessor()

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
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
        'seed': [1],
        # HalfCheetah
        # 'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrain-model-me-ppo/2019_07_15_18_34_37_0/params.pkl'],
        # InvertedPendulum
        # 'model_path': ['/home/yunzhi/mb/meta-mb/data/pretrain-model-me-ppo-IP/2019_07_16_12_49_53_0/params.pkl'],
        'fit_model': [True],

        # Problem
        'env': [HalfCheetahEnv], #InvertedPendulumEnv],
        'max_path_length': [100],
        'normalize': [False],
         'n_itr': [50],
        'discount': [1.],

        # Policy
        'n_candidates': [1000], # K
        'horizon': [20], # Tau
        'use_cem': [False],
        'num_cem_iters': [5],
        'use_opt': [False],
        'use_opt_w_policy': [True],
        'num_opt_iters': [10], #20, 40,],
        'opt_learning_rate': [1e-3, 1e-2],

        # Training
        'num_rollouts': [20],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],

        # Dynamics Model
        'recurrent': [False],
        'num_models': [1],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(500, 500)],
        'dynamic_model_epochs': [15],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'cell_type': ['lstm'],

        #  Other
        'n_parallel': [1],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
