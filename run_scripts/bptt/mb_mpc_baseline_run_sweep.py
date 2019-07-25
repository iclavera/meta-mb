from meta_mb.trainers.policy_only_trainer import PolicyOnlyTrainer
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.policies.mpc_delta_controller import MPCDeltaController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.logger import logger
from experiment_utils.run_sweep import run_sweep
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.dynamics.rnn_dynamics_ensemble import RNNDynamicsEnsemble
from meta_mb.envs.mb_envs import InvertedPendulumEnv, HalfCheetahEnv, ReacherEnv
from meta_mb.utils.utils import ClassEncoder
import json
import os
import tensorflow as tf
import joblib


EXP_NAME = 'bptt-mb-mpc'
INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    if config['env'] is HalfCheetahEnv:
        repr = 'hc'
    elif config['env'] is InvertedPendulumEnv:
        repr = 'ip'
    elif config['env'] is ReacherEnv:
        repr = 'reacher'
    repr += '-reg-' + str(config['reg_coef'])
    if config['reg_str'] is not None:
        repr += '-' + config['reg_str']
    repr += '-init-' + config['initializer_str'] + '-' + config['controller_str'] + '-' + config['method_str']
    repr += '-' + config['dyn_pred_str']

    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + repr + config.get('exp_name', '')
    print(f'===================================== exp_dir = {exp_dir} =====================')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:

        env = config['env']()

        if config.get('model_path', None) is None:
            if config['controller_str'] == 'rnn':
                dynamics_model = RNNDynamicsEnsemble(
                    name="dyn_model",
                    env=env,
                    hidden_sizes=(500,),  #config['hidden_sizes_model'],  # FIXME
                    learning_rate=0.01, #config['learning_rate'],
                    backprop_steps=config['backprop_steps'],
                    cell_type=config['cell_type'],
                    num_models=config['num_models'],
                    batch_size=5, #config['batch_size_model'],
                    normalize_input=True,
                )
                sample_processor = ModelSampleProcessor(recurrent=True)
            else:
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
                sample_processor = ModelSampleProcessor()
        else:
            data = joblib.load(config['model_path'])
            dynamics_model = data['dynamics_model']

        if config['controller_str'] == 'delta':
            policy = MPCDeltaController(
                name="policy",
                env=env,
                dynamics_model=dynamics_model,
                discount=config['discount'],
                horizon=config['horizon'],
                use_opt_w_policy=False,
                reg_coef=config['reg_coef'],
                reg_str=config['reg_str'],
                initializer_str=config['initializer_str'],
                num_opt_iters=config['num_opt_iters'],
                opt_learning_rate=config['opt_learning_rate'],
                num_rollouts=config['num_rollouts'],
            )
        elif config['controller_str'] == 'rnn':
            policy = RNNMPCController(
                name='policy',
                env=env,
                dynamics_model=dynamics_model,
                discount=config['discount'],
                n_candidates=config['n_candidates'],
                horizon=config['horizon'],
                method_str=config['method_str'],
                dyn_pred_str=config['dyn_pred_str'],
                reg_coef=config['reg_coef'],
                reg_str=config['reg_str'],
                initializer_str=config['initializer_str'],
                num_cem_iters=config['num_cem_iters'],
                num_opt_iters=config['num_opt_iters'],
                opt_learning_rate=config['opt_learning_rate'],
                num_rollouts=config['num_rollouts'],
            )
        elif config['controller_str'] == 'mpc':
            policy = MPCController(
                name="policy",
                env=env,
                dynamics_model=dynamics_model,
                discount=config['discount'],
                n_candidates=config['n_candidates'],
                horizon=config['horizon'],
                method_str=config['method_str'],
                dyn_pred_str=config['dyn_pred_str'],
                reg_coef=config['reg_coef'],
                reg_str=config['reg_str'],
                initializer_str=config['initializer_str'],
                num_cem_iters=config['num_cem_iters'],
                num_opt_iters=config['num_opt_iters'],
                opt_learning_rate=config['opt_learning_rate'],
                num_rollouts=config['num_rollouts'],
            )
        else:
            raise NotImplementedError

        sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
            n_parallel=config['n_parallel'],
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
        'plot_freq': [1000],

        # Problem
        'env': [ReacherEnv, HalfCheetahEnv],
        'max_path_length': [100],
        'normalize': [False],
         'n_itr': [101],
        'discount': [1.0, 0.99],
        'controller_str': ['mpc'], # ['rnn', 'delta', 'mpc'],

        # Policy
        'initializer_str': ['zeros',], #['uniform', 'zeros'],
        'reg_coef': [0.1], #[1, 0],
        'reg_str': ['uncertainty'], #['uncertainty'],
        'method_str': ['opt_act', 'opt_policy'],  # ['opt_policy', 'opt_act', 'cem', 'rs']
        'dyn_pred_str': ['rand', 'mean'],  # 'mean', 'batches'
        'horizon': [25,], # Tau

        'num_opt_iters': [20,], #20, 40,],
        'opt_learning_rate': [1e-4, 1e-3,], #1e-2],
        'clip_norm': [-1], #1e2, 1e1, 1e6],

        'n_candidates': [1000], # K
        'num_cem_iters': [5],

        # Training
        'num_rollouts': [20],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.99],
        'initial_random_samples': [True],
        'initial_sinusoid_samples': [False],

        # Dynamics Model
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(512,)],  # (500, 500)
        'dynamic_model_epochs': [15],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'cell_type': ['lstm'],  # ['lstm', 'gru', 'rnn']

        #  Other
        'n_parallel': [1],
    }

    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
