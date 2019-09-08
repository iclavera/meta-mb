import os
import json
import tensorflow as tf
import numpy as np
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = "early-stopping-test"

from pdb import set_trace as st
from meta_mb.algos.sac_edit import SAC_MB
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.mb_envs import *
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.sac_edit_trainer import Trainer
from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.logger import logger
from meta_mb.value_functions.value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline

from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble_try import ProbMLPDynamicsEnsembleTry
# from meta_mb.dynamics.done_predictor import DonePredictor
# import os.path as osp




def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)
    sess = tf.Session(config=config)

    with sess.as_default() as sess:

        # Instantiate classes
        set_seed(kwargs['seed'])

        baseline = kwargs['baseline']()

        env = normalize(kwargs['env']())
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))


        Qs = [ValueFunction(name="q_fun_%d" % i,
                            obs_dim=obs_dim,
                            action_dim=action_dim,
                            hidden_nonlinearity=kwargs['vfun_hidden_nonlineariy'],) for i in range(2)]

        Q_targets = [ValueFunction(name="q_fun_target_%d" % i,
                                   obs_dim=obs_dim,
                                   action_dim=action_dim,
                                   hidden_nonlinearity=kwargs['vfun_hidden_nonlineariy'],) for i in range(2)]

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
            hidden_nonlinearity=kwargs['policy_hidden_nonlinearity'],
            squashed=True
        )
        if kwargs['policy_hidden_sizes'] == ():
            ground_truth = True
        else:
            ground_truth = False

        env_sampler = BaseSampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],

        )

        env_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        assert kwargs['model_type'] in [0, 3]
        if kwargs['model_type'] == 0:
            dynamics_model = ProbMLPDynamicsEnsemble('dynamics-ensemble',
                                                    env=env,
                                                    num_models=kwargs['num_models'],
                                                    hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                    hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                    output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                                    batch_size=kwargs['dynamics_batch_size'],
                                                    learning_rate=kwargs['model_learning_rate'],
                                                    buffer_size=kwargs['dynamics_buffer_size'],
    												rolling_average_persitency=kwargs['rolling_average_persitency'],
                                                    restore=kwargs['restore'],
                                                    )
            algo = SAC_MB(
                policy=policy,
                discount=kwargs['discount'],
                learning_rate=kwargs['learning_rate'],
                target_entropy=kwargs['target_entropy'],
                env=env,
                dynamics_model=dynamics_model,
                obs_dim = obs_dim,
                action_dim = action_dim,
                Qs=Qs,
                Q_targets=Q_targets,
                reward_scale=kwargs['reward_scale'],
                num_actions_per_next_observation=kwargs['num_actions_per_next_observation'],
                prediction_type=kwargs['prediction_type'],
                T=kwargs['T'],
    			q_function_type=kwargs['q_function_type'],
    			q_target_type=kwargs['q_target_type'],
    			H=kwargs['H'],
    			model_used_ratio=kwargs['model_used_ratio'],
    			experiment_name=EXP_NAME,
    			exp_dir=exp_dir,
                target_update_interval=kwargs['n_train_repeats'],
                dynamics_type=kwargs['model_type'],
                restore=kwargs['restore'],
                ground_truth=ground_truth,
            )
        elif kwargs['model_type'] == 3:
            dynamics_model = ProbMLPDynamicsEnsembleTry('dynamics-ensemble',
                                                    env=env,
                                                    num_models=kwargs['num_models'],
                                                    hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                    hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                    output_nonlinearity=kwargs['dyanmics_output_nonlinearity'],
                                                    batch_size=kwargs['dynamics_batch_size'],
                                                    learning_rate=kwargs['model_learning_rate'],
                                                    buffer_size=kwargs['dynamics_buffer_size'],
    												rolling_average_persitency=kwargs['rolling_average_persitency'],
                                                    q_loss_importance=kwargs['q_loss_importance'],
                                                    Qs=Qs,
                                                    Q_targets=Q_targets,
                                                    policy=policy,
                                                    T=kwargs['T'],
                                                    reward_scale=kwargs['reward_scale'],
                                                    discount=kwargs['discount'],
                                                    normalize_input=kwargs['normalize_input'],
                                                    restore=kwargs['restore'],
                                                    type=3,
                                                    )
            algo = SAC_MB(
                policy=policy,
                discount=kwargs['discount'],
                learning_rate=kwargs['learning_rate'],
                target_entropy=kwargs['target_entropy'],
                env=env,
                dynamics_model=dynamics_model,
                obs_dim = obs_dim,
                action_dim = action_dim,
                reward_scale=kwargs['reward_scale'],
                num_actions_per_next_observation=kwargs['num_actions_per_next_observation'],
                prediction_type=kwargs['prediction_type'],
                T=kwargs['T'],
    			q_function_type=kwargs['q_function_type'],
    			q_target_type=kwargs['q_target_type'],
    			H=kwargs['H'],
    			model_used_ratio=kwargs['model_used_ratio'],
    			experiment_name=EXP_NAME,
    			exp_dir=exp_dir,
                target_update_interval=kwargs['n_train_repeats'],
                dynamics_type=kwargs['model_type'],
                ground_truth=ground_truth,
                start_itr=kwargs['start_itr'],
                actor_H=kwargs['actor_H'],
                restore=kwargs['restore'],
            )



        trainer = Trainer(
            algo=algo,
            env=env,
            env_sampler=env_sampler,
            env_sample_processor=env_sample_processor,
            dynamics_model=dynamics_model,
            policy=policy,
            n_itr=kwargs['n_itr'],
            sess=sess,
            n_initial_exploration_steps=kwargs['n_initial_exploration_steps'],
            env_max_replay_buffer_size=kwargs['env_replay_buffer_max_size'],
            model_max_replay_buffer_size=kwargs['model_replay_buffer_max_size'],
            rollout_length_params=kwargs['rollout_length_params'],
            rollout_batch_size=kwargs['rollout_batch_size'],
            model_train_freq=kwargs['model_train_freq'],
            n_train_repeats=kwargs['n_train_repeats'],
            real_ratio=kwargs['real_ratio'],
			dynamics_model_max_epochs=kwargs['dynamics_model_max_epochs'],
			sampler_batch_size=kwargs['sampler_batch_size'],
            dynamics_type=kwargs['model_type'],
            T=kwargs['T'],
            ground_truth=ground_truth,
            restore=kwargs['restore'],
        )

        trainer.train()
    sess.__exit__()


if __name__ == '__main__':
    sweep_params = {
        'seed': [90,29],
        'baseline': [LinearFeatureBaseline],
        'env': [Walker2dEnv],
        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],

        # Env Sampling
        'num_rollouts': [1],
        'model_type': [0],

        # replay_buffer
		'n_initial_exploration_steps': [5e3],
        'env_replay_buffer_max_size': [1e6],
        'model_replay_buffer_max_size': [2e6],
		'n_itr': [1000],
        'n_train_repeats': [8],
        'max_path_length': [1001],
		'rollout_length_params': [[20, 100, 1, 1]],
        'model_train_freq': [250],
		'rollout_batch_size': [100e3],
		'dynamics_model_max_epochs': [200],
		'rolling_average_persitency':[0.9, 0.4],
		'q_function_type':[0],
		'q_target_type': [0],
		'num_actions_per_next_observation':[5],
        'H': [2],
        'T': [2],
        'actor_H': [2, 3, 4],
		'reward_scale': [1],
		'target_entropy': [1],
		'num_models': [4],
		'model_used_ratio': [1],
		'dynamics_buffer_size': [1e4],
        'q_loss_importance': [1],
        'method': [-3],
        'restore': [False],

        'policy_hidden_nonlinearity': ['tanh'],

        # Value Function
        'vfun_hidden_nonlineariy': ['tanh'],
        'normalize_input': [True],


        # Problem Conf
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
		'prediction_type':['rand'],

        # Dynamics Model
		'sampler_batch_size': [256],
        'real_ratio': [0.05],

        'model_learning_rate': [1e-3],
        'dynamics_hidden_sizes': [(200, 200, 200, 200)],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dyanmics_output_nonlinearity': [None],
        'dynamics_batch_size': [256],
        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
