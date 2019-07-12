import os
import json
import tensorflow as tf
import numpy as np
INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'mbpo-4'
import sys
sys.path.append('/home/vioichigo/meta-mb/tf_mbpo/mbpo/handful-of-trails')

from meta_mb.algos.sac import SAC
from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import set_seed, ClassEncoder
from meta_mb.envs.mb_envs import *
from meta_mb.envs.normalized_env import normalize
from meta_mb.trainers.sac_edit_trainer import Trainer
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
from meta_mb.logger import logger
from meta_mb.value_functions.value_function import ValueFunction
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import ProbMLPDynamicsEnsemble

save_model_dir = 'home/vioichigo/Desktop/meta-mb/Saved_Model/' + EXP_NAME + '/'


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

        Qs = [ValueFunction(name="q_fun_%d" % i,
                            obs_dim=int(np.prod(env.observation_space.shape)),
                            action_dim=int(np.prod(env.action_space.shape))
                            ) for i in range(2)]

        Q_targets = [ValueFunction(name="q_fun_target_%d" % i,
                                   obs_dim=int(np.prod(env.observation_space.shape)),
                                   action_dim=int(np.prod(env.action_space.shape))
                                   ) for i in range(2)]

        policy = GaussianMLPPolicy(
            name="policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            hidden_sizes=kwargs['policy_hidden_sizes'],
            learn_std=kwargs['policy_learn_std'],
            output_nonlinearity=kwargs['policy_output_nonlinearity'],
            squashed=True
        )

        env_sampler = Sampler(
            env=env,
            policy=policy,
            num_rollouts=kwargs['num_rollouts'],
            max_path_length=kwargs['max_path_length'],
            n_parallel=kwargs['n_parallel'],
        )

        env_sample_processor = ModelSampleProcessor(
            baseline=baseline,
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        dynamics_model = ProbMLPDynamicsEnsemble('dynamics-ensemble',
                                                 env=env,
                                                 rolling_average_persitency=kwargs['rolling_average_persitency'],
                                                 num_models=kwargs['num_models'],
                                                 hidden_nonlinearity=kwargs['dyanmics_hidden_nonlinearity'],
                                                 hidden_sizes=kwargs['dynamics_hidden_sizes'],
                                                 learning_rate=kwargs['dynamics_learning_rate'],
                                                 buffer_size=kwargs['dynamics_buffer_size'],
                                                )

        algo = SAC(
            policy=policy,
            discount=kwargs['discount'],
            learning_rate=kwargs['learning_rate'],
            env=env,
            dynamics_model=dynamics_model,
            Qs=Qs,
            Q_targets=Q_targets,
            reward_scale=kwargs['reward_scale'],
            target_entropy=kwargs['target_entropy']
        )

        trainer = Trainer(
            algo=algo,
            env=env,
            env_name=str(kwargs['env']),
            env_sampler=env_sampler,
            env_sample_processor=env_sample_processor,
            dynamics_model=dynamics_model,
            policy=policy,
            n_itr=kwargs['n_itr'],
            num_model_rollouts=kwargs['num_model_rollouts'],
            sess=sess,
            n_initial_exploration_steps=kwargs['n_initial_exploration_steps'],
            env_max_replay_buffer_size=kwargs['env_replay_buffer_max_size'],
            model_max_replay_buffer_size=kwargs['model_replay_buffer_max_size'],
            rollout_length_params=kwargs['rollout_length_params'],
            rollout_batch_size=kwargs['rollout_batch_size'],
            model_train_freq=kwargs['model_train_freq'],
            model_reset_freq=kwargs['model_reset_freq'],
            n_train_repeats=kwargs['n_train_repeats'],
            real_ratio=kwargs['real_ratio'],
            max_model_t=kwargs['max_model_t'],
            epoch_length=kwargs['epoch_length'],
            restore_path=save_model_dir+kwargs['restore_path'],
        )

        trainer.train()
    sess.__exit__()


if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        'seed': [11, 22],
        'baseline': [LinearFeatureBaseline],
        'env': [HalfCheetahEnv, HopperEnv],
        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],
        'num_model_rollouts': [1],

        # Env Sampling
        'num_rollouts': [1],
        'n_parallel': [1],

        # replay_buffer
        'env_replay_buffer_max_size': [1e4, 5e4],
        'model_replay_buffer_max_size': [2e6],
        'rolling_average_persitency': [0.4, 0.9],

        # Problem Conf
        'n_itr': [300],
        'n_train_repeats': [2, 20],
        'max_path_length': [1000],
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
        'reward_scale': [1],

        # Dynamics Model
        'num_models': [5],
        'dynamics_learning_rate': [3e-4],
        'dyanmics_hidden_nonlinearity': ['relu'],
        'dynamics_buffer_size': [5e4],
        'dynamics_hidden_sizes': [(200, 200, 200, 200)],

        'num_actions_per_next_observation': [3],
        'prediction_type': ['mean'],
        'T': [3],
        'n_initial_exploration_steps': [5e3],
        'rollout_length_params': [[20, 150, 1, 1]],
        'model_reset_freq': [1000],
        'model_train_freq': [250],
        'rollout_batch_size': [100e3],
        'real_ratio': [0.05],
        'max_model_t': [1e10],
        'epoch_length': [1000],
        'restore_path': [''],
        'target_entropy': [-3, -0.75],

        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
