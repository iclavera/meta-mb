import gym
import json
import os
import tensorflow as tf

from experiment_utils.run_sweep import run_sweep
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
# from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble import MLPDynamicsEnsemble as ProbMLPDynamicsEnsemble
from meta_mb.dynamics.rnn_dynamics_ensemble import RNNDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.reward_model.mlp_reward_ensemble import MLPRewardEnsemble
from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.trainers.mb_trainer import Trainer
from meta_mb.utils.utils import ClassEncoder
from meta_mb.unsupervised_learning.cpc.cpc import CPCEncoder
from meta_mb.unsupervised_learning.vae import VAE

# envs
from meta_mb.envs.img_wrapper_env import ImgWrapperEnv
from meta_mb.envs.normalized_env import NormalizedEnv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

EXP_NAME = 'PTMASS'

INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    # exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    #exp_dir = '%s-ptsize=%d-codesize=%d%s' % (config['encoder'], config['ptsize'], config['latent_dim'], config['suffix'])
    exp_dir = '%s-distractor%s' % (config['encoder'], config['suffix'])
    model_path = os.path.join('meta_mb/unsupervised_learning/cpc/data', exp_dir, 'encoder.h5' if config['encoder'] == 'cpc' else 'vae')
    exp_dir = os.path.join(os.getcwd(), 'data', exp_dir)
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:
        if 'distractor' in exp_dir:
            from meta_mb.envs.mujoco.point_pos_distractor import PointEnv
        else:
            from meta_mb.envs.mujoco.point_pos import PointEnv

        if config['use_image']:
            if config['encoder'] == 'cpc':
                encoder = CPCEncoder(path=model_path)
            elif config['encoder'] == 'vae':
                encoder = VAE(latent_dim=config['latent_dim'], decoder_bernoulli=True, model_path=model_path)
            env = ImgWrapperEnv(NormalizedEnv(PointEnv(ptsize=config['ptsize'], random_reset=False)), time_steps=1, vae=encoder, latent_dim=config['latent_dim'])
        else:
            env = NormalizedEnv(PointEnv(ptsize=config['ptsize'], random_reset=False))


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


            reward_model = MLPRewardEnsemble(
                name="rew_model",
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

            policy = RNNMPCController(
                name="policy",
                env=env,
                dynamics_model=dynamics_model,
                discount=config['discount'],
                n_candidates=config['n_candidates'],
                horizon=config['horizon'],
                use_cem=config['use_cem'],
                num_cem_iters=config['num_cem_iters'],
                use_reward_model=config['use_reward_model'],
                reward_model=reward_model if config['use_reward_model'] else None
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

            reward_model = MLPRewardEnsemble(
                name="rew_model",
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
                num_cem_iters=config['num_cem_iters'],
                use_reward_model=config['use_reward_model'],
                reward_model= reward_model if config['use_reward_model'] else None
            )

        sampler = BaseSampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=config['max_path_length'],
            #n_parallel=config['n_parallel'],
        )

        sample_processor = ModelSampleProcessor()

        algo = Trainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            reward_model=reward_model if config['use_reward_model'] else None,
            sampler=sampler,
            dynamics_sample_processor=sample_processor,
            n_itr=config['n_itr'],
            initial_random_samples=config['initial_random_samples'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            reward_model_max_epochs=config['reward_model_epochs'],
            sess=sess,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    config = {
                'seed': [5],

                # Problem
                'ptsize': [2],  # 'HalfCheetahEnv'
                'max_path_length': [32],
                'normalize': [False],
                 'n_itr': [50],
                'discount': [1.],

                # Policy
                'n_candidates': [1000], # K
                'horizon': [10], # Tau
                'use_cem': [False],
                'num_cem_iters': [5],

                # Training
                'num_rollouts': [5],
                'learning_rate': [0.001],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.99],
                'initial_random_samples': [False],

                # Dynamics Model
                'recurrent': [False],
                'num_models': [2],
                'hidden_nonlinearity_model': ['relu'],
                'hidden_sizes_model': [(500, 500,)],
                'dynamic_model_epochs': [15],
                'backprop_steps': [100],
                'weight_normalization_model': [False],  # FIXME: Doesn't work
                'batch_size_model': [64],
                'cell_type': ['lstm'],
                'use_reward_model': [True],

                # Reward Model
                'reward_model_epochs': [15],

                #  Other
                'n_parallel': [1],
                'suffix': [1, 2],


                # representation learning

                'use_image': [True],
                # 'model_path': ['meta_mb/unsupervised_learning/cpc/data/neg-15rand_reset/encoder.h5'],
                'encoder': ['vae', 'cpc'],
                'latent_dim': [32],

    }


    run_sweep(run_experiment, config, EXP_NAME, INSTANCE_TYPE)
