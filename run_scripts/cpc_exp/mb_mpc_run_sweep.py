import gym
import json
import os
import tensorflow as tf

from experiment_utils.run_sweep import run_sweep
from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
from meta_mb.dynamics.rnn_dynamics_ensemble import RNNDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.policies.mpc_controller import MPCController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.reward_model.mlp_reward_ensemble import MLPRewardEnsemble
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.trainers.mb_trainer import Trainer
from meta_mb.utils.utils import ClassEncoder
from meta_mb.unsupervised_learning.cpc.cpc import CPCEncoder, CPCContextNet
from meta_mb.unsupervised_learning.vae import VAE

# envs
from meta_mb.envs.envs_util import make_env
from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
from meta_mb.envs.img_wrapper_env import ImgWrapperEnv
from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv
from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.obs_stack_env import ObsStackEnv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

EXP_NAME = 'IP/rnn_cpc2'

INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    exp_name = ''
    if isinstance(config['env'], str):
        folder = '%s-neg%d-hist%d-fut%d-code%d%s' % (config['env'], config['negative'], config['history'], config['future'],
                                                           config['latent_dim'], config['run_suffix'])
        model_path = os.path.join('meta_mb/unsupervised_learning/cpc/data', folder,
                                  'vae' if config['encoder'] == 'vae' else
                                  'encoder.h5' if not config['use_context_net'] else
                                  'context.h5')
        raw_env, max_path_length = make_env(config['env'])

    else:
        raw_env = config['env']()
        max_path_length = config['max_path_length']
        if 'model_path' in config:
            # exp_name = config['model_path'] + '_rnnmodel'
            model_path = os.path.join('meta_mb/unsupervised_learning/cpc/data', config['model_path'],
                                      'vae' if config['encoder'] == 'vae' else
                                      'encoder.h5' if not config['use_context_net'] else
                                      'context.h5')
        else:
            exp_name = 'cp-neg%d-hist%d-fut%d-code%d%s' % (config['negative'], config['history'], config['future'],
                                                           config['latent_dim'], config['run_suffix'])
            model_path = os.path.join('meta_mb/unsupervised_learning/cpc/data', exp_name,
                                      'vae' if config['encoder'] == 'vae' else
                                      'encoder.h5' if not config['use_context_net'] else
                                      'context.h5')
            # exp_name =  'len=125-cem=%b' % config['use_cem'] + exp_name
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + (exp_name or config.get('exp_name', ''))
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    config_sess = tf.ConfigProto()
    config_sess.gpu_options.allow_growth = True
    config_sess.gpu_options.per_process_gpu_memory_fraction = config.get('gpu_frac', 0.95)
    sess = tf.Session(config=config_sess)
    with sess.as_default() as sess:
        if config['use_image']:
            if config['encoder'] == 'cpc':
                if config['use_context_net']:
                    encoder = CPCContextNet(path=model_path)
                    env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=config['history'], vae=encoder,
                                        latent_dim=config['latent_dim'], time_major=True)
                else:
                    encoder = CPCEncoder(path=model_path)
                    env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=1, vae=encoder,
                                        latent_dim=config['latent_dim'])
            elif config['encoder'] == 'vae':
                encoder = VAE(latent_dim=config['latent_dim'], decoder_bernoulli=True, model_path=model_path)
                env = ImgWrapperEnv(NormalizedEnv(config['env']()), time_steps=1, vae=encoder, latent_dim=config['latent_dim'])
        else:
            env = NormalizedEnv(raw_env)

        if config['obs_stack'] > 1:
            env = ObsStackEnv(env, time_steps=config['obs_stack'])

        if config['recurrent']:
            dynamics_model = RNNDynamicsEnsemble(
                name="dyn_model",
                env=env,
                hidden_sizes=config['hidden_sizes_model'],
                learning_rate=config['learning_rate'],
                backprop_steps=config['backprop_steps'],
                cell_type=config['cell_type'],
                num_models=config['num_models'],
                rolling_average_persitency=config['rolling_average_persitency'],
                valid_split_ratio=config['valid_split_ratio'],
                batch_size=config['batch_size_model'],
                normalize_input=config['normalize'],
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
                reward_model= reward_model if config['use_reward_model'] else None,
                use_graph=config['use_graph']
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
            max_path_length=max_path_length,
        )

        sample_processor = ModelSampleProcessor(recurrent=config['recurrent'])

        algo = Trainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            reward_model=reward_model if config['use_reward_model'] else None,
            sampler=sampler,
            dynamics_sample_processor=sample_processor,
            reward_sample_processor=ModelSampleProcessor(recurrent=False),
            n_itr=config['n_itr'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            reward_model_max_epochs=config['reward_model_epochs'],
            sess=sess,
        )
        algo.train()


if __name__ == '__main__':
    # -------------------- Define Variants -----------------------------------

    def make_pt_env():
        return PointEnv(random_reset=False)

    def make_dm_cpb_env():
        from dm_control import suite
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole', 'balance')),
                                                 keys=['position', 'velocity']), amount=8)
        return raw_env


    config_ip = {
                'seed': [1, 2],

                # Problem
                'env': [InvertedPendulumEnv],  # 'HalfCheetahEnv'
                'max_path_length': [32],
                'normalize': [True],
                 'n_itr': [50],
                'discount': [1.],
                'obs_stack': [1],

                # Policy
                'n_candidates': [1000],  # K
                'horizon': [10],  # Tau
                'use_cem': [False],
                'num_cem_iters': [5],
                'use_graph': [True],

                # Training
                'num_rollouts': [20],
                'learning_rate': [0.001],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.4],

                # Dynamics Model
                'recurrent': [False],
                'num_models': [5],
                'hidden_nonlinearity_model': ['relu'],
                'hidden_sizes_model': [(500,)],
                'dynamic_model_epochs': [50],
                'backprop_steps': [100],
                'weight_normalization_model': [False],  # FIXME: Doesn't work
                'batch_size_model': [64],
                'cell_type': ['lstm'],
                'use_reward_model': [True],

                # Reward Model
                'reward_model_epochs': [15],

                #  Other
                'n_parallel': [1],

                # representation learning

                'use_image': [True],
                # 'model_path': ['ip-neg-15'],
                'encoder': ['cpc'],
                'latent_dim': [32],
                'negative': [10],
                'history': [3],
                'future': [1],
                'use_context_net': [True]

    }

    config_ip_rnn = {
                'seed': [1],
                'run_suffix': [''],

                # Problem
                'env': [InvertedPendulumEnv],  # 'HalfCheetahEnv'
                'max_path_length': [32],
                'normalize': [True],
                 'n_itr': [50],
                'discount': [1.],
                'obs_stack': [1],

                # Policy
                'n_candidates': [1000],  # K
                'horizon': [10],  # Tau
                'use_cem': [False],
                'num_cem_iters': [5],
                'use_graph': [True],

                # Training
                'num_rollouts': [20],
                'learning_rate': [0.01, 0.001, 0.0001],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.8, 0.95],

                # Dynamics Model
                'recurrent': [True],
                'num_models': [5],
                'hidden_nonlinearity_model': ['relu'],
                'hidden_sizes_model': [(500, )],
                'dynamic_model_epochs': [30],
                'backprop_steps': [100],
                'weight_normalization_model': [False],  # FIXME: Doesn't work
                'batch_size_model': [10],
                'cell_type': ['lstm'],
                'use_reward_model': [True],

                # Reward Model
                'reward_model_epochs': [15],

                #  Other
                'n_parallel': [1],

                # representation learning

                'use_image': [True],
                'model_path': ['ip-neg10-hist3-fut1-code82'],
                'encoder': ['cpc'],
                'latent_dim': [8,],
                'negative': [10],
                'history': [3],
                'future': [1],
                'use_context_net': [False]

    }

    config_pt_rnn = {
                'seed': [1],
                'run_suffix': [''],

                # Problem
                'env': [make_pt_env],  # 'HalfCheetahEnv'
                'max_path_length': [32],
                'normalize': [True],
                 'n_itr': [50],
                'discount': [1.],
                'obs_stack': [1],

                # Policy
                'n_candidates': [1000],  # K
                'horizon': [10],  # Tau
                'use_cem': [False],
                'num_cem_iters': [5],
                'use_graph': [True],

                # Training
                'num_rollouts': [50],
                'learning_rate': [0.001],
                'valid_split_ratio': [0.1],
                'rolling_average_persitency': [0.4, 0.99],

                # Dynamics Model
                'recurrent': [True],
                'num_models': [5],
                'hidden_nonlinearity_model': ['relu'],
                'hidden_sizes_model': [(500, ), (32, )],
                'dynamic_model_epochs': [50],
                'backprop_steps': [100],
                'weight_normalization_model': [False],  # FIXME: Doesn't work
                'batch_size_model': [20],
                'cell_type': ['lstm'],
                'use_reward_model': [True],

                # Reward Model
                'reward_model_epochs': [15],

                #  Other
                'n_parallel': [1],

                # representation learning

                'use_image': [True],
                'model_path': ['cpc-ptsize=2-codesize=321'],
                'encoder': ['cpc'],
                'latent_dim': [32],
                'negative': [10],
                'history': [3],
                'future': [1],
                'use_context_net': [False]

    }

    config_cp = {
        'seed': [1],
        'run_suffix': ['1', '2'],

        # Problem
        'env': [make_dm_cpb_env],  # 'HalfCheetahEnv'
        'max_path_length': [125],
        'normalize': [True],
        'n_itr': [50],
        'discount': [1.],
        'obs_stack': [3],

        # Policy
        'n_candidates': [1000],  # K
        'horizon': [12],  # Tau
        'use_cem': [True, False],
        'num_cem_iters': [5],
        'use_graph': [True],

        # Training
        'num_rollouts': [5],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.4],

        # Dynamics Model
        'recurrent': [False],
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(500,)],
        'dynamic_model_epochs': [50],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'cell_type': ['lstm'],
        'use_reward_model': [True],

        # Reward Model
        'reward_model_epochs': [15],

        #  Other
        'n_parallel': [1],

        # representation learning

        'use_image': [True],
        # 'model_path': ['ip-neg-15'],
        'encoder': ['cpc'],
        'latent_dim': [32],
        'negative': [10],
        'history': [3],
        'future': [1],
        'use_context_net': [False]

    }

    config_envs = {
        'seed': [1, 2],
        'run_suffix': ['1'],

        # Problem
        'env': ['reacher_easy', 'cheetah_run', 'cartpole_swingup'],
        'normalize': [True],
        'n_itr': [150],
        'discount': [1.],
        'obs_stack': [3],

        # Policy
        'n_candidates': [1000],  # K
        'horizon': [12],  # Tau
        'use_cem': [True],
        'num_cem_iters': [5],
        'use_graph': [True],

        # Training
        'num_rollouts': [5],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.1],
        'rolling_average_persitency': [0.4],

        # Dynamics Model
        'recurrent': [False],
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(500,)],
        'dynamic_model_epochs': [50],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'cell_type': ['lstm'],
        'use_reward_model': [True],

        # Reward Model
        'reward_model_epochs': [15],

        #  Other
        'n_parallel': [1],

        # representation learning

        'use_image': [True],
        # 'model_path': ['ip-neg-15'],
        'encoder': ['cpc'],
        'latent_dim': [32],
        'negative': [10],
        'history': [3],
        'future': [3],
        'use_context_net': [False]

    }

    run_sweep(run_experiment, config_ip_rnn, EXP_NAME, INSTANCE_TYPE)
