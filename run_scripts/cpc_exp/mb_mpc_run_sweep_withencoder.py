import gym
import json
import os
import keras
import tensorflow as tf
import keras.backend as K

from experiment_utils.run_sweep import run_sweep
from meta_mb.dynamics.rnn_dynamics_ensemble import RNNDynamicsEnsemble
from meta_mb.dynamics.mlp_dynamics_ensemble_withencoder import MLPDynamicsEnsemble
from meta_mb.dynamics.probabilistic_mlp_dynamics_ensemble_withencoder import ProbMLPDynamicsEnsemble
from meta_mb.logger import logger
from meta_mb.policies.mpc_controller_withencoder import MPCController
from meta_mb.policies.rnn_mpc_controller import RNNMPCController
from meta_mb.replay_buffers.image_embedding_buffer import ImageEmbeddingBuffer
from meta_mb.reward_model.mlp_reward_ensemble_withencoder import MLPRewardEnsemble
from meta_mb.samplers.sampler import Sampler
from meta_mb.samplers.base import BaseSampler
from meta_mb.samplers.mb_sample_processor import ModelSampleProcessor
from meta_mb.trainers.mb_trainer_withencoder import Trainer
from meta_mb.utils.utils import ClassEncoder
from meta_mb.unsupervised_learning.cpc.cpc import CPCContextNet
from meta_mb.unsupervised_learning.vae import VAE

# envs
from meta_mb.envs.envs_util import make_env
from meta_mb.envs.img_wrapper_env import ImgWrapperEnv
from meta_mb.envs.normalized_env import NormalizedEnv
from meta_mb.envs.obs_stack_env import ObsStackEnv

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

EXP_NAME = 'cheetah'

INSTANCE_TYPE = 'c4.2xlarge'


def run_experiment(**config):
    exp_name = ''

    folder = '%s-neg%d-hist%d-fut%d-code%d-withaction%r-finetune-%s' % (config['env'], config['negative'], config['history'], \
                                                               config['future'], config['latent_dim'], \
                                                               config['include_action'], config['run_suffix'])
    # folder = '%s-neg%d-hist%d-fut%d-code%d%s' % (config['env'], config['negative'], config['history'], config['future'],
    #                                                    config['latent_dim'], config['run_suffix'])
    model_path = os.path.join('meta_mb/unsupervised_learning/cpc/data', EXP_NAME, folder,
                              'vae' if config['encoder'] == 'vae' else
                              'encoder.h5' if not config['use_context_net'] else
                              'context.h5')
    raw_env, max_path_length = make_env(config['env'], render_size=config['img_shape'][:2])


    if config['use_image']:
        from meta_mb.unsupervised_learning.cpc.cpc import CPC

        cpc_model = CPC(config['img_shape'], raw_env.action_space.shape[0], config['include_action'],
                        config['history'], config['future'], config['negative'], code_size=config['latent_dim'],
                        learning_rate=config['cpc_initial_lr'], encoder_arch='default',
                        context_network='stack', context_size=32, predict_action=config['predict_action'],
                        contrastive=config['contrastive'], grad_penalty=config['grad_penalty'], lambd=config['cpc_lambd'])
    else:
        cpc_model = None

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
                    encoder = CPCContextNet(None, model=cpc_model.get_layer('context_network'))
                    env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=config['history'], encoder=encoder,
                                        latent_dim=config['latent_dim'], time_major=True, img_size=config['img_shape'])
                else:
                    encoder = cpc_model.encoder
                    if config['env_produce_img']:
                        env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=1, img_size=config['img_shape'])
                    else:
                        env = ImgWrapperEnv(NormalizedEnv(raw_env), time_steps=1, img_size=config['img_shape'],
                                            latent_dim=config['latent_dim'], encoder=cpc_model)
                        env = ObsStackEnv(env, time_steps=config['obs_stack'])
            elif config['encoder'] == 'vae':
                encoder = VAE(latent_dim=config['latent_dim'], decoder_bernoulli=True, model_path=model_path)
                env = ImgWrapperEnv(NormalizedEnv(config['env']()), time_steps=1, encoder=encoder,
                                    latent_dim=config['latent_dim'], img_size=config['img_shape'])
        else:
            env = NormalizedEnv(raw_env)



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
            if config['env_produce_img']:
                buffer = ImageEmbeddingBuffer(config['batch_size_model'], env, encoder, config['input_is_img'],
                                              config['latent_dim'], config['obs_stack'], config['num_models'],
                                              config['valid_split_ratio'], normalize_input=config['normalize'],
                                              buffer_size=12500 // max_path_length)
                if config['prob_dyn']:
                    DYN_CLASS = ProbMLPDynamicsEnsemble
                else:
                    DYN_CLASS = MLPDynamicsEnsemble
                dynamics_model = DYN_CLASS(
                    name="dyn_model",
                    env=env,
                    num_stack=config['obs_stack'],
                    encoder=encoder,
                    latent_dim=config['latent_dim'],
                    model_grad_thru_enc=config['model_grad_thru_enc'],
                    learning_rate=config['learning_rate'],
                    hidden_sizes=config['hidden_sizes_model'],
                    weight_normalization=config['weight_normalization_model'],
                    num_models=config['num_models'],
                    valid_split_ratio=config['valid_split_ratio'],
                    rolling_average_persitency=config['rolling_average_persitency'],
                    hidden_nonlinearity=config['hidden_nonlinearity_model'],
                    batch_size=config['batch_size_model'],
                    normalize_input=config['normalize'],
                    buffer=buffer,
                    input_is_img=config['input_is_img'],
                    cpc_loss_weight=config['cpc_loss_weight'],
                    cpc_model=cpc_model,
                )

                reward_model = MLPRewardEnsemble(
                    name="rew_model",
                    env=env,
                    encoder=encoder,
                    input_is_img=config['input_is_img'],
                    latent_dim=config['latent_dim'],
                    buffer=buffer,
                    learning_rate=config['learning_rate'],
                    hidden_sizes=config['hidden_sizes_model'],
                    weight_normalization=config['weight_normalization_model'],
                    num_models=config['num_models'],
                    valid_split_ratio=config['valid_split_ratio'],
                    rolling_average_persitency=config['rolling_average_persitency'],
                    hidden_nonlinearity=config['hidden_nonlinearity_model'],
                    batch_size=config['batch_size_model'],
                    normalize_input=config['normalize'],
                )

                policy = MPCController(
                    name="policy",
                    env=env,
                    num_stack=config['obs_stack'] if config['env_produce_img'] else 1,
                    encoder=encoder,
                    latent_dim=config['latent_dim'],
                    dynamics_model=dynamics_model,
                    discount=config['discount'],
                    n_candidates=config['n_candidates'],
                    horizon=config['horizon'],
                    use_cem=config['use_cem'],
                    num_cem_iters=config['num_cem_iters'],
                    use_reward_model=config['use_reward_model'],
                    reward_model=reward_model if config['use_reward_model'] else None,
                    use_image=config['env_produce_img'],
                )


            else:
                from meta_mb.dynamics.mlp_dynamics_ensemble import MLPDynamicsEnsemble
                from meta_mb.reward_model.mlp_reward_ensemble import MLPRewardEnsemble
                from meta_mb.policies.mpc_controller import MPCController
                buffer = None
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
                    normalize_input=config['normalize'],
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
                    reward_model=reward_model if config['use_reward_model'] else None,
                )


            sample_processor = ModelSampleProcessor(recurrent=config['env_produce_img'])


        sampler = BaseSampler(
            env=env,
            policy=policy,
            num_rollouts=config['num_rollouts'],
            max_path_length=max_path_length,
        )
        sampler_initial_cpc = BaseSampler(
            env=env,
            policy=policy,
            num_rollouts=config['cpc_num_initial_rollouts'],
            max_path_length=max_path_length,
        )



        algo = Trainer(
            env=env,
            policy=policy,
            dynamics_model=dynamics_model,
            reward_model=reward_model if config['use_reward_model'] else None,
            sampler=sampler,
            buffer=buffer,
            dynamics_sample_processor=sample_processor,
            reward_sample_processor=ModelSampleProcessor(recurrent=False),
            n_itr=config['n_itr'],
            dynamics_model_max_epochs=config['dynamic_model_epochs'],
            reward_model_max_epochs=config['reward_model_epochs'],
            sess=sess,
            cpc_model=cpc_model,
            cpc_terms = config['history'],
            cpc_predict_terms=config['future'],
            cpc_epoch=config['cpc_epoch'],
            cpc_lr=config['cpc_lr'],
            cpc_initial_epoch=config['cpc_initial_epoch'],
            cpc_initial_lr = config['cpc_initial_lr'],
            cpc_negative_samples=config['negative'],
            cpc_negative_same_traj=config['negative'] // 3 * 2 if config['env'] == 'reacher_easy' else 0,
            cpc_initial_sampler=sampler_initial_cpc,
            cpc_train_interval=config['cpc_train_interval'],
            cpc_predict_action=config['predict_action'],
            cpc_contrastive=config['contrastive'],
            cpc_batch_size=config['batch_size_model'],

            path_checkpoint_interval=config['path_checkpoint_interval']
        )
        algo.train()


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('EXP_NAME', type=str)
    #
    # args = parser.parse_args()


    # -------------------- Define Variants -----------------------------------

    config_normal = {
        'seed': [1],
        'run_suffix': ['1'],

        # Problem

        'env': ['walker'],
        'normalize': [False],
        'n_itr': [150],
        'discount': [1.],
        'obs_stack': [5],
        'img_shape': [(32, 32, 3)],

        # Policy
        'n_candidates': [1000],  # K
        'horizon': [12],  # Tau
        'use_cem': [True],
        'num_cem_iters': [5],
        'use_graph': [True],

        # Training
        'num_rollouts': [5],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.2],
        'rolling_average_persitency': [0.9],
        'path_checkpoint_interval': [10],

        # Dynamics Model / reward model
        'recurrent': [False],
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(500, 500)],
        'dynamic_model_epochs': [15],
        'reward_model_epochs': [15],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'cell_type': ['lstm'],
        'use_reward_model': [True],
        'input_is_img': [True],
        'model_grad_thru_enc': [True],
        'prob_dyn': [False],
        #  Other
        'n_parallel': [1],

        # representation learning

        'use_image': [True],
        'encoder': ['cpc'],
        'latent_dim': [16],
        'negative': [10],
        'history': [3],
        'future': [3],
        'use_context_net': [False],
        'include_action': [False],
        'predict_action': [False],
        'contrastive': [True],
        'cpc_epoch': [0],
        'cpc_lr': [5e-4],
        'cpc_initial_epoch': [0],
        'cpc_initial_lr': [1e-3],
        'cpc_num_initial_rollouts': [64],
        'cpc_train_interval': [1],
        'cpc_loss_weight': [1, 10],
        'cpc_lambd': [0],
        'grad_penalty': [False],
    }

    config_pretrain = {
        'seed': [1],
        'run_suffix': ['1'],

        # Problem
        'env': ['cheetah_run', 'walker'],
		'env_produce_img': [False],
        'normalize': [True],
        'n_itr': [150],
        'discount': [1.],
        'obs_stack': [3],
        'img_shape': [(32, 32, 3)],

        # Policy
        'n_candidates': [1000],  # K
        'horizon': [12],  # Tau
        'use_cem': [True],
        'num_cem_iters': [5],
        'use_graph': [True],

        # Training
        'num_rollouts': [5],
        'learning_rate': [0.001],
        'valid_split_ratio': [0.2],
        'rolling_average_persitency': [0.9],
        'path_checkpoint_interval': [10],

        # Dynamics Model / reward model
        'recurrent': [False],
        'num_models': [5],
        'hidden_nonlinearity_model': ['relu'],
        'hidden_sizes_model': [(500, 500)],
        'dynamic_model_epochs': [50],
        'reward_model_epochs': [15],
        'backprop_steps': [100],
        'weight_normalization_model': [False],  # FIXME: Doesn't work
        'batch_size_model': [64],
        'cell_type': ['lstm'],
        'use_reward_model': [True],
        'input_is_img': [False],
        'model_grad_thru_enc': [False],
        'prob_dyn': [False],
        #  Other
        'n_parallel': [1],

        # representation learning

        'use_image': [True],
        'encoder': ['cpc'],
        'latent_dim': [16],
        'negative': [10],
        'history': [3],
        'future': [3],
        'use_context_net': [False],
        'include_action': [True, False],
        'predict_action': [False],
        'contrastive': [True],
        'cpc_epoch': [0],
        'cpc_lr': [5e-4],
        'cpc_initial_epoch': [30],
        'cpc_initial_lr': [1e-3],

        'cpc_num_initial_rollouts': [256],
        'cpc_train_interval': [1],
        'cpc_loss_weight': [0.],
        'cpc_lambd': [0],
        'grad_penalty': [False],
    }


    run_sweep(run_experiment, config_pretrain, EXP_NAME, INSTANCE_TYPE)
