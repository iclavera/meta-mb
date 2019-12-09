import os
import json
from datetime import datetime

from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder
from meta_mb.envs.mb_envs.maze import ParticleMazeEnv
from meta_mb.envs.robotics.fetch.reach import FetchReachEnv
from meta_mb.envs.robotics.fetch.push import FetchPushEnv
from meta_mb.envs.robotics.fetch.slide import FetchSlideEnv
from meta_mb.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from meta_mb.trainers.ve_self_play_trainer import Trainer
from meta_mb.logger import logger
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
# from meta_mb.envs.normalized_env import normalize
from meta_mb.utils.utils import set_seed


INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'sac'
GLOBAL_SEED = 1


def run_experiment(**kwargs):
    set_seed(GLOBAL_SEED)

    # env = normalize(kwargs['env']())  # FIXME

    # preprocess kwargs
    if kwargs['env'] is ParticleMazeEnv:
        env = kwargs['env']()
        env_name = 'PMazeEnv-' + env.name
        if env.name == 'easy':
            kwargs['n_itr'] = 1501 # 5001
            kwargs['snapshot_gap'] = 500
        elif env.name == 'medium':
            kwargs['n_itr'] = 20001
            kwargs['snapshot_gap'] = 500
    elif kwargs['env'] is FetchReachEnv:
        env = kwargs['env'](obj_range=0)
        env_name = 'FReachEnv'
        kwargs['n_itr'] = 1501
        kwargs['snapshot_gap'] = 100
    elif kwargs['env'] is FetchPushEnv:
        env = kwargs['env'](obj_range=0)
        env_name = 'FPushEnv'
        kwargs['n_itr'] = 50001
        kwargs['snapshot_gap'] = 500
    elif kwargs['env'] is FetchPickAndPlaceEnv:
        env = kwargs['env'](obj_range=0)
        env_name = 'FPAPEnv'
        kwargs['n_itr'] = 50001
        kwargs['snapshot_gap'] = 500
    elif kwargs['env'] is FetchSlideEnv:
        env = kwargs['env'](obj_range=0)
        env_name = 'FSlideEnv'
        kwargs['n_itr'] = 50001
        kwargs['snapshot_gap'] = 500
    else:
        raise NotImplementedError
    # kwargs['refresh_interval'], kwargs['num_mc_goals'], kwargs['goal_buffer_size'] = kwargs['goal_sampling_params']
    # assert kwargs['num_mc_goals'] >= kwargs['goal_buffer_size']

    exp_name = f"{env_name}-ve-{kwargs['size_value_ensemble']}"
    if kwargs.get('exp_name') is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        kwargs['exp_name'] = f"{exp_name}-{timestamp}"
    else:
        kwargs['exp_name'] = f"{exp_name}-{kwargs['exp_name']}"

    exp_dir = os.path.join(os.getcwd(), "data", EXP_NAME, kwargs['exp_name'])

    logger.configure(dir=exp_dir, format_strs=['csv', 'stdout', 'log'],
                     snapshot_mode='gap', snapshot_gap=kwargs['snapshot_gap'])
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    for k, v in kwargs.items():
        logger.log(f"{k}: {v}")

    # logger.log('ray init...', ray.init())
    trainer = Trainer(
        size_value_ensemble=kwargs['size_value_ensemble'],
        seed=kwargs['seed'],
        instance_kwargs=kwargs,
        gpu_frac=kwargs.get('gpu_frac', 0.95),
        env=env,
        eval_interval=kwargs['eval_interval'],
        n_itr=kwargs['n_itr'],
        greedy_eps=kwargs['greedy_eps'],
    )

    trainer.train()
    # logger.log('ray shutdown...', ray.shutdown())


if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        'seed': [1],
        'baseline': [LinearFeatureBaseline],
        'env': [ParticleMazeEnv, FetchReachEnv], #[ParticleMazeEnv], #[FetchPickAndPlaceEnv, FetchSlideEnv, FetchPushEnv], #[FetchReachEnv], [ParticleMazeEnv],

        # Value ensemble
        'size_value_ensemble': [0],
        'vfun_batch_size': [-1],
        'vfun_num_grad_steps': [-1],
        'vfun_max_replay_buffer_size': [-1],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': ['tanh'], #['relu'],  # TODO
        'policy_output_nonlinearity': [None],
        'policy_num_grad_steps': [500, 200],  # 100
        'policy_max_std': [2e0],
        'policy_min_std': [1e-3],
        'policy_max_replay_buffer_size': [1e5],
        'learning_rate': [3e-4],
        'target_update_interval': [1, 10],

        # Value function
        'vfun_hidden_nonlinearity': ['tanh'],  # TODO
        'vfun_output_nonlinearity': [None],
        'vfun_hidden_sizes': [(256, 256)],

        # Env Sampling
        'num_mc_goals': [1000],
        'num_rollouts': [50],
        'n_parallel': [1],
        'max_replay_buffer_size': [1e5],
        'eval_interval': [100],
        'replay_k': [-1], # 4, -1],
        'greedy_eps': [0.3], #[0, 0.1], #0.1, 0.3],  any exploration not following policy would introduce problem for training value ensemble
        'action_noise_str': ['none'], #'ou_0.05'],
        # 'curiosity_percentage': [0.8],

        # Problem Conf
        'max_path_length': [50], #100],
        'discount': [0.99], #0.95],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'reward_scale': [1.],
        'sampler_batch_size': [512],
        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
