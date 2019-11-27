import os
import json
import ray
from datetime import datetime

from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder
from meta_mb.envs.mb_envs.maze import ParticleMazeEnv
from meta_mb.envs.robotics.fetch.reach import FetchReachEnv
from meta_mb.envs.robotics.fetch.push import FetchPushEnv
from meta_mb.envs.robotics.fetch.slide import FetchSlideEnv
from meta_mb.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from meta_mb.trainers.self_play_trainer_v1 import TrainerV1
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
    env = kwargs['env']()

    # preprocess kwargs
    if kwargs['env'] is ParticleMazeEnv:
        env_name = 'PMazeEnv-' + env.name
        if env.name == 'easy':
            kwargs['n_itr'] = 51
            kwargs['snapshot_gap'] = 10
        elif env.name == 'medium':
            kwargs['n_itr'] = 201
            kwargs['snapshot_gap'] = 40
    elif kwargs['env'] is FetchReachEnv:
        env_name = 'FReachEnv'
        kwargs['n_itr'] = 41
        kwargs['snapshot_gap'] = 10
    elif kwargs['env'] is FetchPushEnv:
        env_name = 'FPushEnv'
        kwargs['n_itr'] = 201
        kwargs['snapshot_gap'] = 40
    elif kwargs['env'] is FetchPickAndPlaceEnv:
        env_name = 'FP&PEnv'
    elif kwargs['env'] is FetchSlideEnv:
        env_name = 'FSlideEnv'
        kwargs['n_itr'] = 201
        kwargs['snapshot_gap'] = 40
    else:
        raise NotImplementedError
    # kwargs['refresh_interval'], kwargs['num_mc_goals'], kwargs['goal_buffer_size'] = kwargs['goal_sampling_params']
    # assert kwargs['num_mc_goals'] >= kwargs['goal_buffer_size']

    if kwargs.get('exp_name') is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        kwargs['exp_name'] = f"{env_name}-alpha-{kwargs['goal_buffer_alpha']}-replay-{kwargs['replay_k']}-{kwargs['action_noise_str']}-{timestamp}"
    else:
        kwargs['exp_name'] = f"{env_name}-alpha-{kwargs['goal_buffer_alpha']}-replay-{kwargs['replay_k']}-{kwargs['action_noise_str']}-" + kwargs['exp_name']

    exp_dir = os.path.join(os.getcwd(), "data", EXP_NAME, kwargs['exp_name'])

    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    for k, v in kwargs.items():
        logger.logkv(k, v)
    logger.dumpkvs()

    logger.log('ray init...', ray.init())
    trainer = TrainerV1(
        num_agents=kwargs['num_agents'],
        seeds=kwargs['seeds'],
        instance_kwargs=kwargs,
        gpu_frac=kwargs.get('gpu_frac', 0.95),
        env=env,
        num_mc_goals=kwargs['num_mc_goals'],
        refresh_interval=kwargs['refresh_interval'],
        alpha=kwargs['goal_buffer_alpha'],
        eval_interval=kwargs['eval_interval'],
        n_itr=kwargs['n_itr'],
        exp_dir=exp_dir,
        greedy_eps=kwargs['greedy_eps'],
        num_grad_steps=kwargs['num_grad_steps'],
        snapshot_gap=kwargs['snapshot_gap'],
    )

    trainer.train()
    logger.log('ray shutdown...', ray.shutdown())


if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        'seeds': [(6,7,8,9,10)],  # (1,2,3,4,5)]
        'baseline': [LinearFeatureBaseline],
        'env': [ParticleMazeEnv], #FetchPickAndPlaceEnv, FetchSlideEnv], #[FetchReachEnv], [ParticleMazeEnv],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': ['tanh'], #['relu'],  # TODO
        'policy_output_nonlinearity': [None],
        'num_grad_steps': [-1],
        'policy_max_std': [2e0],
        'policy_min_std': [1e-3],

        # Value function
        'vfun_hidden_nonlinearity': ['tanh'],  # TODO
        'vfun_output_nonlinearity': [None],

        # Goal Sampling
        # goal_sampling_params = (refresh_interval, num_mc_goals, goal_buffer_size)
        # need num_mc_goals > goal_buffer_size to avoid error (repeated sampling not allowed)
        'refresh_interval': [1, 2],
        'num_mc_goals': [100],
        'goal_buffer_size': [50],
        'goal_buffer_alpha': [0, 0.5], # TODO
        'goal_sampling_rule': ['norm_diff'], #'softmax'],  # ['softmax'],

        # Env Sampling
        'num_rollouts': [1],
        'n_parallel': [1],
        'max_replay_buffer_size': [1e5],
        'eval_interval': [1],
        'replay_k': [3], # 4, -1],
        'greedy_eps': [0.1, 0.3],
        'action_noise_str': ['none', 'ou_0.05'],
        # 'curiosity_percentage': [0.8],

        # Problem Conf
        'num_agents': [3],
        'max_path_length': [50], #100],
        'discount': [0.99], #0.95],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
        'reward_scale': [1.],
        'sampler_batch_size': [256],
        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
