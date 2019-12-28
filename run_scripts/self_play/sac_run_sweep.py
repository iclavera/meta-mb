import os
import json
import tensorflow as tf
from datetime import datetime

from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder
from meta_mb.envs.mb_envs.maze import ParticleMazeEnv
from meta_mb.envs.robotics.fetch.reach import FetchReachEnv
from meta_mb.envs.robotics.fetch.push import FetchPushEnv
from meta_mb.envs.robotics.fetch.slide import FetchSlideEnv
from meta_mb.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from meta_mb.trainers.self_play_trainer_v2 import Trainer
from meta_mb.logger import logger
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
# from meta_mb.envs.normalized_env import normalize
from meta_mb.utils.utils import set_seed


INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'multi-sac'


def run_experiment(**kwargs):
    set_seed(kwargs['seed'])

    # env = normalize(kwargs['env']())  # FIXME

    # preprocess kwargs
    if kwargs['env'] is ParticleMazeEnv:
        env = ParticleMazeEnv(grid_name='guided')
        env_name = 'PMazeEnv-' + env.name
        if env.name == 'easy':
            kwargs['n_itr'] = 501 # 5001
        elif env.name == 'medium':
            kwargs['n_itr'] = 2001
        elif env.name == 'guided':
            kwargs['n_itr'] = 501
        kwargs['snapshot_gap'] = 25
    elif kwargs['env'] is FetchReachEnv:
        env = FetchReachEnv(obj_range=0)
        env_name = 'FReachEnv'
        kwargs['n_itr'] = 5
        kwargs['snapshot_gap'] = 1
    elif kwargs['env'] is FetchPushEnv:
        env = FetchPushEnv(obj_range=0)
        env_name = 'FPushEnv'
        kwargs['n_itr'] = 1001
        kwargs['snapshot_gap'] = 25
    elif kwargs['env'] is FetchPickAndPlaceEnv:
        env = FetchPickAndPlaceEnv(obj_range=0)
        env_name = 'FPAPEnv'
        kwargs['n_itr'] = 1001
        kwargs['snapshot_gap'] = 25
    elif kwargs['env'] is FetchSlideEnv:
        env = FetchSlideEnv(obj_range=0)
        env_name = 'FSlideEnv'
        kwargs['n_itr'] = 1001
        kwargs['snapshot_gap'] = 25
    else:
        raise NotImplementedError

    exp_name = f"{env_name}-eps-{kwargs['goal_greedy_eps']}"
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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = kwargs.get('gpu_frac', 0.95)

    trainer = Trainer(
        num_agents=kwargs['num_agents'],
        instance_kwargs=kwargs,
        config=config,
        env=env,
        num_mc_goals=kwargs['num_mc_goals'],
        update_expert_interval=kwargs['update_expert_interval'],
        eval_interval=kwargs['eval_interval'],
        n_itr=kwargs['n_itr'],
        policy_greedy_eps=kwargs['policy_greedy_eps'],
        goal_greedy_eps=kwargs['goal_greedy_eps'],
    )

    trainer.train()


if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        'seed': [1],  # (1,2,3,4,5)]
        'baseline': [LinearFeatureBaseline],
        'env': [ParticleMazeEnv, FetchReachEnv], #FetchPickAndPlaceEnv, FetchSlideEnv], #[FetchReachEnv], [ParticleMazeEnv],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_hidden_nonlinearity': ['tanh'], #['relu'],  # TODO
        'policy_output_nonlinearity': [None],
        'policy_num_grad_steps': [1000, 10000],  # FIXME
        'policy_max_std': [2e0],
        'policy_min_std': [1e-3],
        'policy_max_replay_buffer_size': [1e5],
        'learning_rate': [3e-4],
        'target_update_interval': [10],  # FIXME

        # Value function
        'vfun_hidden_nonlinearity': ['relu'], # 'tanh' # TODO
        'vfun_output_nonlinearity': [None],
        'vfun_hidden_sizes': [(512, 512)], #[(256, 256)],

        # Goal Sampling
        'update_expert_interval': [1, 10],
        'num_mc_goals': [1000],
        'goal_buffer_size': [50],
        'goal_greedy_eps': [0, 0.1, 1], # TODO

        # Env Sampling
        'num_rollouts': [20],
        'n_parallel': [1],
        'eval_interval': [1],
        'replay_k': [4],
        'policy_greedy_eps': [0, 0.3],
        'action_noise_str': ['none'],

        # Problem Conf
        'num_agents': [3],
        'max_path_length': [50],
        'discount': [0.99], #0.95],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'reward_scale': [1.],
        'sampler_batch_size': [512],  # 256
        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
