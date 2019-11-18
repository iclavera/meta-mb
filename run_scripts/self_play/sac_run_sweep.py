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
from meta_mb.trainers.self_play_trainer import Trainer
from meta_mb.logger import logger
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
# from meta_mb.envs.normalized_env import normalize
from meta_mb.utils.utils import set_seed


INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'sac'
GLOBAL_SEED = 1


def run_experiment(**kwargs):
    set_seed(GLOBAL_SEED)

    if kwargs['env'] is ParticleMazeEnv:
        env_name = 'PMazeEnv'
        # kwargs['n_itr'] = 1001
        # kwargs['snapshot_gap'] = 100
    elif kwargs['env'] is FetchReachEnv:
        env_name = 'FReachEnv'
    elif kwargs['env'] is FetchPushEnv:
        env_name = 'FPushEnv'
    elif kwargs['env'] is FetchPickAndPlaceEnv:
        env_name = 'FP&PEnv'
    elif kwargs['env'] is FetchSlideEnv:
        env_name = 'FSlideEnv'
    else:
        raise NotImplementedError

    if kwargs.get('exp_name') is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        kwargs['exp_name'] = f"agent_{kwargs['num_agents']}-{env_name}-{kwargs['goal_buffer_alpha']}-replay_k-{kwargs['replay_k']}-{timestamp}"

    exp_dir = os.path.join(os.getcwd(), "data", EXP_NAME, kwargs['exp_name'])

    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # env = normalize(kwargs['env']())  # FIXME
    env = kwargs['env']()

    for k, v in kwargs.items():
        logger.logkv(k, v)
    logger.dumpkvs()

    logger.log('ray init...', ray.init())
    trainer = Trainer(
        num_agents=kwargs['num_agents'],
        seeds=kwargs['seeds'],
        instance_kwargs=kwargs,
        gpu_frac=kwargs.get('gpu_frac', 0.95),
        env=env,
        num_sample_goals=kwargs['num_sample_goals'],
        alpha=kwargs['goal_buffer_alpha'],
        eval_interval=kwargs['eval_interval'],
        n_itr=kwargs['n_itr'],
        exp_dir=exp_dir,
        goal_update_interval=kwargs['goal_update_interval'],
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
        'policy_output_nonlinearity': [None],
        'num_grad_steps': [-1],

        # Env Sampling
        'num_rollouts': [1],
        'n_parallel': [1],
        'max_replay_buffer_size': [1e5],
        'max_goal_buffer_size': [30],
        'num_sample_goals': [20],
        'goal_buffer_alpha': [0, 0.5, 1], #[0, 0.5, 1],  # [0, 0.1, 0.5, 0.9, 1],
        'goal_update_interval': [2],
        'eval_interval': [1],
        'sample_rule': ['norm_diff'],  # 'softmax'],
        'replay_k': [4, -1],
        # 'curiosity_percentage': [0.8],

        # Problem Conf
        'num_agents': [3],
        'n_itr': [501], # [3001],
        'snapshot_gap': [10], # [500],
        'max_path_length': [50], #100],
        'discount': [0.95], #0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
        'reward_scale': [1.],
        'sampler_batch_size': [256],
        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
