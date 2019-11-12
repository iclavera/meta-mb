import os
import json
import ray
from datetime import datetime

from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder
from meta_mb.envs.mb_envs.maze import ParticleEnv, ParticleMazeEnv, ParticleFixedEnv
from meta_mb.trainers.self_play_trainer import Trainer
from meta_mb.logger import logger
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.normalized_env import normalize
from meta_mb.utils.utils import set_seed


INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'sac'
GLOBAL_SEED = 1


def run_experiment(**kwargs):
    set_seed(GLOBAL_SEED)

    if kwargs['env'] is ParticleEnv:
        env_name = 'PEnv'
    elif kwargs['env'] is ParticleFixedEnv:
        env_name = 'PFixedEnv'
    elif kwargs['env'] is ParticleMazeEnv:
        env_name = 'PMazeEnv'
    else:
        raise NotImplementedError

    if kwargs.get('exp_name') is None:
        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        kwargs['exp_name'] = "%s" % timestamp

    exp_dir = os.path.join(os.getcwd(), "data", EXP_NAME, f"agent_{kwargs['num_agents']}-{env_name}-{kwargs['goal_buffer_alpha']}-{kwargs['reward_str']}-{kwargs['sample_rule']}-{kwargs['exp_name']}")

    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(kwargs['env'](grid_name=kwargs['grid_name'], reward_str=kwargs['reward_str']))

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
        goal_update_interval=kwargs['goal_update_interval']
    )

    trainer.train()
    logger.log('ray shutdown...', ray.shutdown())

if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        'seeds': [(1,2,3,4,5)],
        'baseline': [LinearFeatureBaseline],
        'env': [ParticleMazeEnv],
        'grid_name': ['novel1'],
        'reward_str': ['sparse'], #'L1', 'L2'],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],

        # Env Sampling
        'num_rollouts': [1],
        'n_parallel': [1],
        'max_replay_buffer_size': [1e5],
        'max_goal_buffer_size': [30],
        'num_sample_goals': [20],
        'goal_buffer_alpha': [0.1, 0.5, 0.9], #[0, 0.1, 0.5, 0.9, 1],
        'goal_update_interval': [2],
        'eval_interval': [20],
        'sample_rule': ['norm_diff'], #'softmax'],

        # Problem Conf
        'num_agents': [3],
        'n_itr': [3001],
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
