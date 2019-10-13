import os
import json
INSTANCE_TYPE = 'c4.xlarge'
EXP_NAME = 'sac'
GLOBAL_SEED = 1


from experiment_utils.run_sweep import run_sweep
from meta_mb.utils.utils import ClassEncoder
from meta_mb.envs.mb_envs.maze import ParticleEnv, ParticleMazeEnv, ParticleFixedEnv
from meta_mb.trainers.self_play_trainer import Trainer
from meta_mb.logger import logger
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.envs.normalized_env import normalize
from meta_mb.utils.utils import set_seed


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

    exp_dir = os.path.join(os.getcwd(), "data", EXP_NAME, f"agent_{kwargs['num_agents']}-{env_name}-{kwargs['goal_buffer_eps']}")

    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    for i in range(kwargs['num_agents']):
        os.makedirs(exp_dir + f'/agent_{i}', exist_ok=True)
        json.dump(kwargs, open(exp_dir + f'/agent_{i}/params.json', 'w+'), indent=2, sort_keys=True, cls=ClassEncoder)

    env = normalize(kwargs['env']())
    trainer = Trainer(
        num_agents=kwargs['num_agents'],
        seeds=kwargs['seeds'],
        instance_kwargs=kwargs,
        env=env,
        num_target_goals=kwargs['num_target_goals'],
        num_eval_goals_sqrt=kwargs['num_eval_goals_sqrt'],
        n_itr=kwargs['n_itr'],
        exp_dir=exp_dir,
        goal_update_interval=kwargs['goal_update_interval']
    )

    trainer.train()

if __name__ == '__main__':
    sweep_params = {
        'algo': ['sac'],
        'seeds': [(1,2)],
        'baseline': [LinearFeatureBaseline],
        'env': [ParticleMazeEnv],

        # Policy
        'policy_hidden_sizes': [(256, 256)],
        'policy_learn_std': [True],
        'policy_output_nonlinearity': [None],

        # Env Sampling
        'num_rollouts': [1],
        'n_parallel': [1],
        'max_replay_buffer_size': [1e5],
        'max_goal_buffer_size': [10],
        'goal_buffer_eps': [0.1],
        'goal_update_interval': [2],
        'num_eval_goals_sqrt': [3],

        # Problem Conf
        'num_agents': [2],
        'num_target_goals': [5],
        'n_itr': [3000],
        'max_path_length': [50],
        'discount': [0.99],
        'gae_lambda': [1.],
        'normalize_adv': [True],
        'positive_adv': [False],
        'learning_rate': [3e-4],
        'reward_scale': [1.],
        'sampler_batch_size': [256],
        }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)
