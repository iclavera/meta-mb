import argparse
from experiment_utils.utils import load_exps_data
import os
import shutil

"""
 python /home/ignasi/GitRepos/meta-mb/experiment_utils/save_videos.py data/s3/mbmpo-pieter/ --speedup 4 -n 1 --max_path_length 300 --ignore_done
"""


def valid_experiment(params, lr, rl, env):
    #           'dyanmics_hidden_nonlinearity': ['relu'],
    #           'dynamics_buffer_size': [10000],
    #           'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}

    # 'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}
    # 'env': [{'$class': 'meta_mb.envs.mujoco.ant_env.AntEnv'}]}
    # 'env': [{'$class': 'meta_mb.envs.mujoco.hopper_env.HopperEnv'}]}
    # #
    values = {
             # 'env': [{'$class': 'meta_mb.envs.mb_envs.pendulumO001.PendulumO001Env'},
             # {'$class': 'meta_mb.envs.mb_envs.pendulumO01.PendulumO01Env'}
             # ]
              'env': [{'$class': 'meta_mb.envs.mb_envs.' + env}],
              'inner_lr': [lr],
              'rollouts_per_meta_task': [rl],
              'max_path_length': [1000],
              'sample_from_buffer': [False],
              'meta_steps_per_iter': [[50, 50]],
              'n_itr': [201],
              'fraction_meta_batch_size': [1.],
              }

    for k, v in values.items():
        if params[k] not in v:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--store", type=str,
            default=os.path.join(os.path.expanduser('~'), 'bench_data_lasts'))
    parser.add_argument('--lr', type=float)
    parser.add_argument('--rl', type=float)
    parser.add_argument('--env', type=str)
    args = parser.parse_args()

    experimet_paths = load_exps_data(args.data, gap=0.)
    counter = 0
    for exp_path in experimet_paths:
        json_file = exp_path['json']
        if valid_experiment(json_file, args.lr, args.rl, args.env):
            env_name = json_file['env']['$class'].split('.')[-2]
            dir_name = os.path.join(args.store, json_file['algo'], env_name, env_name + str(counter))
            os.makedirs(dir_name)
            shutil.copy2(os.path.join(exp_path['exp_name'], "params.json"), dir_name)
            shutil.copy2(os.path.join(exp_path['exp_name'], "progress.csv"), dir_name)
            counter += 1





