import argparse
from experiment_utils.utils import load_exps_data
import os
import shutil

"""
 python /home/ignasi/GitRepos/meta-mb/experiment_utils/save_videos.py data/s3/mbmpo-pieter/ --speedup 4 -n 1 --max_path_length 300 --ignore_done
"""


def valid_experiment(params):
    #           'dyanmics_hidden_nonlinearity': ['relu'],
    #           'dynamics_buffer_size': [10000],
    #           'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}

    # 'env': [{'$class': 'meta_mb.envs.mujoco.walker2d_env.Walker2DEnv'}]}
    # 'env': [{'$class': 'meta_mb.envs.mujoco.ant_env.AntEnv'}]}
    # 'env': [{'$class': 'meta_mb.envs.mujoco.hopper_env.HopperEnv'}]}
    # #
    values = {'svg_learning_rate': [0.001],
              'kl_penalty': [1],
              'env': [{'$class': 'meta_mb.envs.mb_envs.acrobot.AcrobotEnv'}]
              }


    for k, v in values.items():
        if params[k] not in v:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("store", type=str, default=os.path.expanduser('~'))
    args = parser.parse_args()

    experimet_paths = load_exps_data(args.data, gap=0.)
    counter = 0
    for exp_path in experimet_paths:
        json_file = exp_path['json']
        if valid_experiment(json_file):
            env_name = json_file['env'].split('.')[-2]
            dir_name = os.path.join(args.store, json_file['algo'], env_name, env_name + str(counter))
            os.makedirs(dir_name)
            shutil.copy2(os.path.join(exp_path, "params.json"), dir_name)
            shutil.copy2(os.path.join(exp_path, "progress.csv"), dir_name)
            counter += 1





