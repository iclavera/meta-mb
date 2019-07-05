import argparse
from experiment_utils.utils import load_exps_data
import os
import shutil

"""
 python /home/ignasi/GitRepos/meta-mb/experiment_utils/save_videos.py data/s3/mbmpo-pieter/ --speedup 4 -n 1 --max_path_length 300 --ignore_done
"""


def valid_experiment(params, algo):
    if algo == 'mb-mpo':
        values = {
              'num_rollouts': [10],
              'rolling_average_persitency': [0.9],
              }

    elif algo == 'me-ppo':
        values = {
            'num_rollouts': [10],
            'rolling_average_persitency': [0.4],
            'clip_eps': [0.2],
            'num_ppo_steps': [5],
            'learning_rate': [0.0003]
        }

    elif algo == 'a-me-ppo':
        values = {
            'simulation_sleep_frac': [1],
            'rolling_average_persitency': [0.4],
        }

    elif algo == 'ppo':
        values = {
             'num_rollouts': [50],
             'clip_eps': [0.2],
             'num_ppo_steps': [5],
             'learning_rate': [0.001]
        }

    else:
        raise NotImplementedError

    for k, v in values.items():
        if params[k] not in v:
            return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=str)
    parser.add_argument("--store", type=str,
            default=os.path.join(os.path.expanduser('~'), 'new'))
    parser.add_argument('--algo', '-a', type=str)
    args = parser.parse_args()

    experimet_paths = load_exps_data(args.data, gap=0.)
    counter = 0
    for exp_path in experimet_paths:
        json_file = exp_path['json']
        if valid_experiment(json_file, args.algo):
            try:
                env_name = json_file['env']['$class'].split('.')[-2]
            except TypeError:
                env_name = json_file['env']
            dir_name = os.path.join(args.store, json_file['algo'], env_name, env_name + str(counter))
            os.makedirs(dir_name, exist_ok=True)
            shutil.copy2(os.path.join(exp_path['exp_name'], "params.json"), dir_name)
            shutil.copy2(os.path.join(exp_path['exp_name'], "progress.csv"), dir_name)
            counter += 1





