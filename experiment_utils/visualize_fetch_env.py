import matplotlib.pyplot as plt
import joblib
import argparse
import os
import tensorflow as tf

from experiment_utils.utils import load_exps_data
from meta_mb.agents.fetch_env_visualizer import FetchEnvVisualizer
from meta_mb.agents.maze_env_visualizer import MazeEnvVisualizer
from meta_mb.envs.robotics import FetchEnv
from meta_mb.envs.mb_envs.maze import ParticleMazeEnv


NUM_EVAL_GOALS = 1

def plot_fetch_env(dir_path_list, max_path_length, num_rollouts=None, gap=1, min_pkl=None, max_pkl=None,
                   ignore_done=False, stochastic=False, force_reload=False):
    exps = []
    for dir_path in dir_path_list:
        exps.extend(load_exps_data(dir_path, gap=1, max=None))

    with tf.Session():
        for exp in exps:
            print('plotting', exp['exp_name'])

            vis, fig, ax_arr = None, None, None

            for itr_idx in range(len(exp['pkl'])//2):
                pkl_paths = exp['pkl'][itr_idx*2:itr_idx*2+2]
                base_name = os.path.splitext(os.path.basename(pkl_paths[0]))[0]
                _, _, itr = base_name.split('_')
                itr = int(itr)

                if itr % gap != 0 or (min_pkl is not None and itr < min_pkl) or (max_pkl is not None and itr > max_pkl):
                    continue

                image_path = os.path.join(exp['exp_name'], f"itr_{itr}.png")
                if not force_reload and os.path.exists(image_path):
                    continue

                fig, ax_arr = plt.subplots(nrows=1, ncols=6, figsize=(30, 4))

                print(f"loading itr {itr}...")
                data = joblib.load(pkl_paths[0])
                data.update(joblib.load(pkl_paths[1]))

                if vis is None:
                    env = data['env']
                    eval_goals = env.sample_2d_goals(num_samples=num_rollouts)
                    discount = exp['json']['discount']
                    _max_path_length = exp['json']['max_path_length'] if max_path_length is None else max_path_length
                    if isinstance(env, ParticleMazeEnv):
                        vis = MazeEnvVisualizer(env, eval_goals, _max_path_length, discount, ignore_done, stochastic)
                    else:
                        vis = FetchEnvVisualizer(env, eval_goals, _max_path_length, discount, ignore_done, stochastic)

                vis.do_plots(fig, ax_arr, policy=data['policy'], q_functions=data['Q_targets'], value_ensemble=data['vfun_tuple'], goal_samples=data.get('goal_samples', []), itr=itr)

                plt.savefig(image_path)
                plt.clf()
                plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str, nargs='*')
    parser.add_argument('--num_rollouts', '-n', type=int, default=1,
                        help='Max length of rollout')
    parser.add_argument('--gap_pkl', type=int, default=1,
                        help='Gap between pkl policies')
    parser.add_argument('--min_pkl', type=int, default=None,
                        help='Minimum value of the pkl policies')
    parser.add_argument('--max_pkl', type=int, default=None,
                        help='Maximum value of the pkl policies')
    parser.add_argument("--max_path_length", "-l", type=int, default=None,
                        help="Max length of rollout")
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    parser.add_argument('--stochastic', action='store_true',
                        help='Apply stochastic action instead of deterministic')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reloading images')
    args = parser.parse_args()

    plot_fetch_env(dir_path_list=args.path, max_path_length=args.max_path_length, num_rollouts=args.num_rollouts,
              gap=args.gap_pkl, min_pkl=args.min_pkl, max_pkl=args.max_pkl, ignore_done=args.ignore_done, stochastic=args.stochastic,
              force_reload=args.force_reload)

