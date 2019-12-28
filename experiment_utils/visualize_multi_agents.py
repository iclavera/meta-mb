import matplotlib.pyplot as plt
import joblib
import argparse
import os
import numpy as np
import tensorflow as tf

from experiment_utils.utils import load_exps_data
from experiment_utils.fetch_env_visualizer import FetchEnvVisualizer
from experiment_utils.maze_env_visualizer import MazeEnvVisualizer
from meta_mb.envs.robotics import FetchEnv
from meta_mb.envs.mb_envs.maze import ParticleMazeEnv


def plot_fetch_env(dir_path_list, max_path_length, num_rollouts=None, gap=1, min_pkl=None, max_pkl=None,
                   ignore_done=False, stochastic=False, force_reload=False):
    exps = []
    for dir_path in dir_path_list:
        exps.extend(load_exps_data(dir_path, gap=1, max=None))

    for exp in exps:
        print('plotting', exp['exp_name'])
        num_agents = exp['json']['num_agents']

        tf.reset_default_graph()
        with tf.Session():
            vis, fig, ax_arr = None, None, None

            for itr in range(len(exp['pkl'])//num_agents):
                if itr % gap != 0 or (min_pkl is not None and itr < min_pkl) or (max_pkl is not None and itr > max_pkl):
                    continue
                image_path = os.path.join(exp['exp_name'], f"itr_{itr}.png")
                if not force_reload and os.path.exists(image_path):
                    continue

                fig, ax_arr = plt.subplots(nrows=num_agents, ncols=6, figsize=(30, 15))
                ax_arr = ax_arr.reshape((num_agents, 6))

                q_values_list = []
                pkl_paths = exp['pkl'][itr*num_agents:(itr+1)*num_agents]
                for pkl_path in pkl_paths:
                    base_name, _ = os.path.splitext(os.path.basename(pkl_path))
                    _, agent_idx, _, _ = base_name.split('_')
                    agent_idx = int(agent_idx)

                    print(f"loading itr {itr}...")
                    data = joblib.load(pkl_path)

                    if vis is None:
                        env = data['env']
                        eval_goals = env.sample_2d_goals(num_samples=num_rollouts)
                        discount = exp['json']['discount']
                        _max_path_length = exp['json']['max_path_length'] if max_path_length is None else max_path_length
                        if isinstance(env, ParticleMazeEnv):
                            vis = MazeEnvVisualizer(env, eval_goals, _max_path_length, discount, ignore_done, stochastic)
                        else:
                            vis = FetchEnvVisualizer(env, eval_goals, _max_path_length, discount, ignore_done, stochastic)

                    q_values = vis.do_plots(fig, ax_arr[agent_idx, :], policy=data['policy'], q_functions=data['Q_targets'], goal_samples=data['goal_samples'], itr=itr)
                    q_values_list.append(q_values)

                # plot disagreement-based goal distribution
                expert_q_values = np.max(q_values_list, axis=0)
                for agent_idx in range(num_agents):
                    diff = expert_q_values - q_values_list[agent_idx]
                    if np.sum(diff) == 0:
                        goal_dist = np.ones((len(diff),)) / len(diff)
                    else:
                        goal_dist = diff / np.sum(diff)
                    vis._goal_distribution_helper(fig, ax_arr[agent_idx, 1], goal_dist, "disagreement")

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

