import matplotlib.pyplot as plt
import joblib
import argparse
import os
import numpy as np
import tensorflow as tf
from collections import defaultdict

from experiment_utils.utils import load_exps_data
from meta_mb.agents.maze_visualizer import MazeVisualizer

NUM_EVAL_GOALS = 1

def plot_maze(dir_path, max_path_length, num_rollouts=None, gap=1, min_pkl=None, max_pkl=None, ignore_done=False, stochastic=False, plot_eval_returns=False, force_reload=False):
    exps = load_exps_data(dir_path, gap=1, max=None)
    eval_goals = None

    for exp in exps:
        print('plotting', exp['exp_name'])
        vis = None
        tf.reset_default_graph()
        with tf.Session():
            q_values_train_goals_dict = defaultdict(dict)

            for pkl_path in exp['pkl']:
                parent_path = os.path.dirname(pkl_path)
                base_name = os.path.splitext(os.path.basename(pkl_path))[0]
                _, itr, _, agent_idx = base_name.split('_')
                if int(itr) % gap != 0 or (min_pkl is not None and int(itr) < min_pkl) or (max_pkl is not None and int(itr) > max_pkl):
                    continue
                print(f'processing itr {itr}, agent {agent_idx}')
                data = joblib.load(pkl_path)

                # FIXME: hacky fix for wrong .pkl
                from meta_mb.envs.mb_envs.maze import ParticleMazeEnv
                data['env'] = ParticleMazeEnv(grid_name='novel1', reward_str='sparse')

                if eval_goals is None:
                    eval_goals = data['env'].sample_goals(mode=None, num_samples=num_rollouts)
                if vis is None:
                    discount = exp['json']['discount']
                    _max_path_length = exp['json']['max_path_length'] if max_path_length is None else max_path_length
                    vis = MazeVisualizer(data['env'], eval_goals, _max_path_length, discount, ignore_done, stochastic, parent_path, force_reload)

                q_values_train_goals = vis.do_plots(policy=data['policy'], q_ensemble=data['Q_targets'], base_title=f"itr_{itr}_agent_{agent_idx}", plot_eval_returns=plot_eval_returns)
                q_values_train_goals_dict[itr][agent_idx] = q_values_train_goals

            # plot goal distribution, split by itr
            for itr, agent_q_dict in q_values_train_goals_dict.items():
                vis.plot_goal_distributions(agent_q_dict=agent_q_dict, sample_rule=exp['json']['sample_rule'], alpha=exp['json']['goal_buffer_alpha'], itr=itr)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
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
    parser.add_argument('--plot_eval_returns', action='store_true',
                        help='Plot evaluated returns by collecting on-policy rollouts')
    parser.add_argument('--force_reload', action='store_true',
                        help='Force reloading images')
    args = parser.parse_args()

    plot_maze(dir_path=args.path, max_path_length=args.max_path_length, num_rollouts=args.num_rollouts,
              gap=args.gap_pkl, min_pkl=args.min_pkl, max_pkl=args.max_pkl, ignore_done=args.ignore_done, stochastic=args.stochastic,
              plot_eval_returns=args.plot_eval_returns, force_reload=args.force_reload)

