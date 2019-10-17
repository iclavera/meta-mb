import matplotlib.pyplot as plt
import joblib
import argparse
import tensorflow as tf
from experiment_utils.utils import load_exps_data
from meta_mb.agents.maze_visualizer import MazeVisualizer

NUM_EVAL_GOALS = 1

def plot_maze(dir_path, max_path_length, num_rollouts=None, gap=1, max=None, ignore_done=False, stochastic=False, save_image=True):
    exps = load_exps_data(dir_path, gap=gap, max=max)
    eval_goals = None

    for exp in exps:
        vis = None
        if max_path_length is None:
            max_path_length = exp['json']['max_path_length']
        for pkl_path in exp['pkl']:
            data = joblib.load(pkl_path)

            if eval_goals is None:
                eval_goals = data['env'].sample_goals(num_rollouts)
            if vis is None:
                discount = exp['json']['discount']
                vis = MazeVisualizer(data['env'], eval_goals, max_path_length, discount, ignore_done, stochastic)

            vis.do_plots(policy=data['policy'], q_ensemble=data['Q_targets'], save_image=save_image, pkl_path=pkl_path)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument('--num_rollouts', '-n', type=int, default=1,
                        help='Max length of rollout')
    parser.add_argument('--gap_pkl', type=int, default=1,
                        help='Gap between pkl policies')
    parser.add_argument('--max_pkl', type=int, default=None,
                        help='Maximum value of the pkl policies')
    parser.add_argument("--max_path_length", "-l", type=int, default=None,
                        help="Max length or rollout")
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    parser.add_argument('--stochastic', action='store_true',
                        help='Apply stochastic action instead of deterministic')
    args = parser.parse_args()

    with tf.Session():
        plot_maze(dir_path=args.path, max_path_length=args.max_path_length, num_rollouts=args.num_rollouts,
                  gap=args.gap_pkl, max=args.max_pkl, ignore_done=args.ignore_done, stochastic=args.stochastic)



