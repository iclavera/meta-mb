import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import tensorflow as tf


POINTS_PER_DIM = 100
FIGSIZE = (7, 7)


class FetchEnvVisualizer(object):
    def __init__(self, env, eval_goals, max_path_length, discount,
                 ignore_done, stochastic):
        """

        :param env: meta_mb.envs.mb_envs.maze
        :param policy:
        :param q_ensemble: list
        :param eval_goals:
        """
        self.env = env
        self.eval_goals = eval_goals
        self.max_path_length = max_path_length
        self.discount = discount
        self.ignore_done = ignore_done
        self.stochastic = stochastic

        # utils variable for plotting heatmap
        self.target_center_2d_coords = target_center_2d_coords = np.asarray(env.initial_gripper_xpos[:3] + env.target_offset)[:2]
        if env.has_object:
            self.target_center_height = env.hight_offset
        else:
            self.target_center_height = (env.initial_gripper_xpos[:3] + env.target_offset)[2]
        self.pos_lim_low = target_center_2d_coords - env.target_range
        self.pos_lim_high = target_center_2d_coords + env.target_range

    def do_plots(self, fig, ax_arr, policy, q_functions, value_ensemble, itr):
        print(f"plotting itr_{itr}")

        x = np.linspace(self.pos_lim_low[0], self.pos_lim_high[0], num=POINTS_PER_DIM)
        y = np.linspace(self.pos_lim_low[1], self.pos_lim_high[1], num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)
        assert xx.shape == (POINTS_PER_DIM, POINTS_PER_DIM)
        input_goals = np.asarray(list(zip(xx.ravel(), yy.ravel(), np.ones(POINTS_PER_DIM*POINTS_PER_DIM) * self.target_center_height)))
        input_obs = np.tile(self.env.init_obs[np.newaxis, ...], (len(input_goals), 1))

        """------------- policy target q functions --------------"""

        actions, agent_infos = policy.get_actions(input_obs, input_goals)
        action_stds = np.exp([agent_info['log_std'] for agent_info in agent_infos])
        print(f'action_stds at itr {itr}', np.mean(action_stds), np.min(action_stds), np.max(action_stds))
        policy_values = [qfun.compute_values(input_obs, actions, input_goals) for qfun in q_functions]
        self._goal_distribution_helper(fig, ax_arr[0], np.mean(policy_values, axis=0).reshape((POINTS_PER_DIM, POINTS_PER_DIM)), f"policy_q_{itr}")

        """------------- value ensemble ------------------"""

        if value_ensemble:
            ensemble_values = [vfun.compute_values(input_obs, input_goals) for vfun in value_ensemble]
            self._goal_distribution_helper(fig, ax_arr[1], np.mean(ensemble_values, axis=0).reshape((POINTS_PER_DIM, POINTS_PER_DIM)), f"ensemble_values_{itr}")
            self._goal_distribution_helper(fig, ax_arr[2], np.var(ensemble_values, axis=0).reshape((POINTS_PER_DIM, POINTS_PER_DIM)), f"disagreement_{itr}")

        print(f"plotting eval returns...")
        self._do_plot_eval_returns(fig, ax_arr[3], policy)

    def _goal_distribution_helper(self, fig, ax, value, title):
        x = np.linspace(self.pos_lim_low[0], self.pos_lim_high[0], num=POINTS_PER_DIM)
        y = np.linspace(self.pos_lim_low[1], self.pos_lim_high[1], num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)

        cb = ax.scatter(xx, yy, c=value, s=0.8, marker='s', vmin=np.min(value), vmax=np.max(value))
        fig.colorbar(cb, shrink=0.5, ax=ax)

        ax.set_facecolor('black')
        ax.set_title(title)
        # ax.axis('equal')
        ax.set(xlim=(self.pos_lim_low[0], self.pos_lim_high[0]), ylim=(self.pos_lim_low[1], self.pos_lim_high[1]))

    def _do_plot_eval_returns(self, fig, ax, policy):
        points_per_dim = POINTS_PER_DIM//4

        """
        Evaluated discounted returns heatmap
        """

        x = np.linspace(self.pos_lim_low[0], self.pos_lim_high[0], num=points_per_dim)
        y = np.linspace(self.pos_lim_low[1], self.pos_lim_high[1], num=points_per_dim)
        xx, yy = np.meshgrid(x, y)
        z = []

        # performance metric
        total_counter, total_success = 0, 0

        for _x, _y in zip(xx.ravel(), yy.ravel()):
            total_counter += 1
            path = self._rollout(goal=np.asarray([_x, _y, self.target_center_height]), policy=policy)
            discounted_return = path["discounted_return"]

            if [_x, _y] in self.eval_goals.tolist():
                observations = path["observations"]
                dones = path["dones"]

                terminal_obs = observations[-1]
                for obs, next_obs, done in zip(observations[:-1], observations[1:], dones[:-1]):
                    if done:
                        terminal_obs = obs
                        break
                    else:
                        ax.add_line(Line2D([obs[0], next_obs[0]], [obs[1], next_obs[1]], color='red', alpha=0.2))

                ax.add_artist(plt.Circle((_x, _y), radius=0.05, lw=2, fill=False, color='darkorange', zorder=1000))
                ax.add_artist(plt.Circle(terminal_obs, radius=0.025, fill=True, color='darkorange', zorder=100000))

            if path['undiscounted_return'] > - len(path['rewards']):  # counted as success
                total_success += 1

            z.append(discounted_return)

        total_success_rate = total_success / total_counter

        z = np.reshape(z, (points_per_dim, points_per_dim))
        cb = ax.scatter(xx, yy, c=z, s=50, marker='s')
        fig.colorbar(cb, shrink=0.5, ax=ax)

        ax.set_facecolor('black')
        ax.set_title(f"total_hit_{round(total_success_rate, 2)}")
        # ax.axis('equal')
        ax.set(xlim=(self.pos_lim_low[0], self.pos_lim_high[0]), ylim=(self.pos_lim_low[1], self.pos_lim_high[1]))

    # def _make_gif(self, goal, observations, dones):
    #     for t in range(len(observations)):
    #         # generate plots
    #         self._do_plot(goal, observations[:t+1], dones[:t+1], save_image=True)

        # plt.title('Ensemble stds on grid (time %d)' % t)
        # plt.axis('equal')

        # if not os.path.isdir(dir_name):
        #     os.makedirs(dir_name)
        # path_name = dir_name + '/grid_var_' + str(t) + '.png'
        # plt.savefig(path_name)
        # print('Saved figure to', path_name)
        # plt.clf()
        # plt.close()

    def _rollout(self, goal, policy):
        env = self.env
        max_path_length = self.max_path_length
        discount = self.discount

        observations, rewards, dones = [], [], []
        discounted_return = 0
        undiscounted_return = 0
        policy.reset()
        obs = env.reset_obs()['observation']
        _ = env.reset_goal(goal)
        path_length = 0

        while path_length < max_path_length:
            action, agent_info = policy.get_action(obs, goal)
            if not self.stochastic:
                action = agent_info['mean']
            next_obs, reward, done, _ = env.step(action)
            next_obs = next_obs['observation']

            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            discounted_return += discount**path_length * reward
            undiscounted_return += reward
            path_length += 1
            if done and not self.ignore_done:
                break
            obs = next_obs

        return dict(observations=observations, discounted_return=discounted_return,
                    undiscounted_return=undiscounted_return, dones=dones, rewards=rewards)
