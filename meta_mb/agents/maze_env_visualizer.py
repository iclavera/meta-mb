import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


POINTS_PER_DIM = 100
FIGSIZE = (7, 7)


class MazeEnvVisualizer(object):
    def __init__(self, env, eval_goals, max_path_length, discount,
                 ignore_done, stochastic):
        """

        :param env: meta_mb.envs.mb_envs.maze
        :param policy:
        :param q_ensemble: list
        :param eval_goals:
        """
        self.env = env
        self.size = self.env.grid_size
        self.eval_goals = eval_goals
        self.max_path_length = max_path_length
        self.discount = discount
        self.ignore_done = ignore_done
        self.stochastic = stochastic

        # utils variable for plotting heatmap
        self.pos_lim = pos_lim = 1 - 2 / self.env.grid_size
        x = y = np.linspace(-pos_lim, pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)
        sweeping_indices = self.env._get_index(list(zip(xx.ravel(), yy.ravel())))

        _free_ind_list = self.env._free_ind.tolist()
        self.mask = np.reshape(list(map(
            lambda ind: ind.tolist() in _free_ind_list,
            sweeping_indices,
        )), (POINTS_PER_DIM, POINTS_PER_DIM))

    def do_plots(self, fig, ax_arr, policy, q_functions, value_ensemble, goal_samples, itr):
        print(f"plotting itr_{itr}")

        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)
        input_goals = np.asarray(list(zip(xx.ravel(), yy.ravel())))
        input_obs = np.tile(self.env.init_obs[np.newaxis, ...], (POINTS_PER_DIM*POINTS_PER_DIM, 1))

        """------------- policy target q functions --------------"""

        actions, agent_infos = policy.get_actions(input_obs, input_goals)
        action_stds = np.exp([agent_info['log_std'] for agent_info in agent_infos])
        print(f'stats for policy std', np.mean(action_stds), np.min(action_stds), np.max(action_stds))
        policy_values = [qfun.compute_values(input_obs, actions, input_goals) for qfun in q_functions]
        self._goal_distribution_helper(fig, ax_arr[0], np.mean(policy_values, axis=0).reshape((POINTS_PER_DIM, POINTS_PER_DIM)), f"policy_q")

        """------------- value ensemble ------------------"""

        if value_ensemble:
            ensemble_values = [vfun.compute_values(input_obs, input_goals) for vfun in value_ensemble]
            self._goal_distribution_helper(fig, ax_arr[1], np.mean(ensemble_values, axis=0).reshape((POINTS_PER_DIM, POINTS_PER_DIM)), f"ensemble_values")
            self._goal_distribution_helper(fig, ax_arr[2], np.var(ensemble_values, axis=0).reshape((POINTS_PER_DIM, POINTS_PER_DIM)), f"disagreement")

        print(f"plotting eval returns...")
        self._do_plot_eval_returns(fig, ax_arr[3], policy)
        self._do_plot_traj(fig, ax_arr[4], policy)

        self._goal_samples_helper(fig, ax_arr[5], goal_samples, f"goal_samples")

    def _goal_samples_helper(self, fig, ax, goal_samples, title):
        grid_size = self.env.grid_size
        wall_size = 2 / grid_size
        for i in range(grid_size):
            for j in range(grid_size):
                if self.env._block_mask[i, j]:
                    ax.add_artist(plt.Rectangle(self.env._get_coords(np.asarray([i, j])) - wall_size/2,
                                                width=wall_size, height=wall_size, fill=True, color='black'))

        for goal in goal_samples:
            ax.add_artist(plt.Circle(goal, radius=0.02, fill=True, color='darkorange', zorder=100000))

        ax.set_title(title)
        ax.axis('equal')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def _goal_distribution_helper(self, fig, ax, values, title):
        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)

        # value_base = np.full((POINTS_PER_DIM, POINTS_PER_DIM), fill_value=np.nan)
        # value_base[self.mask] = values[self.mask].flatten()
        cb = ax.scatter(xx, yy, c=values, s=0.8, marker='s', vmin=np.min(values), vmax=np.max(values))
        fig.colorbar(cb, shrink=0.5, ax=ax)

        ax.set_facecolor('black')
        ax.set_title(title)
        ax.axis('equal')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def _do_plot_eval_returns(self, fig, ax, policy):
        points_per_dim = POINTS_PER_DIM//4

        """
        Evaluated discounted returns heatmap
        """
        _free_ind_list = self.env._free_ind.tolist()

        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=points_per_dim)
        xx, yy = np.meshgrid(x, y)
        z = []

        # performance metric
        total_counter, total_success = 0, 0

        for _x, _y in zip(xx.ravel(), yy.ravel()):
            # discounted_return = None
            goal = np.asarray([_x, _y])
            # goal_ind = self.env._get_index(goal).tolist()
            # if goal_ind in _free_ind_list:
            total_counter += 1
            # if goal_ind in _target_goals_ind_list:
            #     target_counter += 1
            path = self._rollout(goal=goal, policy=policy)
            discounted_return = path["discounted_return"]

            if path['undiscounted_return'] > - len(path['rewards']):  # counted as success
                total_success += 1
                # if goal_ind in _target_goals_ind_list:
                #     target_success += 1

            z.append(discounted_return)

        total_success_rate = total_success / total_counter

        z = np.reshape(z, (points_per_dim, points_per_dim))
        cb = ax.scatter(xx, yy, c=z, s=50, marker='s')
        fig.colorbar(cb, shrink=0.5, ax=ax)

        ax.set_facecolor('black')
        ax.set_title(f"total_hit_{round(total_success_rate, 2)}")
        ax.axis('equal')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def _do_plot_traj(self, fig, ax, policy):
        grid_size = self.env.grid_size
        wall_size = 2 / grid_size

        """
        Wall
        """
        for i in range(grid_size):
            for j in range(grid_size):
                if self.env._block_mask[i, j]:
                    ax.add_artist(plt.Rectangle(self.env._get_coords(np.asarray([i, j])) - wall_size/2,
                                                width=wall_size, height=wall_size, fill=True, color='black'))

        # """
        # Value function heatmap
        # """
        # x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        # xx, yy = np.meshgrid(x, y)
        # cb = ax.scatter(xx, yy, c=q_values.reshape((POINTS_PER_DIM, POINTS_PER_DIM)), s=40, marker='s', cmap='winter')
        # fig.colorbar(cb, shrink=0.5, ax=ax)

        """
        Trajectory 
        """
        for goal in self.eval_goals:
            path = self._rollout(goal, policy)
            observations = path["observations"]
            dones = path["dones"]

            terminal_obs = observations[-1]
            for obs, next_obs, done in zip(observations[:-1], observations[1:], dones[:-1]):
                if done:
                    terminal_obs = obs
                    break
                else:
                    ax.add_line(Line2D([obs[0], next_obs[0]], [obs[1], next_obs[1]], color='red', alpha=0.2))

            ax.add_artist(plt.Circle(goal, radius=0.05, lw=2, fill=False, color='darkorange', zorder=1000))
            ax.add_artist(plt.Circle(terminal_obs, radius=0.025, fill=True, color='darkorange', zorder=100000))

        ax.set_facecolor('white')
        ax.axis('equal')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

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
        obs = env.reset_obs()
        _ = env.reset_goal(goal)
        path_length = 0

        while path_length < max_path_length:
            action, agent_info = policy.get_action(obs, goal)
            if not self.stochastic:
                action = agent_info['mean']
            next_obs, reward, done, _ = env.step(action)
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
