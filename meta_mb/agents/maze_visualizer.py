import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import tensorflow as tf


POINTS_PER_DIM = 100
FIGSIZE = (7, 7)


class MazeVisualizer(object):
    def __init__(self, env, eval_goals, max_path_length, discount,
                 ignore_done, stochastic, parent_path, force_reload):
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
        self.parent_path = parent_path
        self.force_reload = force_reload

        self.goal_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.goal_dim), name='goal')
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.env.obs_dim,), name='obs')

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

        _target_goals_ind_list = self.env._target_goals_ind.tolist()
        target_mask = np.asarray(list(map(
            lambda ind: ind.tolist() in _target_goals_ind_list,
            sweeping_indices[self.mask.ravel()],
        )))  # length = number of free sweeping index pairs
        self.u = target_mask.ravel().astype(np.int) / np.sum(target_mask)

    def do_plots(self, policy_arr, q_ensemble_arr, itr):
        image_path = os.path.join(self.parent_path, f"itr_{itr}.png")
        if not self.force_reload and os.path.exists(image_path):
            return None

        print(f"plotting itr_{itr}")

        q_values_arr = []
        num_agents = len(policy_arr)
        fig, ax_arr = plt.subplots(nrows=2, ncols=num_agents, figsize=(20, 10))

        for agent_idx in range(num_agents):
            q_values = self._compute_q_values(policy_arr[agent_idx], q_ensemble_arr[agent_idx])
            q_values_arr.append(q_values[self.mask.ravel()])

            self._do_plot_eval_returns(fig, ax_arr[0, agent_idx], policy_arr[agent_idx])
            self._do_plot_q_values(fig, ax_arr[1, agent_idx], policy_arr[agent_idx], q_values)

        plt.savefig(image_path)
        plt.clf()
        plt.close()

        return q_values_arr

    def plot_goal_distributions(self, q_values_arr, sample_rule, alpha, itr):
        if not 0 < alpha < 1:
            return
        image_path = os.path.join(self.parent_path, f"itr_{itr}_p_dist.png")
        if not self.force_reload and os.path.exists(image_path):
            return

        print(f"plotting itr_{itr}_p_dist")

        num_agents = len(q_values_arr)
        max_q, min_q = np.max(q_values_arr, axis=0), np.min(q_values_arr, axis=0)

        fig, ax_arr = plt.subplots(nrows=1, ncols=num_agents+1, figsize=(20, 5))

        if sample_rule == 'softmax':
            for agent_idx in range(num_agents):
                log_p = np.max(q_values_arr[:agent_idx] + q_values_arr[agent_idx+1:], axis=0)  # - agent_q
                p = np.exp(log_p - np.max(log_p))
                p /= np.sum(p)

                p = (1 - alpha) * p + alpha * self.u
                self._goal_distribution_helper(fig, ax_arr[agent_idx], p)

        elif sample_rule == 'norm_diff':
            for agent_idx in range(num_agents):
                p = max_q - q_values_arr[agent_idx]
                if np.sum(p) == 0:
                    p = np.ones_like(p) / len(p)
                else:
                    p = p / np.sum(p)

                p = (1 - alpha) * p + alpha * self.u
                self._goal_distribution_helper(fig, ax_arr[agent_idx], p)

        else:
            raise ValueError

        # plot curiosity
        self._goal_distribution_helper(fig, ax_arr[-1], max_q - min_q)

        fig.suptitle(f"itr_{itr}_p_dist")
        plt.savefig(image_path)
        plt.clf()
        plt.close()

    def _goal_distribution_helper(self, fig, ax, value):
        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)

        value_base = np.full((POINTS_PER_DIM, POINTS_PER_DIM), fill_value=np.nan)
        value_base[self.mask] = value
        cb = ax.scatter(xx, yy, c=value_base, s=10, marker='s', vmin=np.min(value), vmax=np.max(value))
        fig.colorbar(cb, shrink=0.5, ax=ax)

        ax.set_facecolor('black')
        ax.axis('equal')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def _compute_q_values(self, policy, q_ensemble):
        """
        compute q values with grid goals
        :param policy:
        :param q_ensemble:
        :return:
        """
        """---------build graph--------"""

        obs_no = tf.tile(self.obs_ph[None], (tf.shape(self.goal_ph)[0], 1))
        goal_ng = self.goal_ph
        dist_info_sym = policy.distribution_info_sym(tf.concat([obs_no, goal_ng], axis=1))
        act_na, _ = policy.distribution.sample_sym(dist_info_sym)
        input_q_fun = tf.concat([obs_no, act_na, goal_ng], axis=1)

        q_var = tf.stack([tf.reshape(q.value_sym(input_var=input_q_fun), (-1,)) for q in q_ensemble], axis=0)
        q_var = tf.reduce_min(q_var, axis=0)

        """---------run graph to compute q values------"""

        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)
        feed_dict = {self.obs_ph: self.env.init_obs, self.goal_ph: list(zip(xx.ravel(), yy.ravel()))}
        sess = tf.get_default_session()
        q_values, = sess.run([q_var,], feed_dict=feed_dict)

        return q_values

    def _do_plot_eval_returns(self, fig, ax, policy):
        points_per_dim = POINTS_PER_DIM//4

        """
        Evaluated discounted returns heatmap
        """
        _free_ind_list = self.env._free_ind.tolist()
        _target_goals_ind_list = self.env._target_goals_ind.tolist()

        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=points_per_dim)
        xx, yy = np.meshgrid(x, y)
        z = []

        # performance metric
        total_counter, total_success = 0, 0
        target_counter, target_success = 0, 0

        for _x, _y in zip(xx.ravel(), yy.ravel()):
            discounted_return = None
            goal = np.asarray([_x, _y])
            goal_ind = self.env._get_index(goal).tolist()
            if goal_ind in _free_ind_list:
                total_counter += 1
                if goal_ind in _target_goals_ind_list:
                    target_counter += 1
                path = self._rollout(goal=goal, policy=policy)
                discounted_return = path["discounted_return"]

                if path['undiscounted_return'] > - len(path['rewards']):  # counted as success
                    total_success += 1
                    if goal_ind in _target_goals_ind_list:
                        target_success += 1

            z.append(discounted_return)

        total_success_rate = total_success / total_counter
        target_success_rate = target_success / target_counter

        z = np.reshape(z, (points_per_dim, points_per_dim))
        cb = ax.scatter(xx, yy, c=z, s=100, marker='s')
        fig.colorbar(cb, shrink=0.5, ax=ax)

        ax.set_facecolor('black')
        ax.set_title(f"total_hit_{round(total_success_rate, 2)}_target_hit_{round(target_success_rate, 2)}")
        ax.axis('equal')
        ax.set(xlim=(-1, 1), ylim=(-1, 1))

    def _do_plot_q_values(self, fig, ax, policy, q_values):
        grid_size = self.env.grid_size
        wall_size = 2 / grid_size

        """
        Wall
        """
        for i in range(grid_size):
            for j in range(grid_size):
                if self.env.grid[i, j]:
                    ax.add_artist(plt.Rectangle(self.env._get_coords(np.asarray([i, j])) - wall_size/2,
                                                width=wall_size, height=wall_size, fill=True, color='black'))

        """
        Value function heatmap
        """
        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)
        cb = ax.scatter(xx, yy, c=q_values.reshape((POINTS_PER_DIM, POINTS_PER_DIM)), s=40, marker='s', cmap='winter')
        fig.colorbar(cb, shrink=0.5, ax=ax)

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

        ax.set_facecolor('black')
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
        obs = env.reset(goal)
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
