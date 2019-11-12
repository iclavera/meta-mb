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

    def do_plots(self, policy, q_ensemble, base_title, plot_eval_returns):
        q_values = self._compute_q_values(policy, q_ensemble)
        if plot_eval_returns:
            self._do_plot_eval_returns(policy, title=f"{base_title}_eval_returns")

        self._do_plot_q_values(policy, q_values, title=f"{base_title}_q_values")
        return q_values[self.mask.ravel()]

    def plot_goal_distributions(self, agent_q_dict, sample_rule, alpha, itr):
        title = f"itr_{itr}_p_dist"
        image_path = os.path.join(self.parent_path, f"{title}.png")
        if not self.force_reload and os.path.exists(image_path):
            return

        print(f"plotting {title}")

        x = y = np.linspace(-self.pos_lim, self.pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)

        fig, ax_arr = plt.subplots(nrows=1, ncols=len(agent_q_dict.keys()), figsize=(40, 10))

        if sample_rule == 'softmax':
            agent_q_values = list(agent_q_dict.values())

            for agent_idx in agent_q_dict.keys():
                ax = ax_arr[int(agent_idx)]
                log_p = np.max(agent_q_values[:agent_idx] + agent_q_values[agent_idx+1:], axis=0)  # - agent_q
                p = np.exp(log_p - np.max(log_p))
                p /= np.sum(p)

                p = (1 - alpha) * p + alpha * self.u
                p_base = np.full((POINTS_PER_DIM, POINTS_PER_DIM), fill_value=np.nan)
                p_base[self.mask] = p
                cb = ax.scatter(xx, yy, c=p_base, s=10, marker='s', vmin=np.min(p), vmax=np.max(p))
                fig.colorbar(cb, shrink=0.5, ax=ax)

                ax.set_facecolor('black')
                ax.axis('equal')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

        elif sample_rule == 'norm_diff':
            max_q = np.max(list(agent_q_dict.values()), axis=0)

            for agent_idx, agent_q in agent_q_dict.items():
                ax = ax_arr[int(agent_idx)]
                p = max_q - agent_q
                if np.sum(p) == 0:
                    p = np.ones_like(p) / len(p)
                else:
                    p = p / np.sum(p)

                p = (1 - alpha) * p + alpha * self.u
                p_base = np.full((POINTS_PER_DIM, POINTS_PER_DIM), fill_value=np.nan)
                p_base[self.mask] = p
                cb = ax.scatter(xx, yy, c=p_base, s=10, marker='s', vmin=np.min(p), vmax=np.max(p))
                fig.colorbar(cb, shrink=0.5, ax=ax)

                ax.set_facecolor('black')
                ax.axis('equal')
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)

        else:
            raise ValueError

        plt.savefig(image_path)
        plt.clf()
        plt.close()

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
        feed_dict = {self.obs_ph: self.env.start_state, self.goal_ph: list(zip(xx.ravel(), yy.ravel()))}
        sess = tf.get_default_session()
        q_values, = sess.run([q_var,], feed_dict=feed_dict)

        return q_values

    def _do_plot_eval_returns(self, policy, title):
        print(f'plotting {title}')
        image_path = os.path.join(self.parent_path, f"{title}.png")
        if not self.force_reload and os.path.exists(image_path):
            return

        points_per_dim = POINTS_PER_DIM//5

        _, ax = plt.subplots(figsize=FIGSIZE)

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

                if discounted_return > 0:  # counted as success
                    total_success += 1
                    if goal_ind in _target_goals_ind_list:
                        target_success += 1

            z.append(discounted_return)

        total_success_rate = total_success / total_counter
        target_success_rate = target_success / target_counter

        z = np.reshape(z, (points_per_dim, points_per_dim))
        cb = plt.scatter(xx, yy, c=z, s=200, marker='s')
        plt.colorbar(cb, shrink=0.5)

        ax.set_facecolor('black')
        plt.title(f"total_hit_{round(total_success_rate, 2)}_target_hit_{round(target_success_rate, 2)}_{title}")
        plt.axis('equal')

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

        plt.savefig(image_path)
        plt.clf()
        plt.close()

    def _do_plot_q_values(self, policy, q_values, title):
        print(f'plotting {title}')
        image_path = os.path.join(self.parent_path, f"{title}.png")
        if not self.force_reload and os.path.exists(image_path):
            return

        grid_size = self.env.grid_size
        wall_size = 2 / grid_size

        _, ax = plt.subplots(figsize=FIGSIZE)

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
        cb = plt.scatter(xx, yy, c=q_values.reshape((POINTS_PER_DIM, POINTS_PER_DIM)), s=200, marker='s', cmap='winter')
        plt.colorbar(cb, shrink=0.5)

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

        plt.title(title)
        plt.axis('equal')

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.savefig(image_path)
        plt.clf()
        plt.close()

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
        policy.reset()
        obs = env.reset()
        env.set_goal(goal)
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
            path_length += 1
            if done and not self.ignore_done:
                break
            obs = next_obs

        return dict(observations=observations, discounted_return=discounted_return, dones=dones, rewards=rewards)
