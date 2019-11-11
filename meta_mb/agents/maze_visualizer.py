import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np
import tensorflow as tf


POINTS_PER_DIM = 100
FIGSIZE = (7, 7)


class MazeVisualizer(object):
    def __init__(self, env, eval_goals, max_path_length, discount,
                 ignore_done, stochastic, parent_path):
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

        self.goal_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.goal_dim), name='goal')
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.env.obs_dim,), name='obs')

        # utils variable for plotting heatmap
        pos_lim = 1 - 2 / self.env.grid_size
        x = np.linspace(-pos_lim, pos_lim, num=POINTS_PER_DIM)
        y = np.linspace(-pos_lim, pos_lim, num=POINTS_PER_DIM)
        xx, yy = np.meshgrid(x, y)
        self.xx, self.yy = xx, yy
        self.mask = list(map(
            lambda _x, _y: self.env._get_index(_x, _y) in self.env._train_goals_ind,
            zip(xx.ravel(), yy.ravel())
        ))
        # mask = np.reshape(np.asarray(mask), (POINTS_PER_DIM, POINTS_PER_DIM))

    def do_plots(self, policy, q_ensemble, base_title):
        q_values = self._compute_q_values(policy, q_ensemble)
        self._do_plot_eval_returns(policy, title=f"{base_title}_eval_returns")

        for goal_idx, goal in enumerate(self.eval_goals):
            # fig, ax_arr = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
            # ax_arr = [None, None]
            self._do_plot_q_values(goal, policy, q_values, title=f"{base_title}_goal_{goal_idx}_q_values")
            # if save_image:
            #     image_path = pkl_path.replace(".pkl", f"_{goal_idx}.png")
            #     plt.savefig(image_path)

        q_values_train_goals = q_values[self.mask]

        return q_values_train_goals

    def plot_goal_distributions(self, agent_q_dict, sample_rule, eps, itr):
        u = None


        if sample_rule == 'softmax':
            agent_q_values = list(agent_q_dict.values())
            for agent_idx in agent_q_dict.keys():
                log_p = np.max(agent_q_values[:agent_idx] + agent_q_values[agent_idx+1:], axis=0)  # - agent_q
                p = np.exp(log_p - np.max(log_p))
                p /= np.sum(p)

                if u is None:
                    u = np.ones(len(p)) / len(p)

                p = eps * u + (1 - eps) * p
                self._do_plot_p_dist(p, title=f'itr_{itr}_agent_{agent_idx}_p_dist')

        elif sample_rule == 'norm_diff':
            max_q = np.max(list(agent_q_dict.values()), axis=0)
            for agent_idx, agent_q in agent_q_dict.items():
                p = max_q - agent_q
                if u is None:
                    u = np.ones(len(p)) / len(p)
                if np.sum(p) > 1e-3:
                    p /= np.sum(p)
                else:
                    p = u

                p = eps * u + (1 - eps) * p
                self._do_plot_p_dist(p, title=f'itr_{itr}_agent_{agent_idx}_p_dist')

        else:
            raise ValueError

    def _do_plot_p_dist(self, p, title):
        print(f'plotting {title}')

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
        p_base = np.zeros((POINTS_PER_DIM, POINTS_PER_DIM))
        p_base[np.where(self.mask)] = p
        cb = plt.scatter(self.xx, self.yy, c=p_base, s=500, marker='s', cmap='winter')
        plt.colorbar(cb, shrink=0.5)

        plt.title(title)
        plt.axis('equal')

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.savefig(os.path.join(self.parent_path, f"{title}.png"))
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

        feed_dict = {self.obs_ph: self.env.start_state, self.goal_ph: list(zip(self.xx.ravel(), self.yy.ravel()))}
        sess = tf.get_default_session()
        q_values, = sess.run([q_var,], feed_dict=feed_dict)

        return q_values

    def _do_plot_eval_returns(self, policy, title):
        print(f'plotting {title}')
        image_path = os.path.join(self.parent_path, f"{title}.png")
        if os.path.exists(image_path):
            return

        grid_size = self.env.grid_size
        points_per_dim = POINTS_PER_DIM//4
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
        Evaluated discounted returns heatmap
        """
        z = []
        for _x, _y in zip(self.xx.ravel(), self.yy.ravel()):
            goal = np.asarray([_x, _y])
            if self.env._is_wall(goal):
                z.append(None)
            else:
                path = self._rollout(goal=np.asarray([_x, _y]), policy=policy)
                discounted_return = path["discounted_return"]
                z.append(discounted_return)
        z = np.reshape(np.asarray(z), (points_per_dim, points_per_dim))

        cb = plt.scatter(self.xx, self.yy, c=z, s=500, marker='s', cmap='winter')
        plt.colorbar(cb, shrink=0.5)

        plt.title(title)

        plt.axis('equal')

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

        plt.savefig(image_path)
        plt.clf()
        plt.close()

    def _do_plot_q_values(self, goal, policy, q_values, title):
        print(f'plotting {title}')
        image_path = os.path.join(self.parent_path, f"{title}.png")
        # if os.path.exists(image_path):
        #     return
        path = self._rollout(goal, policy)
        observations = path["observations"]
        dones = path["dones"]

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
        cb = plt.scatter(self.xx, self.yy, c=q_values.reshape((POINTS_PER_DIM, POINTS_PER_DIM)), s=200, marker='s', cmap='winter')
        plt.colorbar(cb, shrink=0.5)

        """
        Trajectory 
        """
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

    def _make_gif(self, goal, observations, dones):
        for t in range(len(observations)):
            # generate plots
            self._do_plot(goal, observations[:t+1], dones[:t+1], save_image=True)

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
