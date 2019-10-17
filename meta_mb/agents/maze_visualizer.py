import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import moviepy.editor as mpy
import numpy as np
import tensorflow as tf


POINTS_PER_DIM = 100


class MazeVisualizer(object):
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

        self.goal_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.env.goal_dim), name='goal')
        self.obs_ph = tf.placeholder(dtype=tf.float32, shape=(self.env.obs_dim,), name='obs')

    def do_plots(self, policy, q_ensemble, save_image=True, pkl_path=None):
        min_q_var = self._build(policy, q_ensemble)
        for goal_idx, goal in enumerate(self.eval_goals):
            # fig, ax_arr = plt.subplots(nrows=1, ncols=2, figsize=(24, 12))
            # ax_arr = [None, None]
            self._do_plot_q_values(goal, goal_idx, policy, min_q_var, save_image=save_image, pkl_path=pkl_path)
            self._do_plot_eval_returns(policy, save_image=save_image, pkl_path=pkl_path)
            # if save_image:
            #     image_path = pkl_path.replace(".pkl", f"_{goal_idx}.png")
            #     plt.savefig(image_path)

    def _build(self, policy, q_ensemble):
        obs_no = tf.tile(self.obs_ph[None], (tf.shape(self.goal_ph)[0], 1))
        goal_ng = self.goal_ph
        dist_info_sym = policy.distribution_info_sym(tf.concat([obs_no, goal_ng], axis=1))
        act_na, _ = policy.distribution.sample_sym(dist_info_sym)
        input_q_fun = tf.concat([obs_no, act_na, goal_ng], axis=1)

        q_vals = tf.stack([tf.reshape(q.value_sym(input_var=input_q_fun), (-1,)) for q in q_ensemble], axis=0)
        return tf.reduce_min(q_vals, axis=0)

    def _compute_min_q(self, obs, goals, min_q_var):
        feed_dict = {self.goal_ph: goals, self.obs_ph: obs}
        sess = tf.get_default_session()
        min_q, = sess.run([min_q_var], feed_dict=feed_dict)
        return min_q

    def _do_plot_eval_returns(self, policy, save_image=True, pkl_path=None, ax=None):
        grid_size = self.env.grid_size
        points_per_dim = POINTS_PER_DIM//10
        wall_size = 2 / grid_size

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(self.env.grid, extent=(-1, 1, -1, 1))
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)

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
        x = np.linspace(-.8, .8, num=points_per_dim)
        y = np.linspace(-.8, .8, num=points_per_dim)
        xx, yy = np.meshgrid(x, y)
        z = []
        for _x, _y in zip(xx.ravel(), yy.ravel()):
            goal = np.asarray([_x, _y])
            if self.env._is_wall(goal):
                z.append(None)
            else:
                path = self._rollout(goal=np.asarray([_x, _y]), policy=policy)
                discounted_return = path["discounted_return"]
                z.append(discounted_return)
        z = np.asarray(z).reshape((points_per_dim, points_per_dim))

        cb = plt.scatter(xx, yy, c=z, s=500, marker='s', cmap='plasma')
        plt.colorbar(cb, shrink=0.5)

        plt.title(os.path.join(*pkl_path.split('/')[-4:]))

        if save_image:
            image_path = pkl_path.replace(".pkl", "_eval_returns.png")
            plt.savefig(image_path)
            plt.clf()
            plt.close()

    def _do_plot_q_values(self, goal, goal_idx, policy, min_q_var, save_image=True, pkl_path=None, ax=None):
        path = self._rollout(goal, policy)
        observations = path["observations"]
        dones = path["dones"]

        grid_size = self.env.grid_size
        points_per_dim = POINTS_PER_DIM
        wall_size = 2 / grid_size

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.imshow(self.env.grid, extent=(-1, 1, -1, 1))
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)

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
        x = np.linspace(-.8, .8, num=points_per_dim)
        y = np.linspace(-.8, .8, num=points_per_dim)
        xx, yy = np.meshgrid(x, y)
        z = self._compute_min_q(self.env.start_state, list(zip(xx.ravel(), yy.ravel())), min_q_var).reshape((points_per_dim, points_per_dim))

        cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='plasma')
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
                ax.add_line(Line2D([obs[0], next_obs[0]], [obs[1], next_obs[1]], color='cyan', alpha=0.2))

        ax.add_artist(plt.Circle(goal, radius=0.05, lw=2, fill=False, color='lime', zorder=1000))
        ax.add_artist(plt.Circle(terminal_obs, radius=0.025, fill=True, color='lime', zorder=100000))

        plt.title(os.path.join(*pkl_path.split('/')[-4:]))

        if save_image:
            image_path = pkl_path.replace(".pkl", f"_{goal_idx}.png")
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
