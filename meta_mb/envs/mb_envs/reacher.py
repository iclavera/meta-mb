import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv
from collections import OrderedDict


class ReacherEnv(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")

        if getattr(self, 'action_space', None):
            a = np.clip(a, self.action_space.low,
                        self.action_space.high)
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    # def reset_model(self):  # FIXME: hack for bptt
    #     self.set_state(self.init_qpos, self.init_qvel)
    #     return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),  # [0:2]
            np.sin(theta),  # [2:4]
            self.sim.data.qpos.flat[2:],  # [4:6]
            self.sim.data.qvel.flat[:2],  # [6:8]
            self.get_body_com("fingertip") - self.get_body_com("target")  # [8:11]
        ])

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        if next_obs is not None:
            assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        dist_vec = obs[:, -3:]
        reward_dist = - np.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - np.sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl
        return reward

    def deriv_reward_obs(self, obs, acts):
        assert obs.ndim == acts.ndim
        deriv = np.zeros_like(obs)
        if obs.ndim == 1:
            dist_vec = obs[-3:]
            reward_dist = - np.linalg.norm(dist_vec)
            deriv[-3:] /= reward_dist
        elif obs.ndim == 2:
            dist_vec = obs[:, -3:]
            reward_dist = - np.linalg.norm(dist_vec, axis=1, keepdims=True)
            deriv[:, -3:] /= reward_dist
        return deriv

    def deriv_reward_act(self, obs, acts):
        return -2 * acts #(-2 * acts).copy()

    # def l_xx(self, obs, act):
    #     """
    #
    #     :param obs: (obs_dim,)
    #     :param act: (act_dim,)
    #     :return: (obs_dim, obs_dim)
    #     """
    #     hess = np.zeros((self.obs_dim, self.obs_dim))
    #     reward_dist = - np.linalg.norm(obs[-3:])
    #     for i in range(-3, 0):
    #         for j in range(-3, 0):
    #             hess[i, j] = obs[i] * obs[j] / reward_dist
    #     for i in range(-3, 0):
    #         hess[i, i] += obs[i] / reward_dist
    #
    #     return hess
    #
    # def l_uu(self, obs, act):
    #     """
    #
    #     :param obs: (obs_dim,)
    #     :param act: (act_dim,)
    #     :return: (act_dim, act_dim)
    #     """
    #     return np.diag(np.ones_like((self.act_dim,)) * (-2))
    #
    # def l_ux(self, obs, act):
    #     """
    #
    #     :param obs: (obs_dim,)
    #     :param act: (act_dim,)
    #     :return: (obs_dim, act_dim)
    #     """
    #     return 0

    # def dl_dict(self, obs, act):
    #     return OrderedDict(l_x=self.deriv_reward_obs(obs, act),
    #                 l_u=self.deriv_reward_act(obs, act),
    #                 l_xx=self.l_xx(obs, act),
    #                 l_uu=self.l_uu(obs, act),
    #                 l_ux=self.l_ux(obs, act),)

    def hessian_l_xx(self, obses, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, obs_dim)
        """
        hess = np.zeros((obses.shape[0], self.obs_dim, self.obs_dim))
        reward_dist = - np.linalg.norm(obses[:, -3:], axis=1)  # (horizon,)
        for i in range(-3, 0):
            for j in range(-3, 0):
                hess[:, i, j] = obses[:, i] * obses[:, j] / reward_dist
        for i in range(-3, 0):
            hess[:, i, i] += obses[:, i] / reward_dist

        return -hess

    def hessian_l_uu(self, obses, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, act_dim, act_dim)
        """
        hess = np.zeros((obses.shape[0], self.act_dim, self.act_dim))
        for i in range(self.act_dim):
            hess[:, i, i] = -2
        return -hess

    def hessian_l_ux(self, obses, acts):
        """

        :param obs: (horizon, obs_dim,)
        :param act: (horizon, act_dim,)
        :return: (horizon, obs_dim, act_dim)
        """
        hess = np.zeros((obses.shape[0], self.act_dim, self.obs_dim))
        return -hess

    def dl_dict(self, inputs_dict):
        # FOR NEGATIVE RETURNS
        obs, act = inputs_dict['obs'], inputs_dict['act']
        return OrderedDict(l_x=-self.deriv_reward_obs(obs, act),
                           l_u=-self.deriv_reward_act(obs, act),
                           l_xx=self.hessian_l_xx(obs, act),
                           l_uu=self.hessian_l_uu(obs, act),
                           l_ux=self.hessian_l_ux(obs, act),)

    def reset_from_obs(self, obs):
        qpos, qvel = np.zeros((self.model.nq,)), np.zeros((self.model.nv,))
        qpos[:2] = np.arctan(obs[2:4]/obs[:2])
        qpos[2:] = obs[4:6]
        qvel[:2] = obs[-5:-3]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def tf_reward(self, obs, acts, next_obs):
        dist_vec = obs[:, -3:]
        reward_dist = - tf.linalg.norm(dist_vec, axis=1)
        reward_ctrl = - tf.reduce_sum(np.square(acts), axis=1)
        reward = reward_dist + reward_ctrl
        return reward


if __name__ == "__main__":
    env = ReacherEnv()
    env.reset()
    fail_ctr = 0
    for _ in range(100):
        x = np.random.random(size=(env.obs_dim,))
        new_x = env.reset_from_obs(x)
        if not np.allclose(x, new_x):
            fail_ctr += 1
            print(x, new_x)

    print(fail_ctr/100, ' percentage of failure')

    # for _ in range(1000):
    #     _ = env.render()
    #     ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
