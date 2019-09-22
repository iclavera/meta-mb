import os
import tensorflow as tf
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv


class AntEnv(MetaEnv, mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, frame_skip=5):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(
            self, '%s/assets/ant.xml' % dir_path, frame_skip=frame_skip
        )
        utils.EzPickle.__init__(self)

        self.obs_dim = self.observation_space.shape[0]
        self.act_dim = self.action_space.shape[0]

    def step(self, action):
        old_ob = self._get_obs()
        self.do_simulation(action, self.frame_skip)

        if getattr(self, 'action_space', None):
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)
        ob = self._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = old_ob[13]
        reward_height = -3.0 * np.square(old_ob[0] - 0.57)
        reward = reward_run + reward_ctrl + reward_height + 1.0
        done = False
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # (self.model.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            # self.get_body_comvel("torso")[:1],
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        # self.prev_qpos = np.copy(self.model.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        reward_ctrl = -0.1 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 13]
        reward_height = -3.0 * np.square(obs[:, 0] - 0.57)
        reward = reward_run + reward_ctrl + reward_height + 1.0
        return reward

    def tf_reward(self, obs, acts, next_obs):
        if obs.get_shape().ndims == 2:
            reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts), axis=1)
            reward_run = next_obs[:, 13]
            reward_height = -3.0 * tf.square(next_obs[:, 0] - 0.57)
        else:
            reward_ctrl = -0.1 * tf.reduce_sum(tf.square(acts))
            reward_run = obs[13]
            reward_height = -3.0 * tf.square(next_obs[0] - 0.57)

        reward = reward_run + reward_ctrl + reward_height + 1.0
        return reward

    def tf_dl(self, obs, act, next_obs):
        assert obs.get_shape().ndims == 1

        # l_x
        e_13 = np.zeros((self.obs_dim,))
        e_13[13] = 1
        e_0 = np.zeros((self.obs_dim,))
        e_0[0] = 1
        r_x = e_13 - 6 * (obs[0] - 0.57) * e_0
        l_x = -r_x

        # l_u
        r_u = -0.2 * act
        l_u = -r_u

        # l_xx
        r_xx = np.zeros((self.obs_dim, self.obs_dim))
        r_xx[0][0] = -6
        l_xx = -r_xx

        # l_ux
        l_ux = tf.zeros((self.act_dim, self.obs_dim))

        # l_uu
        r_uu = -0.2 * np.eye(self.act_dim)
        l_uu = -r_uu

        return l_x, l_u, l_xx, l_uu, l_ux


if __name__ == "__main__":
    env = AntEnv()
    env.reset()
    print(env.sim.derivative().shape)
    # for _ in range(1000):
    #     _ = env.render()
    #     ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
