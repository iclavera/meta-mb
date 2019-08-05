import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv
from collections import OrderedDict
import os


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle, MetaEnv):

    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'inverted_pendulum.xml', 2)

    def step(self, a):
        # reward = 1.0
        reward = self._get_reward()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reward(self):
        old_ob = self._get_obs()
        reward = -((old_ob[1]) ** 2)
        return reward

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        if next_obs is not None:
            assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        return -(obs[:, 1]) ** 2

    def tf_reward(self, obs, acts, next_obs):
        return - tf.square(obs[:, 1])

    def deriv_reward_obs(self, obs, acts):
        assert obs.ndim == acts.ndim
        if obs.ndim == 1:
            deriv = np.zeros_like(obs)
            deriv[1] = -2 * obs[1]
        elif obs.ndim == 2:
            deriv = np.zeros_like(obs)
            deriv[:, 1] = -2 * obs[:, 1]
        else:
            raise NotImplementedError
        return deriv

    def deriv_reward_act(self, obs, acts):
        return np.zeros_like(acts)

    def hessian_l_xx(self, obs, acts):
        hess = np.zeros((obs.shape[0], self.obs_dim, self.obs_dim))
        hess[:, 1, 1] = -2
        return -hess

    def hessian_l_uu(self, obs, acts):
        hess = np.zeros((obs.shape[0], self.act_dim, self.act_dim))
        return -hess

    def hessian_l_ux(self, obs, acts):
        hess = np.zeros((obs.shape[0], self.act_dim, self.obs_dim))
        return -hess

    def dl_dict(self, inputs_dict):
        # FOR NEGATIVE RETURNS
        obs, acts = inputs_dict['obs'], inputs_dict['act']
        return OrderedDict(l_x=-self.deriv_reward_obs(obs, acts),
                           l_u=-self.deriv_reward_act(obs, acts),
                           l_xx=self.hessian_l_xx(obs, acts),
                           l_uu=self.hessian_l_uu(obs, acts),
                           l_ux=self.hessian_l_ux(obs, acts),)

    def reset_from_obs(self, obs):
        self.set_state(obs[:self.model.nq], obs[self.model.nq:])
        return self._get_obs()


class InvertedPendulumSwingUpEnv(mujoco_env.MujocoEnv, utils.EzPickle, MetaEnv):

    def __init__(self):
        utils.EzPickle.__init__(self)

        dir_path = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/inverted_pendulum.xml' % dir_path, 2)

    def step(self, a):
        # reward = 1.0
        reward = self._get_reward()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # notdone = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        # done = not notdone
        done = False
        return ob, reward, done, {}

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)
        return self._get_obs()

    def reset_from_obs(self, obs):
        qpos, qvel = obs[:self.model.nq], obs[self.model.nq:]
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reward(self):
        old_ob = self._get_obs()
        reward = -((old_ob[1] - np.pi) ** 2)  # swing up
        return reward

    def _get_obs(self):
        return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        return -(obs[:, 1] - np.pi) ** 2

    def tf_reward(self, obs, acts, next_obs):
        return - tf.square(obs[:, 1] - np.pi)

    def deriv_reward_obs(self, obs, acts):
        assert obs.ndim == acts.ndim
        if obs.ndim == 1:
            deriv = np.zeros_like(obs)
            deriv[1] = -2 * (obs[1] - np.pi)
        elif obs.ndim == 2:
            deriv = np.zeros_like(obs)
            deriv[:, 1] = -2 * (obs[:, 1] - np.pi)
        else:
            raise NotImplementedError
        return deriv

    def deriv_reward_act(self, obs, acts):
        return np.zeros_like(acts)

    def goal_obs(self):
        obs = np.concatenate([self.init_qpos, self.init_qvel])
        obs += self.np_random.uniform(size=self.model.nq+self.model.nv, low=-0.01, high=0.01)
        obs[1] = np.pi
        return obs


if __name__ == "__main__":
    import pickle as pickle

    env = InvertedPendulumEnv()
    # for _ in range(5):
    #      ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
    # pickled_env_state = pickle.dumps(env.sim.get_state())
    #
    # env2 = InvertedPendulumEnv()
    # env2.sim.set_state(pickle.loads(pickled_env_state))
    #
    # print(f'compare env, env2 obs: {env._get_obs()}, {env2._get_obs()}')
    #
    # action = env.action_space.sample()
    # print(f'about to apply action {action}')
    #
    # print('env:')
    # print(env.step(action))
    #
    # print('env2:')
    # print(env2.step(action))
    #
    # env2.sim.forward()
    # print(env2._get_obs())

    env.reset()
    for _ in range(1000):
        _ = env.render()
        ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
