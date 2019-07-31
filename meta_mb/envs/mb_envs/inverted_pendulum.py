
import numpy as np
import tensorflow as tf
from gym import utils
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import MetaEnv
import os


class InvertedPendulumEnv(mujoco_env.MujocoEnv, utils.EzPickle, MetaEnv):

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
        deriv = np.zeros_like(obs)
        deriv[:, 1] = -2 * (obs[:, 1] - np.pi)
        return deriv

    def deriv_reward_acts(self, obs, acts):
        return np.zeros_like(acts)

if __name__ == "__main__":
    import pickle as pickle

    env = InvertedPendulumEnv()
    for _ in range(5):
         ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
    pickled_env_state = pickle.dumps(env.sim.get_state())

    env2 = InvertedPendulumEnv()
    env2.sim.set_state(pickle.loads(pickled_env_state))

    print(f'compare env, env2 obs: {env._get_obs()}, {env2._get_obs()}')

    action = env.action_space.sample()
    print(f'about to apply action {action}')

    print('env:')
    print(env.step(action))

    print('env2:')
    print(env2.step(action))

    env2.sim.forward()
    print(env2._get_obs())

    # env.reset()
    # for _ in range(1000):
    #     _ = env.render()
    #     ob, rew, done, info = env.step(env.action_space.sample())  # take a random action
