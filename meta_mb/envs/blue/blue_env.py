import numpy as np
from meta_mb.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv
from meta_mb.meta_envs.base import RandomEnv

import os


class BlueReacherEnv(RandomEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.goal = np.ones((3,))
        gym.utils.EzPickle.__init__(self)
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),  "assets", "blue_right_v2.xml")
        gym.utils.EzPickle.__init__(self)
        MujocoEnv.__init__(self, model_path, 2)

        frame_skip = 2    
        RandomEnv.__init__(self, model_path=model_path, frame_skip=frame_skip)


    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec = self.get_body_com("right_l_finger_tip_link") - self.goal
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act), axis=1)
            reward_dist = -np.linalg.norm(obs_next[:,-3:], axis=1)
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e3, 1e3)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act))
            reward_dist = -np.linalg.norm(obs_next[-3:])
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e3, 1e3)
        else:
            raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos # + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=3)
            self.goal = np.array([-.3, -.3, 1])
            # self.goal = np.array([.73,  .6, 1.2]) # 4/23 seond exp, 

            if np.linalg.norm(self.goal) < 2:
                break
        qvel = self.init_qvel # + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_body_com("right_l_finger_tip_link") - self.goal
        ])
    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [-path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDist', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDist', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))


if __name__ == "__main__":
    env = BlueReacherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
