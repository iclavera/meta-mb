import numpy as np
from meta_mb.logger import logger
import gym
from gym import error, spaces
from meta_mb.meta_envs.base import MetaEnv
from blue_interface.blue_interface import BlueInterface
import os


class BlueReacherEnv(MetaEnv, BlueInterface, gym.utils.EzPickle):
    def __init__(self, side, ip, port=9090):
        self.goal = np.ones((3,))
        max_torques = np.array([15, 15, 15, 10, 10, 4, 4])
        self.frame_skip = 3
        super(BlueReacherEnv, self).__init__(side, ip, port)
        self.init_qpos = self.get_joint_positions()
        self.act_dim = len(self.get_joint_positions())
        self.obs_dim = len(self._get_obs())
        self._low, self._high = -max_torques, max_torques
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec = self.get_vec_gripper_to_goal()
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def get_vec_gripper_to_goal(self):
        gripper_pos = self.get_tip_position()
        vec_gripper_to_goal = self.goal - gripper_pos
        return vec_gripper_to_goal

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def do_simulation(self, action, frame_skip):
        action = np.clip(action, self._low, self._high)
        assert frame_skip > 0
        for _ in range(frame_skip):
            self.set_joint_torques(action)

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

    def reset(self):
        self.set_joint_positions(self.init_qpos)
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=3)
            self.goal = np.array([-.3, -.3, 1]) # TODO: Remove this line
            if np.linalg.norm(self.goal) < 2:
                break
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.get_joint_positions(),
            self.get_joint_velocities(),
            self.get_tip_position(),
            self.get_vec_gripper_to_goal(),
            ]).reshape(-1)

    def get_tip_position(self):
        pose = self.get_cartesian_pose()
        return pose['position']

    def log_diagnostics(self, paths, prefix=''):
        dist = [path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

if __name__ == "__main__":
    env = BlueReacherEnv()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
