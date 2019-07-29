import os
import gym
import time
import numpy as np
import tensorflow as tf
from meta_mb.logger import logger
from gym import error, spaces
from std_msgs.msg import Float64
from meta_mb.meta_envs.base import MetaEnv
from meta_mb.envs.darwin.darwin_interface.darwin_interface import Darwin

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class DarwinWalker(MetaEnv, Darwin, gym.utils.EzPickle):
    def __init__(self, ip='127.0.0.1', port=9090):
        super(DarwinWalker, self).__init__()
        self.ordered_joints = [
            'j_pan',
            'j_tilt',
            'j_shoulder_r',
            'j_high_arm_r',
            'j_low_arm_r',
            'j_wrist_r',
            'j_gripper_r',
            'j_pelvis_r',
            'j_thigh1_r',
            'j_thigh2_r',
            'j_tibia_r',
            'j_ankle1_r',
            'j_ankle2_r',
            'j_shoulder_l',
            'j_high_arm_l',
            'j_low_arm_l',
            'j_wrist_l',
            'j_gripper_l',
            'j_pelvis_l',
            'j_thigh1_l',
            'j_thigh2_l',
            'j_tibia_l',
            'j_ankle1_l',
            'j_ankle2_l'
        ]
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        joint_dict = self.get_angles()
        obs = np.array([])
        for joint in self.ordered_joints:
            obs = np.append(obs, joint_dict[joint])
        return obs

    def step(self, a):
        pos_before = mass_center(self.model, self.sim)[0]
        self.do_simulation(a, self.frame_skip)
        pos_after = mass_center(self.model, self.sim)[0]
        alive_bonus = 5.0
        data = self.sim.data
        lin_vel_cost = 0.25 * (pos_after - pos_before) / self.model.opt.timestep
        quad_ctrl_cost = 0.1 * np.square(data.ctrl).sum()
        reward = lin_vel_cost - quad_ctrl_cost + alive_bonus
        done = False
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost,
                                                   reward=reward, reward_alive=alive_bonus)
    def do_simulation(self, action, frame_skip):
        obs = self._get_obs()
        for _ in range(frame_skip):
            action_index = 0
            for joint in self.ordered_joints:
                if joint in ['j_wrist_l', 'j_gripper_l', 'j_wrist_r', 'j_gripper_r']:
                    msg = Float64()
                    msg.data = 0
                    self._pub_joints[joint].publish(msg)
                else:
                    msg = Float64()
                    msg.data = action[action_index]
                    self._pub_joints[joint].publish(msg)
        return obs

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            vel = obs_next[:, 22:25]
            lin_vel_reward = 0.25 * vel[:, 0]
            alive_bonus = 5.0
            ctrl_cost = .1 * np.sum(np.square(act), axis=1)
            reward = lin_vel_reward + alive_bonus - ctrl_cost
            return reward
        else:
            return self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))[0]

    def done(self, obs):
        if obs.ndim == 2:
            notdone = np.all(np.isfinite(obs), axis=1) * (obs[:, 0] >= 0.2) * (obs[:, 0] <= 1)
            return np.logical_not(notdone)
        else:
            notdone = np.isfinite(obs).all() and obs[0] >= 0.2 and obs[0] <= 1
            return not notdone

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv, )
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.6
        self.viewer.cam.elevation = 3
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.type = 0

    def log_diagnostics(self, paths, prefix=''):
        vel_cost = [-path["env_infos"]['reward_linvel'] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_quadctrl'] for path in paths]
        rwd = [-path["env_infos"]["reward"] for path in paths]

        logger.logkv(prefix + 'AvgDist', np.mean(rwd))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        logger.logkv(prefix + 'AvgVelCost', np.mean(vel_cost))


if __name__ == "__main__":
    env = DarwinWalker()
    while True:
        env.reset()
        for _ in range(2000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample() * 100)  # take a random action


