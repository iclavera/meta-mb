import os
import gym
import time
import rospy
import numpy as np
import tensorflow as tf
from meta_mb.logger import logger
from gym import error, spaces
from std_msgs.msg import Float64
from meta_mb.utils.serializable import Serializable
from meta_mb.envs.darwin.darwin_interface.darwin_interface import Darwin

def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))

class DarwinWalker(Darwin, Serializable):
    def __init__(self, ip='127.0.0.1', port=9090):
        Serializable.quick_init(self, locals())
        rospy.init_node("real_darwin")

        rospy.loginfo("Instantiating Darwin Client")
        rospy.sleep(1)

        rospy.loginfo("Darwin Walker Demo Starting")
        super(DarwinWalker, self).__init__()
        self.init_obs = np.array(
            [2.30036266e-08,  7.60837544e-07, -1.01350070e-04,  1.41789449e-03,
             6.97692317e-07, -1.53916334e-04,  2.63116532e-07,  1.69119796e-05,
             5.09295894e-04, -2.59973121e-04, -2.64801842e-04,  2.23206362e-04,
            -2.50477490e-04,  1.14467985e-04,  1.41967736e-03,  3.11767856e-08,
            -1.55366282e-04, -2.53245152e-07,  6.78718148e-06,  4.95561635e-04,
             2.47159634e-04,  2.56885893e-04, -1.97326791e-04, -3.21648182e-04,
             1.79345502e-03,  2.39432333e-03,  2.06551147e-04,  5.28752679e-03,
            -3.69303366e-05, -4.27761717e-05, -4.68474839e-04, -6.15110123e-04,
            -1.67463228e-06, -8.18625329e-06,  1.07773234e-05, -7.46292704e-05,
             1.30064536e-05, -3.91993730e-06,  1.77093710e-05,  3.18140419e-04,
             1.51865506e-04,  2.86533053e-04,  2.66065604e-04, -1.73811884e-03,
            -2.90223773e-04,  1.17746824e-06, -4.95102240e-04, -5.14221699e-04]
        )
        self.dt = 0.2
        self.init_qpos = self._joint_positions
        self.ordered_joints = [
            'j_pan',
            'j_tilt',
            'j_shoulder_l',
            'j_high_arm_l',
            'j_low_arm_l',
            'j_wrist_l',
            'j_gripper_l',
            'j_shoulder_r',
            'j_high_arm_r',
            'j_low_arm_r',
            'j_wrist_r',
            'j_gripper_r',
            'j_pelvis_l',
            'j_thigh1_l',
            'j_thigh2_l',
            'j_tibia_l',
            'j_ankle1_l',
            'j_ankle2_l'
            'j_pelvis_r',
            'j_thigh1_r',
            'j_thigh2_r',
            'j_tibia_r',
            'j_ankle1_r',
            'j_ankle2_r',
        ]

        self.obs_dim = 48
        self.act_dim = 24

        self.frame_skip = 2
        gym.utils.EzPickle.__init__(self)
        print(self.action_space)

    def _get_obs(self):
        joint_dict = self.get_angles()
        qpos = np.array([])
        for joint in self.ordered_joints:
            qpos = np.append(qpos, joint_dict[joint])
        return np.concatenate([qpos, self.get_joint_velocities()])

    def step(self, a):
        pos_before = self._get_obs()[:24]
        pos_after = self.do_simulation(a, self.frame_skip)[:24]

        alive_bonus = 5.0
        lin_vel_cost = 0.25 * (pos_after - pos_before)
        quad_ctrl_cost = 0.1 * np.square(a).sum()
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

    def reset(self):
        joint_dict = dict(zip(self.ordered_joints, self.init_qpos[:24]))
        self.set_angles_slow(joint_dict, 2)
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

    @property
    def action_space(self):
        return spaces.Box(low=np.random.uniform(-2, 2, self.act_dim),
                          high=np.random.uniform(-2, 2, self.act_dim),
                          dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)


if __name__ == "__main__":
    #rospy.init_node("real_darwin")

    #rospy.loginfo("Instantiating Darwin Client")
    #rospy.sleep(1)

    #rospy.loginfo("Darwin Walker Demo Starting")
    env = DarwinWalker()
    while True:
        env.reset()
        for _ in range(2000):
            _, reward, _, _ = env.step(np.random.uniform(-2, 2, 24) * 100)  # take a random action
            print(env._get_obs())


