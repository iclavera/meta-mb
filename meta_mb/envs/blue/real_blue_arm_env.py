import numpy as np
import tensorflow as tf
import time
from meta_mb.logger import logger
import gym
from gym import error, spaces
from meta_mb.meta_envs.base import MetaEnv

from blue_interface.blue_interface import BlueInterface

class ArmReacherEnv(MetaEnv, BlueInterface, gym.utils.EzPickle):
    def __init__(self,
                 max_torques=[2]*7,
                 exp_type='reach',
                 ctrl_penalty=1.25e-1,
                 vel_penalty=1.25e-2,
                 fixed=False,
                 side='right',
                 ip='169.229.223.208',
                 port=9090):
        self.fixed = True
        if exp_type == 'reach':
            self.init_qpos = np.array([4.64195583e-03,  8.63199317e-03, -3.53298290e-04, -1.20736697e-02,
                                      -4.36734625e-02,  3.99635356e-02, -5.22832424e-02])
            self.init_obs = np.array([4.64195583e-03,  8.63199317e-03, -3.53298290e-04, -1.20736697e-02,
                                     -4.36734625e-02,  3.99635356e-02, -5.22832424e-02,  7.27691469e-04,
                                     -3.06611558e-03,  2.76888153e-03,  4.25122445e-03, -4.96030552e-04,
                                      3.52372508e-03,  3.00150844e-03, -7.27309631e-03, -1.83452888e-02,
                                      1.10363540e+00])
            self.goal = np.array([-0.44003503,  0.26426241,  0.25445901])

        elif exp_type == 'peg':
            self.init_qpos = np.zeros(7)
            self.init_obs = np.zeros(20)
            self.goal = np.zeros(3)

        elif exp_type == 'bottle':
            self.init_qpos = np.zeros(7)
            self.init_obs = np.zeros(20)
            self.goal = np.zeros(3)

        elif exp_type == 'stack':
            self.init_qpos = np.zeros(7)
            self.init_obs = np.zeros(20)
            self.goal = np.zeros(3)

        elif exp_type == 'match':
            self.init_qpos = np.zeros(7)
            self.init_obs = np.zeros(20)
            self.goal = np.zeros(3)

        elif exp_type == 'push':
            self.init_qpos = np.zeros(7)
            self.init_obs = np.zeros(20)
            self.goal = np.zeros(3)

        else:
            raise NotImplementedError

        self.alpha = 10e-5
        self.ctrl_penalty = ctrl_penalty
        self.vel_penalty = vel_penalty
        max_torques = np.array(max_torques)

        self.joint_goal_pos = np.zeros(7) # joint position control vs torque control

        self.frame_skip = 1
        self.dt = 0.2
        super(ArmReacherEnv, self).__init__(side, ip, port)
        self.init_qpos = self.get_joint_positions()
        self._prev_qpos = self.init_qpos.copy()

        self.act_dim = len(max_torques)
        self.obs_dim = len(self._get_obs())

        self._low = -max_torques
        self._high = max_torques
        self.positions = {}
        self.actions = {}
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        return np.concatenate([
            self.get_joint_positions(), #10
            self.get_joint_velocities(), #7
            self.tip_position, #3
            ]).reshape(-1)

    def step(self, act):
        self._prev_qpos = self.get_joint_positions()
        self._prev_qvel = self.get_joint_velocities()
        if len(act) == 1:
            act = act[0]

        ob = self.do_simulation(act, self.frame_skip)
        joint_vel = self.get_joint_velocities()

        norm_end = np.linalg.norm(ob[-3:] - self.goal)
        reward_ctrl = -self.ctrl_penalty * np.square(np.linalg.norm(act))
        reward_vel = -self.vel_penalty * np.square(np.linalg.norm(joint_vel))
        reward_dist = self.gps_reward()
        reward = reward_dist + reward_ctrl + reward_vel

        done = False

        if self.actions is not None:
            action_num = len(self.actions)
            self.actions.update({action_num : act})

        if self.positions is not None:
            if len(self.positions) == 0:
                self.positions = dict({0 : np.vstack((self._prev_qpos, self._prev_qvel))})
            else:
                arr = np.vstack((self.get_joint_positions(), self.get_joint_velocities()))
                self.positions.update({len(self.positions) : arr})

        return ob, reward, done, dict(dist=norm_end, reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_vel=reward_vel)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def do_simulation(self, action,frame_skip):
        action = np.clip(action, self._low, self._high)
        assert frame_skip > 0
        for _ in range(frame_skip):
            time.sleep(self.dt)
            self.set_joint_torques(action)
        ob = self._get_obs()
        return ob


    def tf_reward(self, obs, act, obs_next):
        norm_end = tf.linalg.norm(obs_next[:, -3:] - self.goal, axis=1)
        reward_ctrl = -self.ctrl_penalty * tf.square(tf.linalg.norm(act, axis=1))
        reward_vel = -self.vel_penalty * tf.square(tf.linalg.norm(obs_next[:, 7:14], axis=1))
        reward_dist = -(tf.square(norm_end) + tf.log(tf.square(norm_end) + self.alpha))
        reward_ctrl = tf.cast(reward_ctrl, tf.float32)
        reward_vel = tf.cast(reward_vel, tf.float32)
        reward_dist = tf.cast(reward_dist, tf.float32)
        reward = reward_dist + reward_ctrl + reward_vel
        return tf.clip_by_value(reward, -100, 100)

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            norm_end = np.linalg.norm(obs_next[:, -3:] - self.goal, axis=1)
            reward_ctrl = -self.ctrl_penalty * np.square(np.linalg.norm(act, axis=1))
            reward_vel = -self.vel_penalty * np.square(np.linalg.norm(obs_next[:, 7:14], axis=1))
            reward_dist = -(np.square(norm_end) + np.log(np.square(norm_end) + self.alpha))
            reward = reward_dist + reward_ctrl + reward_vel
            return np.clip(reward, -1e2, 1e2)

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

        else:
            raise NotImplementedError

    def reset(self):
        self.set_joint_positions(np.zeros(7), duration=5)
        if not self.fixed:
            while True:
                self.goal = np.random.uniform(low=[-0.75, -0.25, 0.25], high=[-0.25, 0.25, 0.5])
                if np.linalg.norm(self.goal) < 2:
                    break
        else:
            self.goal = np.array([-0.44003503,  0.26426241,  0.25445901])
        obs = self._get_obs()
        return obs

    def joint_goal(self):
        base_roll = np.random.uniform(low=-(np.pi/4.0), high=np.pi/4.0)
        right_shoulder_lift = np.random.uniform(low=-2.2944, high=-np.pi/4.0)
        right_shoulder_roll = np.random.uniform(low=-(np.pi/4.0), high=np.pi/4.0)
        right_elbow_lift = np.random.uniform(low=0, high=0)
        right_elbow_roll = np.random.uniform(low=0, high=0)
        right_wrist_lift = np.random.uniform(low=0, high=0)
        right_wrist_roll = np.random.uniform(low=0, high=0)
        return np.array([
            base_roll,
            right_shoulder_lift,
            right_shoulder_roll,
            right_elbow_lift,
            right_elbow_roll,
            right_wrist_lift,
            right_wrist_roll
        ])

    def joint_match(self):
        real_joint_positions = self.get_joint_positions()

        return -np.linalg.norm(real_joint_positions, self.joint_goal_pos)

    def gps_reward_tf(self):
        norm = tf.linalg.norm(self.vec_gripper_to_goal)
        return -(tf.square(norm) + tf.log(tf.square(norm) + self.alpha))

    def gps_reward(self):
        norm = np.linalg.norm(self.vec_gripper_to_goal)
        return -(np.square(norm) + np.log(np.square(norm) + self.alpha))

    @property
    def tip_position(self):
        pose = self.get_cartesian_pose()
        return pose['position']

    @property
    def vec_gripper_to_goal(self):
        gripper_pos = self.tip_position
        vec_gripper_to_goal = self.goal - gripper_pos
        return vec_gripper_to_goal

    @property
    def vec_arm_to_goal_pos(self):
        arm_pos = self.get_joint_positions()
        vec_arm_to_goal = self.goal_pos - arm_pos
        return vec_arm_to_goal
        

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def log_diagnostics(self, paths, prefix=''):
        dist = [path["env_infos"]['dist'] for path in paths]
        final_dist = [path["env_infos"]['dist'][-1] for path in paths]
        reward_cost = [-path["env_infos"]['reward_dist'] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]
        vel_cost = [-path["env_infos"]['reward_vel'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgRewardDist', np.mean(reward_cost))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        logger.logkv(prefix + 'AvVelocityCost', np.mean(vel_cost))


if __name__ == "__main__":
    env = ArmReacherEnv()
    while True:
        env.reset()
        for _ in range(100000000):
            env.set_joint_torques(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
            #env.step(env.action_space.sample() * 0)
            #env.render()
