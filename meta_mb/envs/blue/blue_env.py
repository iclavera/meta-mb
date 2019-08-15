import numpy as np
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import RandomEnv
from meta_mb.logger import logger
from gym import utils
import os
import pickle


class BlueEnv(RandomEnv, utils.EzPickle): 
    def __init__(self, 
                arm='right', 
                log_rand=0, 
                max_torques=[2] * 7,
                ctrl_penalty=1.25e-2,
                vel_penalty=1.25e-1):
        utils.EzPickle.__init__(**locals())

        assert arm in ['left', 'right']
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_' + arm + '_v2.xml')

        self.goal = np.zeros((3,))
        self._arm = arm
        self.iters = 0

        self.ctrl_penalty = ctrl_penalty
        self.vel_penalty = vel_penalty
        self.alpha=10e-5

        max_torques = np.array(max_torques)
        self._low = -max_torques
        self._high = max_torques

        RandomEnv.__init__(self, log_rand, xml_file, 20)

    def _get_obs(self):
        obs = np.concatenate([
            self.sim.data.qpos.flat[:-3],
            self.sim.data.qvel.flat[:-3],
            self.get_body_com("right_gripper_link"),
        ])
        return obs

    def step(self, act):
        act = np.clip(act, self._low, self._high)
        self.do_simulation(act, self.frame_skip)

        norm = np.linalg.norm(self.get_body_com("right_gripper_link") - self.goal)
        joint_vel = self.sim.data.qvel[:-3]

        reward_ctrl = -self.ctrl_penalty * np.square(np.linalg.norm(act))
        reward_vel = -self.vel_penalty * np.square(np.linalg.norm(joint_vel))
        reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))

        reward = reward_dist + reward_ctrl + reward_vel

        observation = self._get_obs()
        info = dict(dist=norm, reward_dist=reward_dist, reward_ctrl=reward_ctrl, reward_vel=reward_vel)
        done = False
        self.iters += 1
        return observation, reward, done, info

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            norm = np.linalg.norm(obs_next[:, -3:] - self.goal)
            joint_vel = obs_next[:, 7:14]

            reward_ctrl = -self.ctrl_penalty * np.square(np.linalg.norm(act, axis=1))
            reward_vel = -self.vel_penalty * np.square(np.linalg.norm(joint_vel, axis=1))
            reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha, axis=1))

            reward = reward_dist + reward_ctrl + reward_vel
            return np.clip(reward, -1e2, 1e2)

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

        else:
            raise NotImplementedError

    def reset_model(self):
        #gravity = np.random.randint(-4, 1) #randomize environment gravity
        #self.model.opt.gravity[2] = gravity
        #self.sim.data.qfrc_applied[-1] = abs(gravity/1.90986) #counteract gravity on goal body
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.goal = np.random.uniform(low=[-0.75, -0.25, 0.25], high=[-0.25, 0.25, 0.5])
        #self.goal = np.array([-0.55, -0.1, 0.21]) #fixed goal
        qpos[-3:] = self.goal
        qvel[-3:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    @property
    def ee_position(self):
        return (self.get_body_com(self._arm + '_r_finger_tip_link')
                + self.get_body_com(self._arm + '_l_finger_tip_link'))/2

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0

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
    env = BlueEnv('right')
    while True:
        env.reset()
        for _ in range(10000):
            action = env.action_space.sample()
            env.step(action)
            env.render()
