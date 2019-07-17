import numpy as np
from meta_mb.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv
import os


class PR2ReacherEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.goal = np.zeros(3)
        self.goal2 = np.array([-0.30982005, 0.71146246, 0.51908543,
                               -0.14216614, 0.78684261, 0.56139753,
                               -0.20410874, 0.64335638, 0.61437626])

        self.init_grip = np.array([-3.06694218e-01, 1.87049223e-01, 1.12720687e-03,
                                   -2.03442785e-01, 1.59440809e-01, 1.02890217e-02,
                                   -3.07411827e-01, 1.18937711e-01, 7.99029507e-02])

        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2.xml')
        self.init_qpos = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   1.31405443e+00, -1.54883181e+00, 1.43069760e-01])
        self.init_qvel = np.zeros(7)
        self.offset_0 = np.array([ 0.8571748,  -0.04651993,  0.76642218])
        self.offset_1 = np.array([ 0.88839121, -0.07360844,  0.68504259])
        self.offset_2 = np.array([ 0.69829279, -0.08945529,  0.76459628])
        MujocoEnv.__init__(self, xml_file, 2)
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        vec = self.get_body_com("l_gripper_r_finger_tip_link") - self.goal
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
        #qpos = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
        #                               1.31405443e+00, -1.54883181e+00, 1.43069760e-01]) # + self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qpos = np.zeros(7)
        self.sim.model.body_pos[-1] = self.goal2[:3] + self.offset_0
        self.sim.model.body_pos[-2] = self.goal2[3:6] + self.offset_1
        self.sim.model.body_pos[-3] = self.goal2[6:] + self.offset_2
        while True:
            self.goal = np.random.uniform(low=-.2, high=.2, size=3)
            #self.goal = np.array([-.3, -.3, 1])
            if np.linalg.norm(self.goal) < 2:
                break
        qvel = self.init_qvel # + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.ee_position
        ])

    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [-path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDist', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDist', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))

    @property
    def ee_position(self):
        p1 = self.sim.data.get_site_xpos("p1")
        p2 = self.sim.data.get_site_xpos("p2")
        p3 = self.sim.data.get_site_xpos("p3")
        return np.concatenate((p1, p2, p3), axis=0)


if __name__ == "__main__":
    env = PR2ReacherEnv()
    while True:
        env.reset()
        print(env.sim.data.get_site_xpos("p1"))
        print(env.sim.data.get_site_xpos("p2"))
        print(env.sim.data.get_site_xpos("p3"))
        offset = env.ee_position - env.init_grip
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()
