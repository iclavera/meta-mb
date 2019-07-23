import numpy as np
from meta_mb.logger import logger
import gym
import mujoco_py
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv, RandomEnv
import os


class PR2ReacherEnv(RandomEnv, gym.utils.EzPickle):
    def __init__(self,
                 exp_type='reach',
                 max_torques=[3] * 7,
                 vel_penalty=1.25e-2,
                 torque_penalty=1.25e-1,
                 log_rand=1):
        self.max_torques = np.array(max_torques)
        self.vel_penalty = -vel_penalty
        self.torque_penalty = -torque_penalty
        self.exp_type = exp_type

        if self.exp_type == 'reach':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2.xml')
            self.goal2 = np.array([-0.30982005, 0.71146246, 0.51908543,
                                   -0.14216614, 0.78684261, 0.56139753,
                                   -0.20410874, 0.64335638, 0.61437626])
        elif self.exp_type == 'shape':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2_shape.xml')
            self.goal2 = np.array([-0.35572354, 0.12565246, -0.0315576,
                                   -0.18394014, 0.18021465, -0.02460234,
                                   -0.2609592, 0.04723381, 0.03987207])

        self.goal = np.zeros(3)

        self.ref_point = np.array([1.48728815e-01, 2.11313570e-01, 3.51235070e-01,
                                   2.44708389e-01, 2.78963699e-01, 2.65735606e-01,
                                   2.15255002e-01, 2.88775077e-01, 3.74087410e-01])

        self.init_grip = np.array([-3.06694218e-01, 1.87049223e-01, 1.12720687e-03,
                                   -2.03442785e-01, 1.59440809e-01, 1.02890217e-02,
                                   -3.07411827e-01, 1.18937711e-01, 7.99029507e-02])

        self.init_qpos = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   1.31405443e+00, -1.54883181e+00, 1.43069760e-01])
        self.init_qvel = np.zeros(7)
        self.alpha = 10e-5
        self.offset = np.array(
           [0.68027119, -0.04831357,  0.48943993,
            0.58429161, -0.1159637,   0.47493939,
            0.613745,   -0.05077508,  0.41658759])
        self.goal2 += self.offset
        #MujocoEnv.__init__(self, xml_file, 4)
        RandomEnv.__init__(self, log_rand, xml_file, 4)
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        vec = self.ee_position - self.goal2
        norm = np.linalg.norm(vec)
        reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))
        reward_vel = self.vel_penalty * np.square(np.linalg.norm(ob[7:14]))
        reward_ctrl = self.torque_penalty * np.square(np.linalg.norm(action))
        reward = reward_dist + reward_vel + reward_ctrl
        done = False
        return ob, reward, done, dict(dist=norm, reward_vel=reward_vel, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def tf_reward(self, obs, act, obs_next):
        return "wut"

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            vec = obs_next[:, -9:] - self.goal2
            norm = np.linalg.norm(vec)
            reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))
            reward_vel = self.vel_penalty * np.square(np.linalg.norm(obs_next[:, 7:14]))
            reward_ctrl = self.torque_penalty * np.square(np.linalg.norm(act))
            reward = reward_dist + reward_vel + reward_ctrl
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
        qpos = self.init_qpos
        #gravity = np.random.randint(-3, 1)  # randomize environment gravity
        #self.model.opt.gravity[2] = gravity
        #self.sim.model.body_pos[-3] = self.goal2[:3] + self.offset[:3]
        #self.sim.model.body_pos[-2] = self.goal2[3:6] + self.offset[3:6]
        #self.sim.model.body_pos[-1] = self.goal2[6:] + self.offset[6:]
        while True:
            self.goal = np.random.uniform(low=0.2, high=.6, size=3)
            #self.goal = np.array([-.3, -.3, 1])
            if np.linalg.norm(self.goal) < 2:
                break
        qvel = self.init_qvel # + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        #for i in range(len(self.model.mesh_normal)):
        #    self.model.mesh_normal[i] *= 5
        #self.sim = mujoco_py.MjSim(self.model)
        #mujoco_py.MjRenderContext.update_sim(mujoco_py.MjRenderContext(self.sim), self.sim)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.ee_position
        ])

    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['dist'] for path in paths]
        final_dist = [-path["env_infos"]['dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]
        #vel_cost = [-path["env_infos"]['reward_vel'] for path in paths]

        logger.logkv(prefix + 'AvgDist', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDist', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        #logger.logkv(prefix + 'AvgVelCost', np.mean(vel_cost))


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
        for _ in range(100):
            env.step(env.action_space.sample())
            env.render()
