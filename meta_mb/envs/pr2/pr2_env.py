import numpy as np
from meta_mb.logger import logger
import gym
import mujoco_py
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv, RandomEnv
import os


class PR2ReacherEnv(RandomEnv, gym.utils.EzPickle):
    def __init__(self,
                 exp_type='reach',
                 max_torques=[3, 3, 2, 1, 1, 0.5, 1],
                 vel_penalty=1.25e-2,
                 torque_penalty=1.25e-1,
                 log_rand=1,
                 joint=True):
        self.max_torques = np.array(max_torques)
        self.vel_penalty = -vel_penalty
        self.torque_penalty = -torque_penalty
        self.exp_type = exp_type
        self.joint = joint

        self.gains = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

        if self.exp_type == 'reach':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2_new.xml')
            self.goal2 = np.array([-0.30982005, 0.71146246, 0.51908543,
                                   -0.14216614, 0.78684261, 0.56139753,
                                   -0.20410874, 0.64335638, 0.61437626])
            self.joint_goal = np.array([1.37154795, -0.20691918, 1.27061209, -1.11557631, 1.46168019, -1.7804405, 0.0283875])
        elif self.exp_type == 'shape':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2_shape.xml')
            self.goal2 = np.array([-0.35572354, 0.12565246, -0.0315576,
                                   -0.18394014, 0.18021465, -0.02460234,
                                   -0.2609592, 0.04723381, 0.03987207])
            self.joint_goal = np.array(
                [1.37154795, -0.20691918, 1.27061209, -1.11557631, 1.46168019, -1.7804405, 0.0283875])

        self.goal = np.zeros(3)

        self.ref_point = np.array([-0.31032794, -0.03286406,  0.19625491,
          -0.23757531,  0.03472754,  0.20802058,
          -0.25541497, -0.03043181,  0.26769475])

        self.init_grip = np.array([-3.06694218e-01, 1.87049223e-01, 1.12720687e-03,
                                   -2.03442785e-01, 1.59440809e-01, 1.02890217e-02,
                                   -3.07411827e-01, 1.18937711e-01, 7.99029507e-02])

        self.init_qpos = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   1.31405443e+00, -1.54883181e+00, 1.43069760e-01])
        self.init_qvel = np.zeros(7)
        self.alpha = 10e-5
        self.offset = np.array(
            [0.76801963, -0.03153535,  0.68872096,
             0.73720673, -0.10671044,  0.65789482,
             0.71356131, -0.04428578,  0.62665989])

        self._low = -self.max_torques
        self._high = self.max_torques
        self.goal2 -= self.offset
        #MujocoEnv.__init__(self, xml_file, 4)
        RandomEnv.__init__(self, log_rand, xml_file, 4)
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, self._low, self._high).astype(np.float32) * self.gains
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if not self.joint:
            vec = self.ee_position - self.goal2
        else:
            vec = ob[:7] - self.joint_goal
        norm = np.linalg.norm(vec)
        reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))
        reward_vel = self.vel_penalty * np.square(np.linalg.norm(ob[7:14]))
        reward_ctrl = self.torque_penalty * np.square(np.linalg.norm(action))
        reward = reward_dist + reward_vel + reward_ctrl
        done = False
        return ob, reward, done, dict(dist=norm, reward_vel=reward_vel, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0
        self.viewer.cam.azimuth = 180

    def tf_reward(self, obs, act, obs_next):
        return "wut"

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            if not self.joint:
                vec = obs_next[:, -9:] - self.goal2
            else:
                vec = obs_next[:, :7] - self.goal_joint
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
        self.frame_skip = np.random.randint(1, 5)  # randomize frameskips
        gravity = np.random.randint(-1, 1)  # randomize environment gravity
        self.model.opt.gravity[2] = gravity
        while True:
            #x = np.random.uniform(low=0.1, high=0.6)
            #y = np.random.uniform(low=0.1, high=0.65)
            #z = np.random.uniform(low=0.5, high=0.9)
            #self.goal = np.array([x, y, z])
            #self.goal2 = np.concatenate([
            #    np.array([0.02, -0.025,  0.05]) + self.goal,
            #    np.array([0.02, -0.025, -0.05]) + self.goal,
            #    np.array([0.02,  0.05,   0.00]) + self.goal])
            if np.linalg.norm(self.goal) < 2:
                break
        qvel = self.init_qvel # + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        #self.goal = self.hole()
        self.set_state(np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   1.31405443e+00, -1.54883181e+00, 1.43069760e-01]), qvel)
        #for i in range(len(self.model.mesh_normal)):
        #    self.model.mesh_normal[i] *= 5
        #self.sim = mujoco_py.MjSim(self.model)
        #mujoco_py.MjRenderContext.update_sim(mujoco_py.MjRenderContext(self.sim), self.sim)
        return self._get_obs()

    def _get_obs(self):
        ob =  np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])
        #noise = np.random.uniform(low=0, high=0.1, size=len(ob))
        return ob #+ noise

    def log_diagnostics(self, paths, prefix=''):
        dist = [-path["env_infos"]['dist'] for path in paths]
        final_dist = [-path["env_infos"]['dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]
        vel_cost = [-path["env_infos"]['reward_vel'] for path in paths]

        logger.logkv(prefix + 'AvgDist', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDist', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        logger.logkv(prefix + 'AvgVelCost', np.mean(vel_cost))

    def hole(self):
        return self.sim.data.get_site_xpos('goal')


if __name__ == "__main__":
    env = PR2ReacherEnv(exp_type='reach')
    while True:
        env.reset()
        for i in range(2000):
            env.step(env.action_space.sample())
            env.render()
