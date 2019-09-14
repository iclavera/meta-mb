import numpy as np
from meta_mb.logger import logger
import gym
import mujoco_py
from mujoco_py import MjSim
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv, RandomEnv
import os


class PR2ReacherEnv(RandomEnv, gym.utils.EzPickle):
    def __init__(self,
                 exp_type='reach',
                 max_torques=[3, 3, 2, 1, 1, 0.5, 1],
                 vel_penalty=1.25e-2,
                 ctrl_penalty=1.25e-1,
                 log_rand=0,
                 joint=False):
        self.max_torques = np.array(max_torques)
        self.vel_penalty = -vel_penalty
        self.torque_penalty = -ctrl_penalty
        self.exp_type = exp_type
        self.joint = joint

        self.gains = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

        self.obs_dim = 17
        self.act_dim = 7

        print("EXPTYPE:", self.exp_type)

        self.init_qpos = np.zeros(7)
        self.init_qvel = np.zeros(7)
        self.alpha = 10e-5

        self._low = -self.max_torques
        self._high = self.max_torques
        self.reached = False
        if self.exp_type == 'reach':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2.xml')
            if self.joint:
                self.goal = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   1.31405443e+00, -1.54883181e+00, 1.43069760e-01])
            else:
                self.goal = np.array([0.61925865, 0.11996082, 0.96922978])
            RandomEnv.__init__(self, log_rand, xml_file, 4)

        elif self.exp_type == 'shape':
            xml_file = os.path.joint(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2_shape.xml')
            self.reached = False
            if self.joint:
                raise NotImplementedError
            else:
                self.goal = np.zeros(3)
            RandomEnv.__init__(self, log_rand, xml_file, 4)

        elif self.exp_type == 'peg':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2_peg.xml')
            if self.joint:
                raise NotImplementedError
            else:
                self.goal = np.zeros(3)
            RandomEnv.__init__(self, log_rand, xml_file, 4)
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, self._low, self._high)#.astype(np.float32) * self.gains
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if self.exp_type == 'reach':
            if not self.joint:
                vec = self.get_ee_pos - self.goal
            else:
                vec = ob[:7] - self.goal
        elif self.exp_type == 'shape':
            raise NotImplementedError
        elif self.exp_type == 'peg':
            if not self.reached:
                vec = ob[-3:] - self.get_body_com('g1')
            else:
                vec = ob[-3:] - self.get_body_com('g3')
        norm = np.linalg.norm(vec)
        if not self.reached and self.exp_type == 'peg' and norm < 3e-2:
            self.reached = True
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

            if self.exp_type == 'reach':
                if not self.joint:
                    vec = obs_next[:, -3:] - self.goal
                else:
                    vec = obs_next[:, :7] - self.goal
                norm = np.linalg.norm(vec, axis=1)
            elif self.exp_type == 'shape':
                raise NotImplementedError
            elif self.exp_type == 'peg':
                norm = np.array()
                for row in obs_next:
                    if not self.reached:
                        v = row[:-3] - self.get_body_com('g1')
                    else:
                        v = row[:-3] - self.get_body_com('g3')
                v_norm = np.linalg.norm(v)
                if not self.reached and v_norm < 3e-2:
                    self.reached = True
                norm.append(v_norm)

            reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))
            reward_vel = self.vel_penalty * np.square(np.linalg.norm(obs_next[:, 7:14], axis=1))
            reward_ctrl = self.torque_penalty * np.square(np.linalg.norm(act, axis=1))
            reward = reward_dist + reward_vel + reward_ctrl
            return np.clip(reward, -1e5, 1e5)
        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]
        else:
            raise NotImplementedError

    def reset_model(self):
        qpos = np.array([0.43909722, 0.07478191, 1.25008667, -1.3573434, 1.67461532, -1.64366539, 12.61531702])
        #self.frame_skip = np.random.randint(1, 5)  # randomize frameskips
        #gravity = np.random.randint(-1, 1)  # randomize environment gravity
        #self.model.opt.gravity[2] = gravity
        while True:
            x = np.random.uniform(low=0.2, high=0.7)
            y = np.random.uniform(low=0.0, high=0.2)
            z = np.random.uniform(low=0.5, high=0.8)
            if np.linalg.norm(self.goal) < 2:
                break
        if self.exp_type == 'reach':
            self.goal = np.array([x, y, z])
        elif self.exp_type == 'shape':
            raise NotImplementedError
        elif self.exp_type == 'peg':
            self.sim.model.body_pos[-4] = np.array([x, y, z])
            self.sim.model.body_pos[-3] = self.g1()
            self.sim.model.body_pos[-2] = self.g2()
            self.sim.model.body_pos[-1] = self.g3()
        else:
            raise NotImplementedError
        qvel = self.init_qvel
        self.set_state(np.zeros(7), qvel)
        return self._get_obs()

    def _get_obs(self):
        ob = np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            self.get_ee_pos
        ])
        #noise = np.random.uniform(low=0, high=0.1, size=len(ob))
        return ob #+ noise

    def log_diagnostics(self, paths, prefix=''):
        dist = [path["env_infos"]['dist'] for path in paths]
        final_dist = [path["env_infos"]['dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]
        vel_cost = [-path["env_infos"]['reward_vel'] for path in paths]

        logger.logkv(prefix + 'AvgDist', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDist', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        logger.logkv(prefix + 'AvgVelCost', np.mean(vel_cost))

    @property
    def get_ee_pos(self):
        return self.sim.model.site_pos[-1]

    def g1(self):
        x = 0.092
        y = 0
        z = 0.15 + np.random.uniform(low=-0.1, high=0.2)
        return np.array([x, y, z])

    def g2(self):
        x = 0.092
        y = 0
        z = 0.078
        return np.array([x, y, z])

    def g3(self):
        x = 0.092
        y = 0
        z = 0.04
        return np.array([x, y, z])

if __name__ == "__main__":
    env = PR2ReacherEnv(exp_type='reach')
    file = 'pr2_'
    while True:
        env.reset()
        for i in range(200):
            env.step(env.action_space.sample())
            env.render()
