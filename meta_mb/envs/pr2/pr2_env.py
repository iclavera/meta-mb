import numpy as np
from meta_mb.logger import logger
import gym
import mujoco_py
from mujoco_py import MjSim
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.meta_envs.base import MetaEnv, RandomEnv
import os


actions = [[-0.29097845, -0.34692378,  -1.14745959, -1.11427337,  7.52388136, -1.74997328,
  1.39198461],
[-0.29097845, -0.34692378,  -1.14745959, -1.11427337,  7.52388136, -1.74997328,
  1.39198461],
[-0.13469948, -0.33558806,  -1.60415013, -0.1487975,   8.80681406, -2.00180291,
 -5.03106323],
[-0.11811815,  -0.33558806,   -1.60479155,  -0.14923181,   8.80681406,
  -2.00119378, -11.86518334],
[-0.26643809, -0.2349201,   -3.74873558, -0.43312718,  4.35900939, -0.55539237,
 -5.93761509],
[-0.32563342, -0.14042757,  -3.74953735, -0.14836319, -1.66298806, -1.78800007,
  0.76006105],
[-0.47793287,  0.20082835, 0.15205593, -0.8526727,  -4.67913517, -2.00706749,
  7.47361795],
[-0.31477265,  0.1715585,  0.62847012, -2.05788887,  0.03632797, -2.02199105,
  3.05872432],
[-0.19074435, -0.31088634,  -1.93977278, -0.14734979,  2.21860933, -1.99997553,
 -3.04992806],
[ 0.01121617, -0.35445273,  -3.74873558, -0.14546777, -1.50176822, -1.99732149,
  2.80798241],
[-0.10186846, -0.26122912,  -1.05124669, -0.3553853,   2.00758315, -1.99762605,
  3.56038217],
[-5.64155754e-01, -5.83704969e-03, 6.40977793e-01, -2.05282189e+00,
  6.52029225e+00, -2.01033066e+00, -3.02251744e+00],
[-0.29106136, -0.24913205,  -3.68491436, -0.14648117,  5.40291792, -1.86718631,
 -3.73519355],
[ 0.23116743, -0.21021838,  -1.35960902, -0.2313168,   7.94946238, -1.37949482,
 -2.89873456],
[-0.51996653,  0.22789872, 0.64562808, -2.10334687, 12.06117548, -1.53482167,
 -9.57926819],
[-0.28426301,  0.04432774,  -2.00295258, -1.8610007,   6.76521682, -1.04264876,
 -3.57599441],
[-0.21130519, -0.34100214,  -3.75001842, -0.14937658,  4.26188413, -2.00010606,
  3.2083515],
[-0.21445564, -0.31714637,  -1.33539544, -0.22132762,  7.89763139, -1.99771307,
  4.06369336],
[-0.52916916, -0.04153611, 0.64161921, -2.0290795,  12.4926569,  -2.00898188,
 -2.64011754],
[-0.21370948, -0.18297882,  -2.4270911,  -2.12057459, 15.75702586, -0.68561457,
 -9.07073593],
[ 0.18548588, -0.16732876,  -3.75210303, -0.15024521, 10.45233229, -1.99553762,
 -3.03965996],
[ 0.17421058, -0.34624702,  -3.75146161, -0.14575731, 10.45215875, -2.00106326,
  3.77405448],
[-0.49658686, -0.10532068, 0.6910085,  -1.8466684,  17.27938765, -2.00772012,
 -2.14259304]]

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
                self.goal = np.array([0.61925865, 0.11996082, 0.96922978]) # 0.66748306  0.09903466  1.04018936
            RandomEnv.__init__(self, log_rand, xml_file, 2)

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
        qpos = np.array([ 0.43271341,  0.07884247,  0.9869444,  -1.53700385,  1.47712537, -2.00212515, 0.09811261])
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
            self.goal = np.array([0.61925865, 0.11996082, 0.96922978])
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
        self.set_state(qpos, qvel)
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
        return self.sim.data.get_site_xpos("ee_pos")

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
        for action in actions:
            #action = env.action_space.sample()
            #print(type(action))
            #action = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
            env.step(np.asarray(action))
            print(env._get_obs())
            env.render()
