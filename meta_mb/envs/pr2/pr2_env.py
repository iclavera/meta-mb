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
                 joint=True):
        self.max_torques = np.array(max_torques)
        self.vel_penalty = -vel_penalty
        self.torque_penalty = -ctrl_penalty
        self.exp_type = exp_type
        self.joint = joint

        self.gains = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

        if self.exp_type == 'reach':
            xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'pr2_new.xml')
            if self.joint:
            	self.goal = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   1.31405443e+00, -1.54883181e+00, 1.43069760e-01])
            else:
            	self.goal2 = np.array([-0.30982005, 0.71146246, 0.51908543,
                                   	   -0.14216614, 0.78684261, 0.56139753,
                                       -0.20410874, 0.64335638, 0.61437626])

        self.init_qpos = np.zeros(7)
        self.init_qvel = np.zeros(7)
        self.alpha = 10e-5

        self._low = -self.max_torques
        self._high = self.max_torques
        RandomEnv.__init__(self, log_rand, xml_file, 4)
        self.obs_dim = 14
        self.act_dim = 7
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        action = np.clip(action, self._low, self._high).astype(np.float32) * self.gains
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        if not self.joint:
            vec = self.ee_position - self.goal
        else:
            vec = ob[:7] - self.goal
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
                vec = obs_next[:, :7] - self.goal
            norm = np.linalg.norm(vec)
            reward_dist = -(np.square(norm) + np.log(np.square(norm) + self.alpha))
            reward_vel = self.vel_penalty * np.square(np.linalg.norm(obs_next[:, 7:14]))
            reward_ctrl = self.torque_penalty * np.square(np.linalg.norm(act))
            reward = reward_dist + reward_vel + reward_ctrl
            return np.clip(reward, -1e5, 1e5)
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
        #self.frame_skip = np.random.randint(1, 5)  # randomize frameskips
        #gravity = np.random.randint(-1, 1)  # randomize environment gravity
        #self.model.opt.gravity[2] = gravity
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
        self.set_state(np.zeros(7), qvel)#np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                   #1.31405443e+00, -1.54883181e+00, 1.43069760e-01]), qvel)
        #for i in range(len(self.model.mesh_normal)):
        #    self.model.mesh_normal[i] *= 5
        #self.sim = mujoco_py.MjSim(self.model)
        #mujoco_py.MjRenderContext.update_sim(mujoco_py.MjRenderContext(self.sim), self.sim)
        return self._get_obs()

    def _get_obs(self):
        #if not self.has_cam and self.cam:
        #    self.configure_cam()
        #    self.has_cam = 1
        #try:
        #data = self.sim.render(camera_name='stereo', width=, height=64, depth=False)

        #except:
        #    data = np.zeros((128, 128))
        #    self.cam = 1

        #print(data)
        ob = np.concatenate([
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
    import time
    file = 'pr2_'
    from matplotlib import pyplot as plt
    while True:
        env.reset()
        for i in range(200):
            env.step(env.action_space.sample())
        #    viewer = mujoco_py.MjRenderContextOffscreen(env.sim, 0)
            #for i in range(3):
                #viewer.render(128, 128, 0)
                #data = np.asarray(viewer.read_pixels(128, 128, depth=False)[:, :, :], dtype=np.uint8)
                #print(data)
                #plt.imshow(data, interpolation='nearest')
                #plt.show()
                #time.sleep(1000000)

            #sim = MjSim(env.model)
            #print(sim)
            #image = sim.render(128, 128, camera_name='stereo')
            #print("HOWHERE")
            #plt.imshow(image, interpolation='nearest')
            #plt.show()
            #time.sleep(100000)
            #fname = 'pr2_images_2/' + file + str(i)
            #plt.savefig(fname, format='png')
            #time.sleep(100000)
            env.render()
