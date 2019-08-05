import numpy as np
import os
from meta_mb.meta_envs.base import MetaEnv, RandomEnv
from meta_mb.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv


def mass_center(model, sim):
    mass = np.expand_dims(model.body_mass, 1)
    xpos = sim.data.xipos
    return (np.sum(mass * xpos, 0) / np.sum(mass))


class DarwinNoTorsoEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self, log_rand=0):
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'darwin.xml')
        MujocoEnv.__init__(self, xml_file, 5)
        #RandomEnv.__init__(self, log_rand, xml_file, 5)
        gym.utils.EzPickle.__init__(self)

    def _get_obs(self):
        data = self.sim.data
        obs = np.concatenate([
            data.qpos.flat[:7],
            data.qvel.flat[:6],
        ])
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
        qpos = self.sim.data.qpos
        pelvis_height_l = self.get_body_com("pelvis_l")[2]
        pelvis_height_r = self.get_body_com("pelvis_r")[2]
        # bool((pelvis_height_l < 0.2) or (pelvis_height_l > 0.4)) #Use this for cripple
        done = bool((pelvis_height_l < 0.2) or (pelvis_height_l > 0.4) or (pelvis_height_r < 0.2) or (pelvis_height_r > 0.4))
        return self._get_obs(), reward, done, dict(reward_linvel=lin_vel_cost, reward_quadctrl=-quad_ctrl_cost,
                                                   reward=reward, reward_alive=alive_bonus)
    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        print(obs.ndim)
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
            notdone = np.isfinite(obs).all()  and obs[0] >= 0.2 and obs[0] <= 1
            return not notdone

    def reset_model(self):
        c = 0.01
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv, )
        )
        obs = self._get_obs()
        return obs

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.6
        self.viewer.cam.elevation = 3
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.type = 0

    def log_diagnostics(self, paths, prefix=''):
        #vel_cost = [-path["env_infos"]['reward_linvel'] for path in paths]
        #ctrl_cost = [-path["env_infos"]['reward_quadctrl'] for path in paths]
        #rwd = [-path["env_infos"]["reward"] for path in paths]

        #logger.logkv(prefix + 'AvgReward', np.mean(rwd))
        #logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))
        #logger.logkv(prefix + 'AvgVelCost', np.mean(vel_cost))
        None


if __name__ == "__main__":
    env = DarwinNoTorsoEnv()
    while True:
        env.reset()
        for _ in range(2000):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample() * 0)  # take a random action