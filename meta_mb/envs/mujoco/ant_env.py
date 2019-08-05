import numpy as np
from meta_mb.meta_envs.base import MetaEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
from meta_mb.logger import logger
import tensorflow as tf
import gym


class AntEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self, goal_direction=None):
        MujocoEnv.__init__(self, 'ant.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.2
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reward(self, obs, acts, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == acts.shape[0]
        reward_ctrl = -0.5 * np.sum(np.square(acts), axis=1)
        reward_run = obs[:, 13]
        reward = reward_run + reward_ctrl + 1.0
        return reward

    def tf_reward(self, obs, acts, next_obs):
        reward_ctrl = -0.5 * tf.reduce_sum(tf.square(acts), axis=1)
        reward_run = next_obs[:, 13]
        reward = reward_run + reward_ctrl + 1.0
        return reward

    def tf_termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        x = next_obs[:, 0]
        not_done = tf.math.logical_and(tf.math.logical_and(tf.reduce_all(tf.is_finite(next_obs), axis = -1, keepdims = False), (x >= 0.2)), (x <= 1.0))
        done = ~not_done
        done = done[:, None]
        return done

    def termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        x = next_obs[:, 0]
        not_done = np.isfinite(next_obs).all(axis=-1) * (x >= 0.2) * (x <= 1.0)
        done = ~not_done
        return done

    def done(self, obs):
        if obs.ndim == 2:
            notdone = np.all(np.isfinite(obs), axis=1) * (obs[:, 0] >= 0.2) * (obs[:, 0] <= 1.0)
            return np.logical_not(notdone)
        else:
            notdone = np.isfinite(obs).all()  and obs[0] >= 0.2 and obs[0] <= 1.0
            return not notdone

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        progs = [np.mean(path["env_infos"]["reward_forward"]) for path in paths]
        ctrl_cost = [-np.mean(path["env_infos"]["reward_ctrl"]) for path in paths]

        logger.logkv(prefix+'AverageForwardReturn', np.mean(progs))
        logger.logkv(prefix+'MaxForwardReturn', np.max(progs))
        logger.logkv(prefix+'MinForwardReturn', np.min(progs))
        logger.logkv(prefix+'StdForwardReturn', np.std(progs))
        logger.logkv(prefix+'AverageCtrlCost', np.mean(ctrl_cost))
