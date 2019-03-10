import numpy as np
import zmq
from meta_mb.logger import logger
import gym
from gym import spaces
from meta_mb.meta_envs.base import MetaEnv
import time


class PR2Env(MetaEnv, gym.utils.EzPickle):
    def __init__(self, ip='127.0.0.1', port=9090):
        self.goal = np.ones((3,))
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        print("Connecting to the server…")
        self.socket.bind("tcp://*:5555")
        max_torques = np.array([3] * 7)
        self.frame_skip = 1
        self.init_qpos = np.array([0., 0., 0., 0., 0., 0., 0.])
        self.act_dim = 7
        self.obs_dim = len(self._get_obs())
        self._low, self._high = -max_torques, max_torques
        gym.utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward_dist = -np.linalg.norm(ob[-3:])
        reward_ctrl = -np.square(action/(2 * self._high)).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def do_simulation(self, action, frame_skip):
        # md = dict(
        #     dtype=str(action.dtype),
        #     shape=A.shape,
        # )
        # socket.send_json(md, 0 | zmq.SNDMORE)
        action = np.clip(action, self._low, self._high)
        self.socket.send(action.astype(np.float32), 0, copy=True, track=False)

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act/(2 * self._high)), axis=1)
            reward_dist = -np.linalg.norm(obs_next[:,-3:], axis=1)
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e3, 1e3)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act/(2 * self._high)))
            reward_dist = -np.linalg.norm(obs_next[-3:])
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e3, 1e3)
        else:
            raise NotImplementedError

    def reset(self):
        pass

    def _get_obs(self):
        msg = self.socket.recv(flags=0, copy=True, track=False)
        buf = memoryview(msg)
        obs = np.frombuffer(buf, dtype=np.float32)
        return obs.reshape(-1)

    def log_diagnostics(self, paths, prefix=''):
        dist = [path["env_infos"]['reward_dist'] for path in paths]
        final_dist = [path["env_infos"]['reward_dist'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgCtrlCost', np.mean(ctrl_cost))

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

if __name__ == "__main__":
    env = PR2Env()
    while True:
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
