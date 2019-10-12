from gym import spaces

import numpy as np
import copy
from meta_mb.envs.mb_envs.pmaze_grids import grids

class ParticleEnv(object):
    def __init__(self):
        self.obs_dim = obs_dim = 2
        self.act_dim = act_dim = 2
        self.goal_dim = 2

        self.dt = 0.1

        self.obs_low, self.obs_high = np.ones(obs_dim) * (-1.0), np.ones(obs_dim) * (1.0)
        self.act_low, self.act_high = np.ones(act_dim) * (-1.0), np.ones(act_dim) * (1.0)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = spaces.Box(low=self.act_low, high=self.act_high)

        self.start_state = self.observation_space.sample()
        _ = self.reset()

    def reset(self): # specify what the goal is here?
        self.state = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.goal = None
        return self._get_ob()

    def _get_ob(self):
        return self.state

    def set_goal(self, goal):
        self.goal = goal

    def sample_goals(self, num_goals):
        return np.random.uniform(low=self.obs_low, high=self.obs_high, size=(num_goals, self.obs_dim))

    def step(self, action):
        ob = self._get_ob()
        act = np.clip(action, self.action_space.low, self.action_space.high)
        next_ob = ob + act * self.dt
        next_ob = np.clip(next_ob, self.observation_space.low, self.action_space.high)
        rew = self.reward(ob, act, next_ob)
        done = False
        return next_ob, rew, done, {}

    def reward(self, ob, act, next_ob):
        rew_run = -np.sum(np.square(next_ob - self.goal))
        rew_ctrl = -0.1 * np.sum(np.square(act))
        return rew_run + rew_ctrl

class ParticleMazeEnv(ParticleEnv):
    def __init__(self, grid_name='1'):
        super().__init__()
        
        self.num_substeps = 10
        self.ddt = self.dt / self.num_substeps
        self.grid = grids[grid_name]

        self._reset_grid()

    def _reset_grid(self):
        # transform str grid to np array
        _grid = self.grid.replace('\n', '')
        self.grid_size = int(np.sqrt(len(_grid)))

        start_ind = _grid.index('S')
        start_ind = np.array([start_ind // self.grid_size, start_ind % self.grid_size])
        _grid = _grid.replace('S', ' ')
        self.start_state = self._get_coords(start_ind)

        self.grid = np.reshape(list(_grid), (self.grid_size, self.grid_size))

    def reset(self):
        ob = self._set_state(self.start_state)
        self.goal = None
        return ob

    def _is_wall(self, obs):
        ind = self._get_index(obs)
        return self.grid[ind[0], ind[1]]

    def _get_coords(self, ind):
        return ((ind + 0.5) / self.grid_size) * 2.0 - 1.0

    def _get_index(self, coords):
        return np.clip((((coords + 1.0) * 0.5) * (self.grid_size)) + 0.0, 0, self.grid_size-1).astype(np.int8)

    def _set_state(self, ob):
        self.state = np.clip(ob, self.observation_space.low, self.observation_space.high)
        return self.state

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        init_ob = ob = self._get_ob()
        print('taking step from ', ob, self._get_index(ob), 'action', action)
        if self._is_wall(init_ob):
            print('resetting')
            self.reset()

        # substep collision detection
        collision = False
        for substep in range(self.num_substeps):
            next_ob = ob + action * self.ddt
            print('substep', substep, 'index', self._get_index(next_ob))
            if self._is_wall(next_ob):
                collision = True
                # ob, *_ = self.step(-0.1*action)
                break
            else:
                ob = self._set_state(next_ob)

        reward = self.reward(init_ob, action, ob) - int(collision)
        done = False  # FIXME
        info = {}

        return ob, reward, done, info


class IterativeEnvExecutor(object):
    def __init__(self, env, num_rollouts, max_path_length):
        self._num_envs = num_rollouts
        self.max_path_length = max_path_length
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self._num_envs)])
        self.ts = np.zeros(self._num_envs, dtype='int')  # time steps

    def set_goal(self, goal_ng):
        for goal, env in zip(goal_ng, self.envs):
            env.set_goal(goal)

    def step(self, act_na):

        assert len(act_na) == self._num_envs
        all_results = [env.step(a) for (a, env) in zip(act_na, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)

        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def reset(self):
        init_ob_no = [env.reset() for env in self.envs]
        return init_ob_no

    @property
    def num_envs(self):
        return self._num_envs


if __name__ == "__main__":
    env = ParticleEnv()
    env.set_goal(env.observation_space.sample())
    env.sample_goals(2)
    env.step(env.action_space.sample())


