from gym import spaces

import numpy as np
import copy
from meta_mb.envs.mb_envs.maze_layouts import maze_layouts


class ParticleMazeEnv(object):
    def __init__(self, grid_name='novel1', reward_str='sparse'):
        self.obs_dim = obs_dim = 2
        self.act_dim = act_dim = 2
        self.goal_dim = 2

        self.obs_low, self.obs_high = np.ones(obs_dim) * (-1.0), np.ones(obs_dim) * (1.0)
        self.act_low, self.act_high = np.ones(act_dim) * (-1.0), np.ones(act_dim) * (1.0)
        self.observation_space = spaces.Box(low=self.obs_low, high=self.obs_high)
        self.action_space = spaces.Box(low=self.act_low, high=self.act_high)

        self.reward_str = reward_str
        self.num_substeps = 10
        self.ddt = 0.01  # dt = 0.1 = ddt * num_substeps
        self.grid = maze_layouts[grid_name]

        self._reset_grid()

    def _reset_grid(self):
        # transform grid to string with no line break
        _grid_flatten = self.grid.replace('\n', '')
        _grid_flatten = np.asarray(list(_grid_flatten))  # string of ' ', 'S', 'E', 'G', '+', '|', '-'

        self.grid_size = grid_size = int(np.sqrt(len(_grid_flatten)))
        _grid = np.reshape(_grid_flatten, (grid_size, grid_size))

        start_ind = np.argwhere(_grid == 'S')
        assert len(start_ind) == 1
        self._init_state = self._get_coords(start_ind[0])

        # grid goals, for performance logging during training
        grid_goals_ind = np.argwhere(_grid == 'E')
        self._grid_goals = np.asarray(list(map(self._get_coords, grid_goals_ind)))
        # clean 'E'
        _grid_flatten[_grid_flatten == 'E'] = ' '

        # target goals spreads across E, uniform distribution for the goal is defined to be normalize(X_E) the indicator function
        self._target_goals_ind = np.argwhere(_grid == 'G')
        self._non_target_goals_ind = np.argwhere(_grid == ' ')
        # clean 'G'
        _grid_flatten[_grid_flatten == 'G'] = ' '

        # clean 'S'
        _grid_flatten[_grid_flatten == 'S'] = ' '
        self.grid = _grid != ' '  # boolean mask, True for blocks ('+', '|', '-')
        self._free_ind = np.argwhere(_grid == ' ')  # np.array of indices with no blocks

        self.reset()
        assert not self._is_wall(self._get_obs())

    @property
    def target_percentage(self):
        return len(self._target_goals_ind) / (len(self._target_goals_ind) + len(self._non_target_goals_ind))

    def set_goal(self, goal):
        assert not self._is_wall(goal)
        self.goal = goal

    def sample_goals(self, mode, num_samples):
        if mode is None:
            num_target_goals = max(int(num_samples * self.target_percentage), 1)
            _sample_ind = np.random.choice(len(self._target_goals_ind), num_target_goals, replace=True)
            samples = list(map(self._get_coords_uniform_noise, self._target_goals_ind[_sample_ind]))
            _sample_ind = np.random.choice(len(self._non_target_goals_ind), num_samples - num_target_goals, replace=True)
            samples.extend(list(map(self._get_coords_uniform_noise, self._non_target_goals_ind[_sample_ind])))

        elif mode == 'target':
            _sample_ind = np.random.choice(len(self._target_goals_ind), num_samples, replace=True)
            samples = list(map(self._get_coords_uniform_noise, self._target_goals_ind[_sample_ind]))
        elif mode == 'non_target':
            _sample_ind = np.random.choice(len(self._non_target_goals_ind), num_samples, replace=True)
            samples = list(map(self._get_coords_uniform_noise, self._non_target_goals_ind[_sample_ind]))
        else:
            raise RuntimeError

        return np.asarray(samples)

    def _is_wall(self, obs):
        ind = self._get_index(obs)
        return self.grid[ind[0], ind[1]]

    def _get_coords(self, ind):
        return ((np.asarray(ind) + 0.5) / self.grid_size * 2 - 1).astype(np.float32)

    def _get_coords_uniform_noise(self, ind):
        return (((np.asarray(ind) + np.random.uniform(low=.05, high=.95, size=np.shape(ind))) / self.grid_size) * 2 - 1).astype(np.float32)

    def _get_index(self, coords):
        return np.clip((np.asarray(coords) + 1) * 0.5 * self.grid_size + 0, 0, self.grid_size-1).astype(np.int8)

    def _set_state(self, ob):
        self.state = np.clip(ob, self.obs_low, self.obs_high)
        return self.state

    def _get_obs(self):
        return self.state

    def get_achieved_goal(self, next_obs):
        return next_obs

    def reward(self, obs, act, next_obs, goal=None):
        if goal is None:
            goal = self.goal
        if self.reward_str == 'L1':
            _reward = - np.sum(np.abs(next_obs - goal), axis=-1)
        elif self.reward_str == 'L2':
            _reward = - np.linalg.norm(next_obs - goal, axis=-1)
        elif self.reward_str == 'sparse':
            _reward = (np.linalg.norm(next_obs - goal, axis=-1) < 0.1).astype(int)
        else:
            raise ValueError
        return _reward

    def reset(self):
        self.state = self._init_state
        self.goal = None
        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.act_low, self.act_high)
        start_obs = obs = self._get_obs()
        assert not self._is_wall(start_obs)

        # collision detection
        # collision = False
        for _ in range(self.num_substeps):
            next_obs = obs + action * self.ddt
            if self._is_wall(next_obs):
                # collision = True
                break

            obs = next_obs

        obs = self._set_state(obs)
        assert not self._is_wall(obs)
        reward = self.reward(start_obs, action, obs) # - 0.1 * int(collision)  # penalty for collision
        done = False
        info = {}

        return obs, reward, done, info

    @property
    def init_obs(self):
        return self._init_state

    @property
    def eval_goals(self):
        return self._grid_goals


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
    env = ParticleMazeEnv()
    # env.set_goal(env.observation_space.sample())
    env.sample_grid_goals(2)
    # env.step(env.action_space.sample())


