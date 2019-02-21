from meta_policy_search.envs.base import MetaEnv

import numpy as np
from gym.spaces import Box


class MetaPointEnvCorner(MetaEnv):
    """
    Simple 2D point meta environment. Each meta-task corresponds to a different goal / corner
    (one of the 4 points (-2,-2), (-2, 2), (2, -2), (2,2)) which are sampled with equal probability
    """

    def __init__(self, reward_type='sparse', sparse_reward_radius=0.5):
        assert reward_type in ['dense', 'dense_squared', 'sparse']
        self.reward_type = reward_type
        print("Point Env reward type is", reward_type)
        self.sparse_reward_radius = sparse_reward_radius
        self.corners = [np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = Box(low=-0.2, high=0.2, shape=(2,))

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment
        Returns:
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        prev_state = self._state
        self._state = prev_state + np.clip(action, -0.2, 0.2)
        reward = self.reward(prev_state, action, self._state)
        done = False # self.done(self._state)
        next_observation = np.copy(self._state)
        return next_observation, reward, done, {}

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._state = np.random.uniform(-0.2, 0.2, size=(2,))
        observation = np.copy(self._state)
        return observation

    def done(self, obs):
        if obs.ndim == 1:
            return self.done(np.array([obs]))
        elif obs.ndim == 2:
            goal_distance = np.linalg.norm(obs - self.goal[None,:], axis=1)
            return np.max(self._state) > 3

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 2:
            goal_distance = np.linalg.norm(obs_next - self.goal[None,:], axis=1)[0]
            if self.reward_type == 'dense':
                return - goal_distance
            elif self.reward_type == 'dense_squared':
                return - goal_distance**2
            elif self.reward_type == 'sparse':
                dist_from_start = np.linalg.norm(obs_next, ord=1, axis=1)[0]
                if dist_from_start < self.sparse_reward_radius:
                    return 0
                dists = [np.linalg.norm(obs_next - corner[None, :], axis=1) for corner in self.corners]
                if np.min(goal_distance) == min(dists):
                    return np.linalg.norm(obs - self.goal[None,:], axis=1)[0] - goal_distance
                return 0
                # return np.maximum(self.sparse_reward_radius - goal_distance, 0)

        elif obs_next.ndim == 1:
            return self.reward(np.array([obs]), np.array([act]), np.array([obs_next]))
        else:
            raise NotImplementedError

    def log_diagnostics(self, *args):
        pass

    def sample_tasks(self, n_tasks):
        return [self.corners[idx] for idx in np.random.choice(range(len(self.corners)), size=n_tasks)]

    def set_task(self, task):
        self.goal = task

    def get_task(self):
        return self.goal

if __name__ == "__main__":
    env = MetaPointEnvCorner()
    task = env.sample_tasks(10)
    print(task[0])
    while True:
        env.set_task(task[0])
        env.reset()
        done = False
        i = 0
        t_r = 0
        while not done:
            obs, reward, done, _ = env.step(env.action_space.sample())  # take a random action
            t_r += reward
            i += 1
            if reward > 0:
                print(obs)
                break
            if i > 200:
                print(obs)
                break
        print(i, t_r)