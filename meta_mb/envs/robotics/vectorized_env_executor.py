import numpy as np
import pickle as pickle
import ray

from meta_mb.agents.remote_env import RemoteEnv


class IterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.
    """

    def __init__(self, env_pickled, num_rollouts, max_path_length):
        self._num_envs = num_rollouts
        self.envs = np.asarray([pickle.loads(env_pickled) for _ in range(self._num_envs)])
        self.ts = np.zeros(self._num_envs, dtype='int')  # time steps
        self.max_path_length = max_path_length

    def step(self, act_na):
        """
        Steps the wrapped environments with the provided actions

        Args:
            act_na (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert len(act_na) == self.num_envs

        all_results = [env.step(a) for (a, env) in zip(act_na, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = map(list, zip(*all_results))
        if isinstance(obs[0], dict):
            obs = list(map(lambda obs_dict: obs_dict['observation'], obs))

        # reset env when done or max_path_length reached
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones).tolist()

        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset(self.envs[i].goal) # assume that env preserves goal state
            if isinstance(obs[i], dict):
                obs[i] = obs[i]['observation']
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def reset(self, goal_ng):
        init_ob_no = [env.reset(goal) for goal, env in zip(goal_ng, self.envs)]
        if isinstance(init_ob_no[0], dict):
            init_ob_no = list(map(lambda obs_dict: obs_dict['observation'], init_ob_no))
        self.ts[:] = 0
        return init_ob_no

    @property
    def num_envs(self):
        return self._num_envs


class ParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    Args:
        env (meta_mb.meta_envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                             the respective environment is reset
    """

    def __init__(self, env_pickled, n_parallel, num_rollouts, max_path_length):
        n_parallel = min(n_parallel, num_rollouts)
        assert num_rollouts % n_parallel == 0
        self._envs_per_worker = num_rollouts // n_parallel
        self._num_envs = num_rollouts
        self._num_workers = n_parallel

        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.workers = [RemoteEnv.remote(env_pickled, self._envs_per_worker, max_path_length, seed) \
                        for seed in seeds]

    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (np.array): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        # split list of actions in list of list of actions per meta tasks
        # chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        act_per_worker = np.split(actions, self._num_workers)
        futures = [worker.step.remote(act) for act, worker in zip(act_per_worker, self.workers)]
        all_results = ray.get(futures)
        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*all_results))

        return obs, rewards, dones, env_infos

    def reset(self, goal_ng):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        goals_per_worker = np.split(goal_ng, self._num_workers)
        futures = [worker.reset.remote(goals) for goals, worker in zip(goals_per_worker, self.workers)]
        all_results = ray.get(futures)
        return sum(all_results, [])

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs
