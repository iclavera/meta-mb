import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
import copy
from pyprind import ProgBar


class IterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    Args:
        env (meta_mb.meta_envs.base.MetaEnv): meta environment object
        meta_batch_size (int): number of meta tasks
        envs_per_task (int): number of environments per meta task
        max_path_length (int): maximum length of sampled environment paths - if the max_path_length is reached,
                               the respective environment is reset
    """

    def __init__(self, env, num_rollouts, max_path_length):
        self._num_envs = num_rollouts
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self._num_envs)])
        self.ts = np.zeros(len(self.envs), dtype='int')  # time steps
        self.max_path_length = max_path_length
        self._buffer = None

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

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

    def _reset(self, i):
        if self._buffer is None:
            return self.envs[i].reset()

        else:
            idx = np.random.randint(len(self._buffer['observations']))
            return self.envs[i].reset_from_obs(self._buffer['observations'][idx])

    def reset(self, buffer=None):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        self._buffer = buffer
        if self._buffer is None:
            obses = [env.reset() for env in self.envs]
        else:
            idxs = np.random.randint(0, len(self._buffer['observations']), size=self.num_envs)
            obses = [env.reset_from_obs(self._buffer['observations'][idx]) for idx, env in zip(idxs, self.envs)]
        self.ts[:] = 0
        return obses

    def reset_from_obs_hard(self, observations):
        assert observations.shape[0] == self.num_envs
        obses = [env.reset_from_obs_hard(obs) for env, obs in zip(self.envs, observations)]
        self.ts[:] = 0
        return obses

    def reset_hard(self):
        obses = [env.reset_hard() for env in self.envs]
        self.ts[:] = 0
        return obses

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
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

    def __init__(self, env, n_parallel, num_rollouts, max_path_length):
        assert num_rollouts % n_parallel == 0
        self.envs_per_proc = int(num_rollouts/n_parallel)
        self._num_envs = n_parallel * self.envs_per_proc
        self.n_parallel = n_parallel
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), self.envs_per_proc, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_meta_task = chunks(actions, self.envs_per_proc)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self, buffer=None):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        if buffer is not None:
            raise NotImplementedError
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker

        Args:
            tasks (list): list of the tasks for each worker
        """
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs


def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == 'set_task':
            for env in envs:
                env.set_task(data)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError


class ParallelActionDerivativeExecutor(object):
    """
    Compute ground truth derivatives.
    """
    def __init__(self, env, n_parallel, horizon, batch_size, eps, discount=1, verbose=False):
        assert discount == 1  # only support discount == 1
        self.n_parallel = n_parallel
        self.horizon = horizon
        self.batch_size = batch_size
        self.eps = eps
        self.action_space_dims = action_space_dims = env.action_space.shape[0]
        num_tasks = horizon * action_space_dims
        assert num_tasks % n_parallel == 0
        num_tasks_per_worker = num_tasks // n_parallel # technically num_tasks_per_worker *= batch_size because each worker has batch_size envs
        array_idx_start_flatten = np.arange(0, horizon*action_space_dims, num_tasks_per_worker)
        array_idx_end_flatten = array_idx_start_flatten + num_tasks_per_worker

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(
                target=deriv_worker,
                args=(work_remote, remote, pickle.dumps(env), eps,
                      horizon, batch_size, action_space_dims, discount,
                      idx_start_flatten, idx_end_flatten, seed, verbose),
            ) for (work_remote, remote, idx_start_flatten, idx_end_flatten, seed) \
            in zip(self.work_remotes, self.remotes, array_idx_start_flatten, array_idx_end_flatten, seeds)
        ]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def get_derivative(self, tau, init_obs_array=None):
        """
        Assume s_0 is the reset state.
        :param tau: (horizon, batch_size, action_space_dims)
        :tf_loss: scalar Tensor R
        :return: dR/da_i for i in range(action_space_dims)
        """
        self.remotes[0].send(('compute_old_return_array', tau, init_obs_array))
        old_return_array = self.remotes[0].recv()
        for remote in self.remotes:
            remote.send(('compute_delta_return_cubic', tau, init_obs_array, old_return_array))

        delta_return_cubic = np.zeros((self.horizon, self.batch_size, self.action_space_dims)) #sum([np.asarray(remote.recv()) for remote in self.remotes])
        for remote in self.remotes:
            for idx_h, idx_a, delta_return_arr in remote.recv():
                delta_return_cubic[idx_h, :, idx_a] = delta_return_arr

        return delta_return_cubic/self.eps, old_return_array

def deriv_worker(remote, parent_remote, env_pickle, eps,
           horizon, batch_size, action_space_dims, discount,
           idx_start_flatten, idx_end_flatten, seed, verbose):

    # batch_size means the num_rollouts in teh original env executors, and it means number of experts
    # when the dynamics model is ground truth

    print('deriv worker starts...')

    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(batch_size)]
    np.random.seed(seed)

    while True:
        # receive command and data from the remote
        cmd, *data = remote.recv()
        # do a step in each of the environment of the worker
        if cmd == 'compute_delta_return_cubic':
            tau, init_obs_array, old_return_array = data
            new_tau = tau.copy()
            # tau = (horizon, batch_size, action_space_dims)
            # init_obs = (batch_size, action_space_dims,)

            # delta_return_cubic = np.zeros((horizon, batch_size, action_space_dims))
            delta_return_cubic = []
            if verbose: pbar = ProgBar(idx_end_flatten - idx_start_flatten)
            for idx_flatten in range(idx_start_flatten, idx_end_flatten):
                idx_horizon, idx_action_space_dims = idx_flatten // action_space_dims, idx_flatten % action_space_dims
                # delta = np.zeros((horizon, batch_size, action_space_dims))
                # delta[idx_horizon, :, idx_action_space_dims] = eps
                # new_tau = tau + delta
                # perturb
                new_tau[idx_horizon, :, idx_action_space_dims] += eps
                delta_return_array = []  # (batch_size,)

                for idx_batch, env, old_return in zip(range(batch_size), envs, old_return_array):
                    # reset environment
                    if init_obs_array is None:
                        _ = env.reset_hard()
                    else:
                        _ = env.reset_from_obs_hard(init_obs_array[idx_batch])

                    # compute new return with discount factor = 1
                    new_return = sum([env.step(act)[1] for act in new_tau[:, idx_batch, :]])
                    delta_return_array.append(new_return - old_return)

                # delta_return_cubic[idx_horizon, :, idx_action_space_dims] = delta_return_array
                delta_return_cubic.append((idx_horizon, idx_action_space_dims, delta_return_array))
                # revert to old tau
                new_tau[idx_horizon, :, idx_action_space_dims] -= eps
                if verbose: pbar.update(1)

            if verbose: pbar.stop()

            remote.send(delta_return_cubic)

        elif cmd == 'compute_old_return_array':
            tau, init_obs_array = data
            old_return_array = []

            for idx_batch, env in zip(range(batch_size), envs):
                # reset environment
                if init_obs_array is None:
                    _ = env.reset_hard()
                else:
                    _ = env.reset_from_obs_hard(init_obs_array[idx_batch])
                old_return = sum([env.step(act)[1] for act in tau[:, idx_batch, :]])
                old_return_array.append(old_return)

            remote.send(old_return_array)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError
