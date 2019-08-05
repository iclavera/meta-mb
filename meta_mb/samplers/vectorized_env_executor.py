import numpy as np
from meta_mb.logger import logger
import time
from collections import OrderedDict
from meta_mb.policies.np_linear_policy import LinearPolicy
from meta_mb.optimizers.gt_optimizer import GTOptimizer
import pickle as pickle
from multiprocessing import Process, Pipe
import copy


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

    def reset_from_pickles(self, pickled_states):
        for env_state, env in zip(pickled_states['envs'], self.envs):
            # TODO: env.sim.reset()??
            env.sim.set_state(env_state)
            # TODO: env.sim.forward()??

        # self.envs = [pickle.loads(env) for env in pickled_states['envs']]
        # self.ts = pickle.loads(pickled_states['ts'])
        return None

    def get_pickles(self):
        # return dict(envs=[pickle.dumps(env) for env in self.envs], ts=pickle.dumps(self.ts))
        return dict(envs=[pickle.dumps(env.sim.get_state()) for env in self.envs])

    # def reset_from_obs_hard(self, observations):
    #     assert observations.shape[0] == self.num_envs
    #     obses = [env.reset_from_obs_hard(obs) for env, obs in zip(self.envs, observations)]
    #     self.ts[:] = 0
    #     return obses

    # def reset_hard(self, init_obs_array=None):
    #     if init_obs_array is None:
    #         obses = [env.reset_hard() for env in self.envs]
    #     else:
    #         obses = [env.reset_hard(init_obs) for env, init_obs in zip(self.envs, init_obs_array)]
    #     self.ts[:] = 0
    #     return obses

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs


def chunks(l, n):
    return [l[x: x+n] for x in range(0, len(l), n)]


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
        n_parallel = min(n_parallel, num_rollouts)
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
        # chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
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

    # def reset_hard(self, init_obs_array=None):
    #     if init_obs_array is None:
    #         for remote in self.remotes:
    #             remote.send(('reset_hard', None))
    #     else:
    #         init_obs_per_meta_task = chunks(init_obs_array, self.envs_per_proc)
    #         for remote, init_obs in zip(self.remotes, init_obs_per_meta_task):
    #             remote.send(('reset_hard', init_obs))
    #
    #     return sum([remote.recv() for remote in self.remotes], [])

    def reset_from_pickles(self, pickled_states):
        # for remote, envs, ts in zip(self.remotes, chunks(pickled_states['envs'], self.envs_per_proc), chunks(pickled_states['ts'], self.envs_per_proc)):
        #     remote.send(('reset_from_pickles', dict(envs=envs, ts=ts)))
        assert isinstance(pickled_states['envs'], list)
        assert len(pickled_states['envs']) == self.n_parallel
        for remote, env_state in zip(self.remotes, pickled_states['envs']):
            remote.send(('reset_from_pickles', dict(envs=env_state)))

        for remote in self.remotes:
            remote.recv()
        # return sum([remote.recv() for remote in self.remotes], [])

    def get_pickles(self):
        for remote in self.remotes:
            remote.send(('get_pickles', None))
        # return: [(pickled_envs, pickled_ts) for remote in self.remotes]
        pickled_states_list = [remote.recv() for remote in self.remotes]
        env_states = sum([pickled_states['env'] for pickled_states in pickled_states_list], [])
        # ts = sum([pickled_states['ts'] for pickled_states in pickled_states_list], [])
        # return dict(envs=envs, ts=ts)
        return dict(envs=env_states)

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

        elif cmd == 'reset_from_pickles':
            # only one env repeated for one worker!!
            env_state = pickle.loads(data['envs'])
            for env in envs:
                # TODO: env.sim.reset()?
                env.sim.set_state(env_state)
                env.sim.forward()

            # ts = pickle.loads(data['ts'])  # FIXME: dones are ignored!
            # obs = [env._get_obs() for env in envs]
            remote.send(None)

        elif cmd == 'get_pickles':
            # remote.send(dict(envs=[pickle.dumps(env) for env in envs], ts=pickle.dumps(ts)))
            remote.send(dict(envs=[pickle.dumps(env.sim.get_state()) for env in envs]))

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
                target=act_deriv_worker,
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

    def get_derivative(self, tau):
        """
        Assume s_0 is the reset state.
        :param tau: (horizon, batch_size, action_space_dims)
        :tf_loss: scalar Tensor R
        :return: dR/da_i for i in range(action_space_dims)
        """
        self.remotes[0].send(('compute_old_return_array', tau))
        old_return_array = self.remotes[0].recv()
        for remote in self.remotes:
            remote.send(('compute_delta_return_cubic', tau, old_return_array))

        delta_return_cubic = np.zeros((self.horizon, self.batch_size, self.action_space_dims)) #sum([np.asarray(remote.recv()) for remote in self.remotes])
        for remote in self.remotes:
            for idx_h, idx_a, delta_return_arr in remote.recv():
                delta_return_cubic[idx_h, :, idx_a] = delta_return_arr

        return delta_return_cubic/self.eps, old_return_array


def act_deriv_worker(remote, parent_remote, env_pickle, eps,
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
            tau, old_return_array = data
            new_tau = tau.copy()
            # tau = (horizon, batch_size, action_space_dims)
            # init_obs = (batch_size, action_space_dims,)

            # delta_return_cubic = np.zeros((horizon, batch_size, action_space_dims))
            delta_return_cubic = []
            # if verbose: pbar = ProgBar(idx_end_flatten - idx_start_flatten)
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
                    _ = env.reset()

                    # compute new return with discount factor = 1
                    new_return = sum([env.step(act)[1] for act in new_tau[:, idx_batch, :]])
                    delta_return_array.append(new_return - old_return)

                # delta_return_cubic[idx_horizon, :, idx_action_space_dims] = delta_return_array
                delta_return_cubic.append((idx_horizon, idx_action_space_dims, delta_return_array))
                # revert to old tau
                new_tau[idx_horizon, :, idx_action_space_dims] -= eps
                # if verbose: pbar.update(1)

            # if verbose: pbar.stop()

            remote.send(delta_return_cubic)

        elif cmd == 'compute_old_return_array':
            tau, = data
            old_return_array = []

            for idx_batch, env in zip(range(batch_size), envs):
                # reset environment
                _ = env.reset()
                old_return = sum([env.step(act)[1] for act in tau[:, idx_batch, :]])
                old_return_array.append(old_return)

            remote.send(old_return_array)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError


class ParallelPolicyGradUpdateExecutor(object):
    """
    Compute ground truth derivatives.
    """
    def __init__(self, env, n_parallel, num_rollouts, horizon, eps,
                 opt_learning_rate, num_opt_iters,
                 discount=1, verbose=False):
        n_parallel = min(n_parallel, num_rollouts)
        self.n_parallel = n_parallel
        self.num_rollouts = num_rollouts
        assert num_rollouts % n_parallel == 0
        n_envs_per_proc = num_rollouts // n_parallel
        action_space_dims = env.action_space.shape[0]
        obs_space_dims = env.observation_space.shape[0]

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(
                target=policy_gard_update_worker,
                args=(work_remote, remote, pickle.dumps(env), eps,
                      horizon, n_envs_per_proc, obs_space_dims, action_space_dims, discount,
                      opt_learning_rate, num_opt_iters,
                      seed, verbose),
            ) for (work_remote, remote, seed) \
            in zip(self.work_remotes, self.remotes, seeds)
        ]

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def do_gradient_steps(self):
        """
        Assume s_0 is the reset state.
        :param tau: (horizon, batch_size, action_space_dims)
        :tf_loss: scalar Tensor R
        :return: dR/da_i for i in range(action_space_dims)
        """
        for remote in self.remotes:
            remote.send(('do_gradient_steps',))

        global_info = dict(old_return=[], grad_norm_W=[], grad_norm_b=[], norm_W=[], norm_b=[])

        for remote in self.remotes:
            info = remote.recv()
            for k, v in info.items():  # (n_envs_per_proc, num_opt_iters)
                global_info[k].extend(v)

        for k, v in global_info.items():
            global_info[k] = np.asarray(v)

        return global_info

    def get_param_values_first_rollout(self):
        self.remotes[0].send(('get_param_values',))
        params = self.remotes[0].recv()
        W, b = params[0]
        return W, b


def policy_gard_update_worker(remote, parent_remote, env_pickle, eps,
                              horizon, n_envs, obs_dim, action_dim, discount,
                              opt_learning_rate, num_opt_iters,
                              seed, verbose):

    # batch_size means the num_rollouts in teh original env executors, and it means number of experts
    # when the dynamics model is ground truth

    print('deriv worker starts...')

    parent_remote.close()

    # FIXME: only one env is needed since the worker is stateless!
    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)
    # linear policy with tanh output activation
    policies = [LinearPolicy(obs_dim=obs_dim, action_dim=action_dim, output_nonlinearity=np.tanh) for _ in range(n_envs)]
    optimizers_W = [GTOptimizer(alpha=opt_learning_rate) for _ in range(n_envs)]
    optimizers_b = [GTOptimizer(alpha=opt_learning_rate) for _ in range(n_envs)]

    def compute_sum_rewards(env, policy):
        obs = env.reset()
        sum_rewards = 0
        for t in range(horizon):
            action, _ = policy.get_action(obs)
            obs, reward, _, _ = env.step(action)
            sum_rewards += discount ** t * reward
        return sum_rewards

    while True:
        # receive command and data from the remote
        cmd, *data = remote.recv()
        # do a step in each of the environment of the worker
        if cmd == 'do_gradient_steps':
            # info array over envs
            info = dict(old_return=[], grad_norm_W=[], grad_norm_b=[], norm_W=[], norm_b=[])

            for env, policy, optimizer_W, optimizer_b in zip(envs, policies, optimizers_W, optimizers_b):

                # info array over opt iters
                local_info = dict(old_return=[], grad_norm_W=[], grad_norm_b=[], norm_W=[], norm_b=[])

                for _ in range(num_opt_iters):
                    old_return = compute_sum_rewards(env, policy)
                    W, b = policy.get_param_values()

                    # optimize W
                    grad_W = np.zeros_like(W)
                    for i in range(action_dim):
                        for j in range(obs_dim):
                            policy.perturb_W(i, j, eps)
                            new_return = compute_sum_rewards(env, policy)
                            grad_W[i, j] = (new_return - old_return) / eps
                            policy.perturb_W(i, j, -eps)  # recover

                    # optimize b
                    grad_b = np.zeros_like(b)
                    for i in range(action_dim):
                        policy.perturb_b(i, eps)
                        new_return = compute_sum_rewards(env, policy)
                        grad_b[i] = (new_return - old_return) / eps
                        policy.perturb_b(i, -eps)

                    # update policy params
                    policy.add_to_params(dict(W=optimizer_W.compute_delta_var(grad_W), b=optimizer_b.compute_delta_var(grad_b)))

                    # collect info
                    local_info['old_return'].append(old_return)
                    local_info['grad_norm_W'].append(np.linalg.norm(grad_W))
                    local_info['grad_norm_b'].append(np.linalg.norm(grad_b))
                    local_info['norm_W'].append(np.linalg.norm(W))
                    local_info['norm_b'].append(np.linalg.norm(b))

                # collect info
                info['old_return'].append(local_info['old_return'])  # (n_envs, num_opt_iters)
                info['grad_norm_W'].append(local_info['grad_norm_W'])
                info['grad_norm_b'].append(local_info['grad_norm_b'])
                info['norm_W'].append(local_info['norm_W'])
                info['norm_b'].append(local_info['norm_b'])

            remote.send(info)

        elif cmd == 'get_param_values':
            remote.send([policy.get_param_values() for policy in policies])

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            print(f'receiving command {cmd}')
            raise NotImplementedError


class ParallelCollocationExecutor(object):
    """
    Compute ground truth derivatives.
    """
    def __init__(self, env, n_parallel, horizon, eps,
                 discount, verbose=False):
        self.n_parallel = n_parallel
        self.horizon = horizon
        assert horizon % n_parallel == 0
        self.n_envs_per_proc = n_envs_per_proc = horizon // n_parallel
        action_space_dims = env.action_space.shape[0]
        obs_space_dims = env.observation_space.shape[0]

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])

        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(
                target=collocation_worker,
                args=(work_remote, remote, pickle.dumps(env), eps,
                      n_envs_per_proc, obs_space_dims, action_space_dims, discount,
                      seed, verbose),
            ) for (work_remote, remote, seed) \
            in zip(self.work_remotes, self.remotes, seeds)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def do_gradient_steps(self, s_array_stacked, a_array_stacked, lmbda):
        s_array_list = np.split(s_array_stacked, self.n_parallel)
        a_array_list = np.split(a_array_stacked, self.n_parallel)
        s_last_list = [s_array_stacked[t] for t in range(self.n_envs_per_proc, self.horizon, self.n_envs_per_proc)] + [None]

        for remote, s_array, a_array, s_last in zip(self.remotes, s_array_list, a_array_list, s_last_list):
            remote.send(('compute_gradients', dict(obs=s_array, act=a_array), s_last, lmbda))

        results = [remote.recv() for remote in self.remotes]
        grad_s_stacked, grad_a_stacked = map(lambda x: np.concatenate(x, axis=0), zip(*results))

        return grad_s_stacked, grad_a_stacked  # (horizon, space_dim)


def collocation_worker(remote, parent_remote, env_pickle, eps,
                       n_envs, obs_dim, act_dim, discount,
                       seed, verbose):

    # batch_size means the num_rollouts in the original env executors, and it means number of experts
    # when the dynamics model is ground truth

    print('collocation state worker starts...')

    parent_remote.close()

    env = pickle.loads(env_pickle)
    np.random.seed(seed)

    def step_wrapper(s, a):
        _ = env.reset_from_obs(s)
        s_next, _, _, _ = env.step(a)
        return s_next

    def df_ds(s, a): # s must not be owned by multiple workers
        """
        :param s: (act_dim,)
        :param a: (obs_dim,)
        :return: (obs_dim, obs_dim)
        """
        old_s_next = step_wrapper(s, a)
        grad_s = np.zeros((obs_dim, obs_dim))
        for idx in range(obs_dim):  # compute grad[:, idx]
            s[idx] += eps
            new_s_next = step_wrapper(s, a)
            s[idx] -= eps
            grad_s[idx] = (new_s_next - old_s_next) / eps

        return grad_s

    def df_da(s, a):
        """
        :param s: (act_dim.)
        :param a: (obs_dim,)
        :return: (obs_dim, act_dim)
        """
        old_s_next = step_wrapper(s, a)
        grad_a = np.zeros((obs_dim, act_dim))
        for idx in range(act_dim): # compute grad[:, idx]
            a[idx] += eps
            new_s_next = step_wrapper(s, a)
            a[idx] -= eps
            grad_a[:, idx] = (new_s_next - old_s_next) / eps
        return grad_a

    def dr_ds(s, a):
        return env.deriv_reward_obs(obs=s, acts=a)

    def dr_da(s, a):
        return env.deriv_reward_act(obs=s, act=a)

    while True:
        # receive command and data from the remote
        cmd, *data = remote.recv()
        # do a step in each of the environment of the worker
        if cmd == 'compute_gradients':
            inputs_dict, s_last, lmbda = data
            s_array, a_array = inputs_dict['obs'], inputs_dict['act']
            # to compute grad for s[1:t], a[1:t], need s[1:t+1], a[1:t]
            # s_last = s[t+1]
            assert s_array.shape == (n_envs, obs_dim)
            assert a_array.shape == (n_envs, act_dim)
            f_array = [step_wrapper(s_array[idx], a_array[idx]) for idx in range(n_envs)]  # array of f(s, a)

            # compute dl_ds
            grad_s_stacked, grad_a_stacked = np.zeros((n_envs, obs_dim)), np.zeros((n_envs, act_dim))

            for t in range(n_envs):
                s, a = s_array[t], a_array[t]
                _grad_s = -discount**t * dr_ds(s, a) + lmbda * (s - f_array[t-1])
                _grad_a = -discount**t * dr_da(s, a)
                if t != n_envs-1:
                    s_next = s_array[t+1]
                    _grad_s += -lmbda * np.matmul(df_ds(s, a).T, s_next - f_array[t])
                    _grad_a += -lmbda * np.matmul(df_da(s, a).T, s_next - f_array[t])
                elif s_last is not None:
                    s_next = s_last
                    _grad_s += -lmbda * np.matmul(df_ds(s, a).T, s_next - f_array[t])
                    _grad_a += -lmbda * np.matmul(df_da(s, a).T, s_next - f_array[t])
                else:
                    pass
                grad_s_stacked[t, :] = _grad_s
                grad_a_stacked[t, :] = _grad_a

            remote.send((grad_s_stacked, grad_a_stacked))

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            print(f'receiving command {cmd}')
            raise NotImplementedError


class ParallelDDPExecutor(object):
    """
    Compute ground truth derivatives.
    """
    def __init__(self, env, n_parallel, horizon, eps,
                 mu_min=1e-6, mu_max=1e10, mu_init=0, delta_0=1.5, alpha_decay_factor=10.0,
                 c_1=1e-5, max_forward_iters=30, max_backward_iters=30, use_hessian_f=False, verbose=False):
        self._env = env
        self.n_parallel = n_parallel
        self.horizon = horizon
        self.x_array, self.u_array = None, None
        self.J_val = None
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_init = mu_init
        self.alpha_decay_factor = alpha_decay_factor
        self.delta_0 = delta_0
        self.c_1 = c_1
        self.max_forward_iters = max_forward_iters
        self.max_backward_iters = max_backward_iters
        self.use_hessian_f = use_hessian_f
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        if use_hessian_f:
            self.fn_str_array = ['jac_f_x', 'jac_f_u', 'hessian_f_xx', 'hessian_f_uu', 'hessian_f_ux']
        else:
            self.fn_str_array = ['jac_f_x', 'jac_f_u']

        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(
                target=DDP_worker,
                args=(work_remote, remote, pickle.dumps(env), eps,
                      horizon, self.obs_dim, self.act_dim,
                      seed, verbose),
            ) for (work_remote, remote, seed) \
            in zip(self.work_remotes, self.remotes, seeds)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()

        for remote in self.work_remotes:
            remote.close()

    def _compute_deriv(self, inputs_dict):
        # for fn_str, remote in zip(self.fn_str_array, self.remotes):
        #     remote.send((fn_str, inputs_dict))
        for i, fn_str in enumerate(self.fn_str_array):
            self.remotes[i%self.n_parallel].send((fn_str, inputs_dict))

        dl = self._env.dl_dict(inputs_dict).values()
        df = [self.remotes[i%self.n_parallel].recv() for i in range(len(self.fn_str_array))]
        return dl, df

    def _f(self, x, u):
        _ = self._env.reset_from_obs(x)
        x_prime, reward, _, _ = self._env.step(u)
        return x_prime, reward

    def update_x_u_for_one_step(self):
        x_array, u_array, J_val = self.x_array, self.u_array, self.J_val
        backward_accept, forward_accept = False, False

        assert x_array.shape == (self.horizon, self.obs_dim)
        assert u_array.shape == (self.horizon, self.act_dim)

        """
        Derivatives
        """
        dl, df = self._compute_deriv(dict(obs=x_array, act=u_array))
        l_x, l_u, l_xx, l_uu, l_ux = dl  # length = horizon
        if self.use_hessian_f:
            f_x, f_u, f_xx, f_uu, f_ux = df  # length = horizon - 1
        else:
            f_x, f_u = df
            f_xx, f_uu, f_ux = None, None, None
        """
        Backward Pass
        """
        backward_pass_counter = 0
        while not backward_accept and backward_pass_counter < self.max_backward_iters and self.mu <= self.mu_max:
            # reset
            V_prime_x, V_prime_xx = l_x[-1], l_xx[-1]
            open_loop_array, feedback_gain_array = [], []
            delta_J_1, delta_J_2 = 0, 0

            try:
                # backward pass
                for i in range(self.horizon-2, -1, -1):
                    # logger.log(f'at {backward_pass_counter}-th backward pass, horizon {i}, with mu = {self.mu}')

                    # compute Q
                    V_prime_xx_reg = V_prime_xx + self.mu * np.identity(self.obs_dim)
                    if self.use_hessian_f:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx_reg @ f_u[i] + np.tensordot(V_prime_x, f_uu[i], axes=1)
                    else:
                        Q_uu = l_uu[i] + f_u[i].T @ V_prime_xx_reg @ f_u[i]

                        # Q_uu_no_reg = l_uu[i] + f_u[i].T @ V_prime_xx @ f_u[i]
                        # logger.log('Q_uu_no_reg min eigen value', np.min(np.linalg.eigvals(Q_uu_no_reg)))
                        # logger.log('Q_uu min eigen value', np.min(np.linalg.eigvals(Q_uu)))

                    if not np.allclose(Q_uu, Q_uu.T):
                        print(Q_uu)
                        raise RuntimeError

                    # if np.all(np.linalg.eigvals(Q_uu) > 0):
                    _ = np.linalg.cholesky(Q_uu)
                    # Q_uu is PD, decrease mu
                        # self._decrease_mu()
                    # else:

                    Q_x = l_x[i] + f_x[i].T @ V_prime_x
                    Q_u = l_u[i] + f_u[i].T @ V_prime_x
                    if self.use_hessian_f:
                        Q_xx = l_xx[i] + f_x[i].T @ V_prime_xx @ f_x[i] + np.tensordot(V_prime_x, f_xx[i], axes=1)
                        Q_ux = l_ux[i] + f_u[i].T @ V_prime_xx_reg @ f_x[i] + np.tensordot(V_prime_x, f_ux[i], axes=1)
                    else:
                        Q_xx = l_xx[i] + f_x[i].T @ V_prime_xx @ f_x[i]
                        Q_ux = l_ux[i] + f_u[i].T @ V_prime_xx_reg @ f_x[i]

                    # compute control matrices
                    # Q_uu_inv = np.linalg.inv(Q_uu)
                    # k = - Q_uu_inv @ Q_u  # k
                    # K = - Q_uu_inv @ Q_ux  # K
                    k = - np.linalg.solve(Q_uu, Q_u)
                    K = - np.linalg.solve(Q_uu, Q_ux)
                    open_loop_array.append(k)
                    feedback_gain_array.append(K)
                    delta_J_1 += k.T @ Q_u
                    delta_J_2 += k.T @ Q_uu @ k

                    # prepare for next i
                    # V_prime_x = Q_x + Q_u @ feedback_gain
                    # V_prime_xx = Q_xx + Q_ux.T @ feedback_gain
                    V_prime_x = Q_x + K.T @ Q_uu @ k + K.T @ Q_u + Q_ux.T @ k
                    V_prime_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
                    V_prime_xx = (V_prime_xx + V_prime_xx.T) * 0.5

                # self._decrease_mu()
                backward_accept = True

            except np.linalg.LinAlgError: # encountered non-PD Q_uu, increase mu, start backward pass again
                logger.log('Q_uu min eigen value', np.min(np.linalg.eigvals(Q_uu)))
                self._increase_mu()
                backward_pass_counter += 1

        if not backward_accept:
            logger.log(f'backward not accepted')
            return backward_accept, forward_accept

        # self._decrease_mu()
        logger.log(f'backward accepted after {backward_pass_counter} failed iterations, mu = {self.mu}')

        """
        Forward Pass
        """
        alpha = 1.0
        forward_pass_counter = 0
        while not forward_accept and forward_pass_counter < self.max_forward_iters:
            # reset
            x_prime = x_array[0]
            opt_x_array, opt_u_array = [], []
            opt_J_val = 0

            # forward pass
            for i in range(self.horizon-1):
                x = x_prime
                u = u_array[i] + alpha * open_loop_array[i] + feedback_gain_array[i] @ (x - x_array[i])
                u = np.clip(u, self.act_low, self.act_high)

                # store updated state/action
                opt_x_array.append(x)
                opt_u_array.append(u)
                time.sleep(0.004)
                x_prime, reward = self._f(x, u)
                opt_J_val += -reward

            opt_x_array.append(x_prime)
            opt_u_array.append(np.zeros((self.act_dim,)))  # last action is arbitrary

            # check convergence
            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            if J_val > opt_J_val and J_val - opt_J_val > self.c_1 * (- delta_J_alpha):
                # store updated x, u array (CLIPPED), J_val
                opt_x_array, opt_u_array = np.stack(opt_x_array, axis=0), np.stack(opt_u_array, axis=0)
                # opt_u_array = np.clip(opt_u_array, self.act_low, self.act_high)
                self.x_array, self.u_array = opt_x_array, opt_u_array
                self.J_val = opt_J_val
                forward_accept = True
            else:
                # continue line search
                alpha /= self.alpha_decay_factor
                forward_pass_counter += 1

            # print(J_val, opt_J_val, delta_J_alpha)

        # # FIXME: adapt mu here?
        if not forward_accept:
            logger.log(f'foward pass not accepted')
            self._increase_mu()
        else:
            logger.log(f'forward pass accepted with alpha = {alpha} after {forward_pass_counter} failed iterations')
            self._decrease_mu()

        return backward_accept, forward_accept

    def _decrease_mu(self):
        self.delta = min(1, self.delta) / self.delta_0
        self.mu *= self.delta
        if self.mu < self.mu_min:
            self.mu = 0.0

    def _increase_mu(self):
        # adapt delta, mu
        self.delta = max(1.0, self.delta) * self.delta_0
        self.mu = max(self.mu_min, self.mu * self.delta)

    def _reset_mu(self):
        self.mu = self.mu_init
        self.delta = self.delta_0

    def shift_x_u_by_one(self, u_new):
        # u_new must be clipped before passed in
        if u_new is None:
            u_new = np.mean(self.u_array, axis=0) + np.random.normal(loc=0, scale=0.05, size=(self.act_dim,))
        x_new, reward = self._f(x=self.x_array[-1, :], u=self.u_array[-1, :])
        self.x_array = np.concatenate([self.x_array[1:, :], x_new[None]])
        self.u_array = np.concatenate([self.u_array[1:, :], u_new[None]])
        self.J_val = -np.sum(self._env.reward(obs=self.x_array[:-1], acts=self.u_array[:-1], next_obs=None))

    def reset_x_u(self, init_u_array):
        self._reset_mu()
        if init_u_array is None:
            init_u_array = self.u_array + np.random.normal(loc=0, scale=0.1, size=self.u_array.shape)
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        self.u_array = init_u_array
        self.x_array, self.J_val = self._run_open_loop(init_u_array)
        logger.log(f'reset_x_u with J_val = {self.J_val}')

    def _run_open_loop(self, u_array):
        x_array, sum_rewards = [self._env.reset()], 0
        for i in range(self.horizon-1):
            x, reward, _, _ = self._env.step(u_array[i])
            x_array.append(x)
            sum_rewards += reward
        x_array = np.stack(x_array, axis=0)
        return x_array, -sum_rewards

    def compute_traj_returns(self, discount=1):
        # rewards = self._env.reward(obs=self.x_array, acts=self.u_array, next_obs=None)
        # return sum([discount**t * reward for t, reward in enumerate(rewards)])
        return -self.J_val


def DDP_worker(remote, parent_remote, env_pickle, eps,
                       horizon, obs_dim, act_dim,
                       seed, verbose):

    # batch_size means the num_rollouts in the original env executors, and it means number of experts
    # when the dynamics model is ground truth

    print('collocation state worker starts...')

    parent_remote.close()

    env = pickle.loads(env_pickle)
    np.random.seed(seed)

    def f(x, u):
        _ = env.reset_from_obs(x)
        x_prime, _, _, _ = env.step(u)
        return x_prime

    def jac_f_x(x, u, centered=True):
        """
        :param x: (act_dim.)
        :param u: (obs_dim,)
        :return: (obs_dim, obs_dim)
        """
        jac = np.zeros((obs_dim, obs_dim))
        e_i = np.zeros((obs_dim,))
        if centered:
            for i in range(obs_dim):
                e_i[i] = eps
                jac[:, i] = (f(x+e_i, u) - f(x-e_i, u)) / (2*eps)
                e_i[i] = 0
        else:
            f_val = f(x, u)
            for i in range(obs_dim):
                e_i[i] = eps
                jac[:, i] = (f(x+e_i, u) - f_val) / eps
                e_i[i] = 0
        return jac

    def jac_f_u(x, u, centered=True):
        """
        :param x: (act_dim.)
        :param u: (obs_dim,)
        :return: (obs_dim, act_dim)
        """
        jac = np.zeros((obs_dim, act_dim))
        e_i = np.zeros((act_dim,))
        if centered:
            for i in range(act_dim):
                e_i[i] = eps
                jac[:, i] = (f(x, u+e_i) - f(x, u-e_i)) / (2*eps)
                e_i[i] = 0
        else:
            f_val = f(x, u)
            for i in range(act_dim):
                e_i[i] = eps
                jac[:, i] = (f(x, u+e_i) - f_val) / eps
                e_i[i] = 0
        return jac

    def hessian_f_xx(x, u):
        hess = np.zeros((obs_dim, obs_dim, obs_dim))
        f_val = f(x, u)
        e_i, e_j = np.zeros((obs_dim,)), np.zeros((obs_dim,))
        for i in range(obs_dim):
            e_i[i] = eps
            f_val_fix_i = f(x+e_i, u) - f_val
            for j in range(obs_dim):
                e_j[j] = eps
                hess[:, i, j] = (f(x+e_i+e_j, u) - f(x+e_j, u) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        return (hess + np.transpose(hess, axes=[0, 2, 1])) * 0.5

    def hessian_f_uu(x, u):
        hess = np.zeros((obs_dim, act_dim, act_dim))
        f_val = f(x, u)
        e_i, e_j = np.zeros((act_dim,)), np.zeros((act_dim,))
        for i in range(act_dim):
            e_i[i] = eps
            f_val_fix_i = f(x, u+e_i) - f_val
            for j in range(act_dim):
                e_j[j] = eps
                hess[:, i, j] = (f(x, u+e_i+e_j) - f(x, u+e_j) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        return (hess + np.transpose(hess, axes=[0, 2, 1])) * 0.5

    def hessian_f_ux(x, u):
        hess = np.zeros((obs_dim, act_dim, obs_dim))
        f_val = f(x, u)
        e_i, e_j = np.zeros((act_dim,)), np.zeros((obs_dim,))
        for i in range(act_dim):
            e_i[i] = eps
            f_val_fix_i = f(x, u+e_i) - f_val
            for j in range(obs_dim):
                e_j[j] = eps
                hess[:, i, j] = (f(x+e_j, u+e_i) - f(x, u+e_i) - f_val_fix_i) / eps**2
                e_j[j] = 0
            e_i[i] = 0
        # return (hess + np.transpose(hessian_f_xu(x, u), axes=[0, 2, 1])) * 0.5
        return hess

    str_to_fn = dict(jac_f_x=jac_f_x, jac_f_u=jac_f_u, hessian_f_xx=hessian_f_xx, hessian_f_uu=hessian_f_uu, hessian_f_ux=hessian_f_ux)

    while True:
        # receive command and data from the remote
        cmd, *data = remote.recv()
        # do a step in each of the environment of the worker
        if cmd == 'close':
            remote.close()
            break
        else:
            inputs_dict, = data
            x_array, u_array = inputs_dict['obs'], inputs_dict['act']

            try:
                fn = str_to_fn[cmd]

            except KeyError:
                print(f'receiving command {cmd}')
                raise NotImplementedError

            result = [fn(x_array[i], u_array[i]) for i in range(horizon-2, -1, -1)]
            remote.send(result)
