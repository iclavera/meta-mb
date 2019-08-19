from meta_mb.samplers.base import BaseSampler
from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils
from pyprind import ProgBar
import numpy as np
import time
import itertools


class Sampler(BaseSampler):
    """
    Sampler for Meta-RL

    Args:
        env (meta_mb.meta_envs.base.MetaEnv) : environment object
        policy (meta_mb.policies.base.Policy) : policy object
        batch_size (int) : number of trajectories per task
        meta_batch_size (int) : number of meta tasks
        max_path_length (int) : max number of steps per trajectory
        envs_per_task (int) : number of meta_envs to run vectorized for each task (influences the memory usage)
    """

    def __init__(
            self,
            env,
            policy,
            num_rollouts,
            max_path_length,
            max_path_length_eval,
            n_parallel=1,
            vae=None,
    ):
        Serializable.quick_init(self, locals())
        super(Sampler, self).__init__(env, policy, num_rollouts, max_path_length)  # changed from n_parallel to num_rollouts

        self.total_samples = num_rollouts * max_path_length
        self.max_path_length_eval = max_path_length_eval
        self.n_parallel = n_parallel
        self.total_timesteps_sampled = 0
        self.vae = vae

        # setup vectorized environment

        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, self.max_path_length)
            self.vec_env_eval = ParallelEnvExecutor(env, n_parallel, num_rollouts, self.max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, num_rollouts, self.max_path_length)
            self.vec_env_eval = IterativeEnvExecutor(env, num_rollouts, self.max_path_length)

    def update_tasks(self):
        pass

    def _closed_loop_performance(self, verbose):
        policy = self.policy
        obses = self.vec_env.reset()
        returns = np.zeros((self.num_envs,))
        if verbose: pbar = ProgBar(self.max_path_length_eval)
        for path_length in range(self.max_path_length_eval):
            actions_array, _, _ = policy.get_actions(np.asarray(obses), verbose=False)
            if path_length + len(actions_array) < self.max_path_length_eval:
                actions_array = actions_array[0:1, :, :]

            for actions in actions_array:
                obses, rewards, _, _ = self.vec_env_eval.step(actions)  # FIXME: use self.vec_env is fine?
                returns += rewards
                if verbose: pbar.update(1)
        if verbose: pbar.stop()

        return returns

    def obtain_samples(self, log_open_loop_performance=False, log_closed_loop_performance=False, log=False, log_prefix='', random=False, sinusoid=False, verbose=True):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        if verbose: pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        if not hasattr(policy, 'warm_reset'):
            policy.reset(dones=[True] * self.vec_env.num_envs)

        # initial reset of meta_envs
        obses_fake = obses = np.asarray(self.vec_env.reset())

        # utils for ilqr method
        u_array = []
        open_loop_returns = np.zeros((self.num_envs,))

        path_length = 0
        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            if self.vae is not None:
                obses_fake = np.array(obses_fake)
                obses_fake = self.vae.encode(obses_fake)
                obses = np.array(obses)
                obses = self.vae.encode(obses)

            if random:
                actions = np.stack([self.env.action_space.sample() for _ in range(self.vec_env.num_envs)], axis=0)
                actions_array = actions[None]
                agent_infos = []
            elif sinusoid:
                action_space = self.env.action_space.shape[0]
                num_envs = self.vec_env.num_envs
                actions = np.stack([policy.get_sinusoid_actions(action_space, t/policy.horizon * 2 * np.pi) for _ in range(num_envs)], axis=0)
                actions_array = actions[None]
                agent_infos = []
            else:
                obses_fake = np.array(obses_fake)
                actions_array, agent_infos, next_obses_fake = policy.get_actions(obses_fake)
                # assert len(actions) == len(obses_fake)  # (num_rollouts, space_dims)
                if path_length + len(actions_array) < self.max_path_length:
                    actions_array = actions_array[0:1, :, :]
            policy_time += time.time() - t

            for actions in actions_array:
                # step environments
                t = time.time()
                next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
                env_time += time.time() - t

                #  stack agent_infos and if no infos were provided (--> None) create empty dicts
                agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)
                open_loop_returns += rewards
                u_array.append(actions)

                new_samples = 0
                for idx, observation, next_observation, action, reward, env_info, agent_info, done in zip(
                        itertools.count(), obses, next_obses, actions, rewards, env_infos, agent_infos, dones,
                ):
                    # append new samples to running paths
                    if isinstance(reward, np.ndarray):
                        reward = reward[0]
                    running_paths[idx]["observations"].append(observation)
                    running_paths[idx]["next_observations"].append(next_observation)
                    running_paths[idx]["actions"].append(action)
                    running_paths[idx]["rewards"].append(reward)
                    running_paths[idx]["dones"].append(done)
                    running_paths[idx]["env_infos"].append(env_info)
                    running_paths[idx]["agent_infos"].append(agent_info)

                    # if running path is done, add it to paths and empty the running path
                    if done:
                        paths.append(dict(
                            observations=np.asarray(running_paths[idx]["observations"]),
                            next_observations=np.asarray(running_paths[idx]["next_observations"]),
                            actions=np.asarray(running_paths[idx]["actions"]),
                            rewards=np.asarray(running_paths[idx]["rewards"]),
                            dones=np.asarray(running_paths[idx]["dones"]),
                            env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                            agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                        ))
                        new_samples += len(running_paths[idx]["rewards"])
                        running_paths[idx] = _get_empty_running_paths_dict()

                if verbose: pbar.update(self.vec_env.num_envs)
                n_samples += new_samples
                path_length += 1

            # step using model
            if not random and not sinusoid:
                logger.log('PathLength', path_length)
                obses_fake = next_obses_fake  # TODO: alternatively, obses_fake = next_obses
            obses = next_obses

        if verbose: pbar.stop()
        self.total_timesteps_sampled += self.total_samples
        u_array = np.stack(u_array, axis=0)  # (max_path_length, num_envs, act_dim)

        if log:
            logger.logkv(log_prefix + "TimeStepsCtr", self.total_timesteps_sampled)
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        if log_open_loop_performance:  # path_length = max_path_length
            # vec_env is reset within run_open_loop
            for idx, ret_env in enumerate(open_loop_returns):
                logger.logkv(log_prefix + f"RetEnvOpen-{idx}", ret_env)

        if log_closed_loop_performance:  # evaluation for dynamics model, path_length = max_path_length_eval
            closed_loop_returns = self._closed_loop_performance(verbose=verbose)

            for idx, ret_env in enumerate(closed_loop_returns):
                logger.logkv(log_prefix + f"RetEnvClosed-{idx}", ret_env)
        # max_path_length_eval > max_path_length for vec_env causes early termination (dones = True at 30)

        policy.warm_reset(u_array)  # TODO: warm_reset(None)?? Shuffle?? CEM?

        return paths

    def _handle_info_dicts(self, agent_infos, env_infos):
        if not env_infos:
            env_infos = [dict() for _ in range(self.vec_env.num_envs)]
        if not agent_infos:
            agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
        return agent_infos, env_infos

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        # dumps policy
        state['policy'] = self.policy.__getstate__()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.policy = state['policy']


def _get_empty_running_paths_dict():
    return dict(observations=[], next_observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
