from meta_mb.utils.serializable import Serializable
from meta_mb.envs.robotics.vectorized_env_executor import IterativeEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils
# from pyprind import ProgBar
import numpy as np
import time
import pickle


class GCSampler(Serializable):
    """
    Sampler for Meta-RL

    Args:
        env :
    """

    def __init__(
            self,
            env_pickled,
            policy,
            goal_buffer,
            num_rollouts,
            max_path_length,
            n_parallel=1,
            vae=None,
    ):
        Serializable.quick_init(self, locals())

        self.env = pickle.loads(env_pickled)
        self.policy = policy
        self.goal_buffer = goal_buffer
        self.vae = vae

        self.num_rollouts = num_rollouts
        self.max_path_length = max_path_length

        self._timesteps_sampled_per_itr = num_rollouts * max_path_length
        self._total_timesteps_sampled = 0

        # setup vectorized environment

        if n_parallel > 1:
            # self.vec_env = ParallelEnvExecutor(env_pickled, n_parallel, num_rollouts, self.max_path_length)
            raise NotImplementedError
        else:
            self.vec_env = IterativeEnvExecutor(env_pickled, num_rollouts, max_path_length)

    def collect_rollouts(self, eval=False, *args, **kwargs):
        paths = []
        for batch in self.goal_buffer.get_batches(eval=eval, batch_size=self.num_rollouts):
            _paths = self._collect_rollouts(batch, *args, **kwargs)
            paths.extend(_paths)

        return paths

    def _collect_rollouts(self, goals, random=False, log=False, log_prefix=''):
        assert goals.ndim == 2 and goals.shape[0] == self.num_rollouts, goals.shape
        policy = self.policy
        paths = []
        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.num_rollouts)]
        policy.reset(dones=[True] * self.num_rollouts)

        # if verbose: pbar = ProgBar(self.total_samples)
        policy_time, env_time, store_time = 0, 0, 0

        # sample goals
        goal_ng = goals
        init_obs_no = self.vec_env.reset(goal_ng)
        obs_no = init_obs_no

        while n_samples < self._timesteps_sampled_per_itr:
            # execute policy
            t = time.time()
            if self.vae is not None:
                obs_no = np.array(obs_no)
                obs_no = self.vae.encode(obs_no)
            if random:
                act_na = np.stack([self.env.action_space.sample() for _ in range(self.num_rollouts)], axis=0)
                agent_info_n = []
            else:
                obs_no = np.array(obs_no)
                act_na, agent_info_n = policy.get_actions(obs_no, goal_ng)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obs_no, reward_n, done_n, env_info_n = self.vec_env.step(act_na)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            t = time.time()
            agent_info_n, env_info_n = self._handle_info_dicts(agent_info_n, env_info_n)

            for idx, goal, observation, action, reward, env_info, agent_info, done in zip(
                    np.arange(self.num_rollouts), goal_ng, obs_no, act_na,
                    reward_n, env_info_n, agent_info_n, done_n,
            ):
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["goals"].append(goal)
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths.append(dict(
                        goals=np.asarray(running_paths[idx]["goals"]),
                        observations=np.asarray(running_paths[idx]["observations"]),
                        actions=np.asarray(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        dones=np.asarray(running_paths[idx]["dones"]),
                        env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                        agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()
            store_time = time.time() - t

            # if verbose: pbar.update(self.vec_env.num_envs)
            obs_no = next_obs_no

        # if verbose: pbar.stop()

        self._total_timesteps_sampled += n_samples

        if log:
            logger.logkv(log_prefix + "TimeStepsCtr", self._total_timesteps_sampled)
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
            logger.logkv(log_prefix + "StoreTime", store_time)

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
    return dict(goals=[], observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
