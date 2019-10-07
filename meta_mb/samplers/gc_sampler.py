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
            goal_sampler,
            env,
            policy,
            num_rollouts,
            max_path_length,
            n_parallel=1,
            vae=None,
    ):
        Serializable.quick_init(self, locals())
        super(Sampler, self).__init__(env, policy, num_rollouts, max_path_length)  # changed from n_parallel to num_rollouts

        self.goal_sampler = goal_sampler
        self.n_parallel = n_parallel
        self.vae = vae

        # setup vectorized environment

        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, self.max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, num_rollouts, self.max_path_length)

    def update_tasks(self):
        pass

    def collect_init_obs(self):
        self.policy.reset(dones=[True] * self.num_envs)
        init_obs_no = self.vec_env.reset()
        return init_obs_no

    def collect_rollouts(
            self, init_obs_no, agent_log_q, max_log_q, target_goals,
            random=False, verbose=False, log=False, log_prefix='',
    ):
        policy = self.policy
        paths = []
        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.num_envs)]

        if verbose: pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        # sample goals
        obs_no = init_obs_no
        goal_ng = np.random.choice(len(target_goals), size=self.num_envs, replace=True, p=max_log_q-agent_log_q)

        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            if self.vae is not None:
                obs_no = np.array(obs_no)
                obs_no = self.vae.encode(obs_no)
            if random:
                act_na = np.stack([self.env.action_space.sample() for _ in range(self.num_envs)], axis=0)
                agent_info_n = []
            else:
                obs_no = np.array(obs_no)
                act_na, agent_info_n = policy.get_actions(obs_no, goal_ng)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obs_no, reward_n, done_n, env_info_n = self.vec_env.step(act_na, goal_ng)
            env_time += time.time() - t

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_info_n, env_info_n = self._handle_info_dicts(agent_info_n, env_info_n)

            new_samples = 0
            for idx, goal, observation, action, reward, env_info, agent_info, done in zip(
                    np.arange(self.num_envs), goal_ng, obs_no, act_na,
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
                    new_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = _get_empty_running_paths_dict()

            if verbose: pbar.update(self.vec_env.num_envs)
            n_samples += new_samples
            obs_no = next_obs_no

        if verbose: pbar.stop()

        self.total_timesteps_sampled += self.total_samples

        if log:
            logger.logkv(log_prefix + "TimeStepsCtr", self.total_timesteps_sampled)
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

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
    return dict(observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
