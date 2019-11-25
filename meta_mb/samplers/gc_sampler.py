from meta_mb.utils.serializable import Serializable
from meta_mb.envs.robotics.vectorized_env_executor import IterativeEnvExecutor, ParallelEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils
from meta_mb.samplers.noise import VecOrnsteinUhlenbeckActionNoise, VecNormalActionNoise, VecActionNoise
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
            action_noise_str,
            n_parallel=1,
    ):
        Serializable.quick_init(self, locals())

        self.env = pickle.loads(env_pickled)
        self.policy = policy
        self.goal_buffer = goal_buffer

        self.num_rollouts = num_rollouts
        self.max_path_length = max_path_length

        self._timesteps_sampled_per_itr = num_rollouts * max_path_length
        self._total_timesteps_sampled = 0

        # setup vectorized environment

        if n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env_pickled, n_parallel, num_rollouts, max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env_pickled, num_rollouts, max_path_length)

        # set up action_noise instance
        if action_noise_str is None:
            self.vec_action_noise = VecActionNoise(self.num_rollouts)
        elif 'ou' in action_noise_str:
            _, stddev = action_noise_str.split('_')
            self.vec_action_noise = VecOrnsteinUhlenbeckActionNoise(
                num_envs=self.num_rollouts,
                mu=np.zeros(self.env.act_dim),
                sigma=float(stddev) * np.ones(self.env.act_dim),
            )
        elif 'normal' in action_noise_str:
            _, stddev = action_noise_str.split('_')
            self.vec_action_noise = VecNormalActionNoise(
                num_envs=self.num_rollouts,
                mu=np.zeros(self.env.act_dim),
                sigma=float(stddev) * np.ones(self.env.act_dim),
            )
        else:
            raise NotImplementedError

    def collect_rollouts(self, goals, greedy_eps, apply_action_noise=True, log=False, log_prefix=''):
        """

        :param goals: (np.array) goals of shape (num_rollouts, goal_dim)
        :param apply_noise: whether to apply self.action_noise
        :param greedy_eps: epsilon greedy policy
        :param log:
        :param log_prefix:
        :return: (list) a list of paths
        """
        assert goals.ndim == 2 and goals.shape[0] == self.num_rollouts, goals.shape
        self.policy.reset(dones=[True] * self.num_rollouts)
        self.vec_action_noise.reset()

        policy = self.policy
        paths = []
        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.num_rollouts)]
        policy_time, env_time, store_time = 0, 0, 0

        # sample goals
        goal_ng = goals
        init_obs_no = self.vec_env.reset(goal_ng)
        obs_no = init_obs_no

        while n_samples < self._timesteps_sampled_per_itr:
            # execute policy
            t = time.time()
            if greedy_eps == 1:
                act_na = np.stack([self.env.action_space.sample() for _ in range(self.num_rollouts)], axis=0)
                agent_info_n = []
            else:
                obs_no = np.array(obs_no)
                act_na, agent_info_n = policy.get_actions(obs_no, goal_ng)
                # sanity check the magnitude of action statistics
                # act_std = np.exp(agent_info_n[0]['log_std'])  # this number if large, almost one
                # logger.log('act std', act_std)

                if apply_action_noise:
                    act_na += self.vec_action_noise()

                # epsilon greedy policy
                greedy_mask = np.random.binomial(1, greedy_eps, self.num_rollouts).astype(np.bool)
                act_na[greedy_mask] = np.reshape([self.env.action_space.sample() for _ in range(np.sum(greedy_mask))], (-1, self.env.act_dim))

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

            obs_no = next_obs_no

        self._total_timesteps_sampled += n_samples

        if log:
            logger.logkv(log_prefix + "TimeStepsCtr", self._total_timesteps_sampled)
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)
            logger.logkv(log_prefix + "StoreTime", store_time)

        assert len(paths) == self.num_rollouts, (len(paths), self.num_rollouts)

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
        state['policy'] = self.policy.__getstate__()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.policy = state['policy']


def _get_empty_running_paths_dict():
    return dict(goals=[], observations=[], actions=[], rewards=[], dones=[], env_infos=[], agent_infos=[])
