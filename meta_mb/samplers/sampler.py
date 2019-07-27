from meta_mb.samplers.base import BaseSampler
from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor
from meta_mb.logger import logger
from meta_mb.utils import utils

from pyprind import ProgBar
import numpy as np
import time
import os
import itertools
import matplotlib.pyplot as plt
from math import ceil
from itertools import accumulate


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
            n_parallel=1,
            dyn_pred_str=None,
            vae=None,
            ground_truth=False,
    ):
        Serializable.quick_init(self, locals())
        super(Sampler, self).__init__(env, policy, n_parallel, max_path_length)

        self.total_samples = num_rollouts * max_path_length
        self.n_parallel = n_parallel
        self.total_timesteps_sampled = 0
        self.vae = vae
        self.dyn_pred_str = dyn_pred_str
        self.ground_truth = ground_truth

        # setup vectorized environment

        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, self.max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, num_rollouts, self.max_path_length)

        self._global_step = 0

    def update_tasks(self):
        pass

    def obtain_samples_ground_truth(self, log, log_prefix='', deterministic=False, verbose=True, plot_first_rollout=False):
        print(plot_first_rollout)
        plot_first_rollout = True  # FIXME
        policy = self.policy
        policy.reset(dones=[True] * self.vec_env.num_envs)

        rollouts, returns_array, grad_norm_array, avg_rollout_norm = policy.get_rollouts(
            observations=None, deterministic=deterministic, plot_info=True, plot_first_rollout=plot_first_rollout
        )
        logger.log(returns_array)

        logger.logkv(log_prefix + 'AverageReturn', np.mean(returns_array))
        if log:
            logger.logkv(log_prefix + 'StdReturn', np.std(returns_array))
            logger.logkv(log_prefix + 'MaxReturn', np.max(returns_array))
            logger.logkv(log_prefix + 'MinReturn', np.min(returns_array))

    def obtain_samples(self, log=False, log_prefix='', random=False, sinusoid=False, deterministic=False,
                       verbose=True, plot_first_rollout=False):
        """
        Collect batch_size trajectories from each task

        Args:
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        if self.ground_truth:
            return self.obtain_samples_ground_truth(log, log_prefix, deterministic, plot_first_rollout)

        # initial setup / preparation
        self._global_step += 1
        paths = []

        n_samples = 0
        running_paths = [_get_empty_running_paths_dict() for _ in range(self.vec_env.num_envs)]

        if verbose: pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True] * self.vec_env.num_envs)

        # initial reset of meta_envs
        obses = np.asarray(self.vec_env.reset())

        if plot_first_rollout:
            init_obs = obses[0]
            tau, act_norm, obs_real, reward_real, loss_reg = [], [], [], [], []  # final shape: (max_path_length, space.dims)
            if deterministic:
                tau_mean, tau_std = None, None
            else:
                tau_mean, tau_std = [], []

        itr_counter = 0
        while n_samples < self.total_samples:
            # execute policy
            t = time.time()
            if self.vae is not None:
                obses = np.array(obses)
                obses = self.vae.encode(obses)
            if random:
                actions = np.stack([self.env.action_space.sample() for _ in range(self.vec_env.num_envs)], axis=0)
                agent_infos = []
            elif sinusoid:
                action_space = self.env.action_space.shape[0]
                num_envs = self.vec_env.num_envs
                actions = np.stack([policy.get_sinusoid_actions(action_space, t/policy.horizon * 2 * np.pi) for _ in range(num_envs)], axis=0)
                agent_infos = []
            else:
                obses = np.array(obses)
                if plot_first_rollout:
                    actions, agent_infos = policy.get_actions(
                        obses,
                        deterministic=deterministic,
                        return_first_info=True,
                        log_grads_for_plot=(itr_counter < self.max_path_length),
                    )
                else:
                    actions, agent_infos = policy.get_actions(obses)
                assert len(actions) == len(obses)  # (num_rollouts, space_dims)
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            if plot_first_rollout and not random and not sinusoid:
                tau.append(actions[0])  # actions = (num_envs, act_space_dims), actions[0] corresponds to the first env
                act_norm.append(np.linalg.norm(actions[0]))
                obs_real.append(next_obses[0])
                reward_real.append(rewards[0])
                if not deterministic:
                    tau_mean.append(agent_infos[0]['mean'])
                    tau_std.append(agent_infos[0]['std'])
                loss_reg.append(agent_infos[0]['reg'])
                agent_infos = []

            #  stack agent_infos and if no infos were provided (--> None) create empty dicts
            agent_infos, env_infos = self._handle_info_dicts(agent_infos, env_infos)

            new_samples = 0
            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                # append new samples to running paths
                if isinstance(reward, np.ndarray):
                    reward = reward[0]
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["dones"].append(done)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)

                # if running path is done, add it to paths and empty the running path
                if done:
                    paths.append(dict(
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
            obses = next_obses
            itr_counter += 1

        if verbose: pbar.stop()

        self.total_timesteps_sampled += self.total_samples

        if log:
            logger.logkv(log_prefix + "TimeStepsCtr", self.total_timesteps_sampled)
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        if not plot_first_rollout:
            return paths

        # plot the first collected rollout, which has max_path_length
        if not random and not sinusoid:
            # obs_hall, obs_hall_mean, obs_hall_std, reward_hall = [], [], [], []
            # obs = init_obs
            # for action in tau:
            #     next_obs, agent_info = policy.dynamics_model.predict(
            #         obs[None],
            #         action[None],
            #         pred_type=self.dyn_pred_str,
            #         deterministic=False,
            #         return_infos=True,
            #     )
            #     next_obs, agent_info = next_obs[0], agent_info[0]
            #     obs_hall.append(next_obs)
            #     obs_hall_mean.append(agent_info['mean'])
            #     obs_hall_std.append(agent_info['std'])
            #     reward_hall.extend(self.env.reward(obs[None], action[None], next_obs[None]))
            #     obs = next_obs
            obs_hall, obs_hall_mean, obs_hall_std, reward_hall = policy.predict_open_loop(init_obs, tau)

            x = np.arange(self.max_path_length)
            obs_space_dims = self.env.observation_space.shape[0]
            action_space_dims = self.env.action_space.shape[0]
            obs_hall = np.transpose(np.asarray(obs_hall))  # (max_path_length, obs_space_dims) -> (obs_space_dims, max_path_length)
            obs_real = np.transpose(np.asarray(obs_real))
            tau = np.transpose(np.asarray(tau))  # (max_path_length, action_space_dims) -> (action_space_dims, max_path_length)

            n_subplots = obs_space_dims + action_space_dims + 2
            nrows = ceil(np.sqrt(n_subplots))
            ncols = ceil(n_subplots/nrows)
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(70, 30))
            axes = axes.flatten()

            # obs_ymin = np.min([obs_hall - obs_hall_std, obs_real]) + 0.1
            # obs_ymax = np.max([obs_hall + obs_hall_std, obs_real]) - 0.1
            for i in range(obs_space_dims):  # split by observation space dimension
                ax = axes[i]
                ax.plot(x, obs_hall[i], label=f'obs_{i}_dyn')
                ax.plot(x, obs_real[i], label=f'obs_{i}_env')
                if obs_hall_std is not None:
                    obs_hall_mean = np.transpose(np.asarray(obs_hall_mean))
                    obs_hall_std = np.transpose(np.asarray(obs_hall_std))
                    ax.fill_between(x, obs_hall_mean[i] + obs_hall_std[i], obs_hall_mean[i] - obs_hall_std[i], alpha=0.2)
                # ax.set_ylim([obs_ymin, obs_ymax])

            for i in range(action_space_dims):
                ax = axes[i+obs_space_dims]
                ax.plot(x, tau[i], label=f'act_{i}', color='r')
                if tau_std is not None:
                    tau_mean = np.transpose(np.asarray(tau_mean))
                    tau_std = np.transpose(np.asarray(tau_std))
                    ax.fill_between(x, tau_mean[i] + tau_std[i], tau_mean[i] - tau_std[i], color='r', alpha=0.2)
                # ax.set_ylim([self.env.action_space.low[i]-0.1, self.env.action_space.high[i]+0.1])

            ax = axes[obs_space_dims+action_space_dims]
            ax.plot(x, reward_hall, label='reward_dyn')
            ax.plot(x, reward_real, label='reward_env')
            ax.plot(x, act_norm, label='act_norm')
            # ax.plot(x, loss_reward, label='reward_planning')  # FIXME: == reward_env??
            ax.plot(x, loss_reg, label='loss_reg')
            ax.legend()

            ax = axes[obs_space_dims+action_space_dims+1]
            ax.plot(x, list(accumulate(reward_hall)), label='reward_dyn')
            ax.plot(x, list(accumulate(reward_real)), label='reward_env')
            # ax.plot(x, list(accumulate(loss_reward)), label='reward_planning')
            ax.legend()

            fig.suptitle(f'{self._global_step}')

            # plt.show()
            if not hasattr(self, 'save_dir'):
                self.save_dir = os.path.join(logger.get_dir(), 'dyn_vs_env')
                os.makedirs(self.save_dir, exist_ok=True)
            plt.savefig(os.path.join(self.save_dir, f'{self._global_step}.png'))
            logger.log('plt saved to', os.path.join(self.save_dir, f'{self._global_step}.png'))

            policy.plot_grads()

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
