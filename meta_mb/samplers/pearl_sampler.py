from meta_mb.utils import utils
from meta_mb.logger import logger
from meta_mb.replay_buffers.simple_replay_buffer import SimpleReplayBuffer
from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.base import BaseSampler
import time
import numpy as np
import copy
from pyprind import ProgBar


class PEARLSampler(BaseSampler):
    """
    Sampler interface

    Args:
        env (gym.Env) : environment object
        policy (meta_mb.policies.policy) : policy object
        batch_size (int) : number of trajectories per task
        max_path_length (int) : max number of steps per trajectory
    """

    def __init__(self,
                 env,
                 policy,
                 num_rollouts,
                 max_path_length,
                 latent_dim,
                 context_encoder
    ):
        assert hasattr(env, 'reset') and hasattr(env, 'step')

        super().__init__()

        self.env = env
        self.policy = policy
        self.max_path_length = max_path_length
        self.context_encoder = context_encoder
        self.latent_dim = latent_dim

        self.total_samples = num_rollouts * max_path_length
        self.total_timesteps_sampled = 0

        self.z_buf = SimpleReplayBuffer(self.env, self.total_samples)
        self.z_means_buf = SimpleReplayBuffer(self.env, self.total_samples)
        self.z_vars_buf = SimpleReplayBuffer(self.env, self.total_samples)

        self.clear_z()

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = tf.math.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = tf.math.mean(params, dim=1)
        self.sample_z()

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = tf.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = tf.ones(num_tasks, self.latent_dim)
        else:
            var = tf.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def sample_z(self):
        if self.use_ib:
            posteriors = [tf.distributions.Normal(m, tf.math.sqrt(s)) for m, s in zip(tf.unstack(self.z_means), tf.unstack(self.z_vars))]
            z = [d.sample() for d in posteriors]
            self.z = np.stack(z)
        else:
            self.z = self.z_means

    def obtain_samples(self, context, log=False, log_prefix='', random=False, deterministic=False):
        """
        Collect batch_size trajectories from each task

        Args:
            context (state [obs, act, rewards, terms, next_obs])
            log (boolean): whether to log sampling times
            log_prefix (str) : prefix for logger
            random (boolean): whether the actions are random

        Returns:
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """

        # initial setup / preparation
        paths = []

        n_samples = 0
        running_paths = _get_empty_running_paths_dict()

        if log: pbar = ProgBar(self.total_samples)
        policy_time, env_time = 0, 0

        policy = self.policy
        policy.reset(dones=[True])

        # initial reset of meta_envs
        obs = np.asarray(self.env.reset())
        z = self.z
        in_ = np.cat([obs, z), dim=1)

        ts = 0

        while n_samples < self.total_samples:
            self.infer_posterior(context)
            self.sample_z()

            # execute policy
            t = time.time()
            if random:
                action = self.env.action_space.sample()
                agent_info = {}
            elif deterministic:
                action, agent_info = policy.get_action(in_)
                action = agent_info['mean']
                if self.policy.squashed:
                    action = np.tanh(action)
            else:
                action, agent_info = policy.get_action(in_)
                if action.ndim == 2:
                    action = action[0]
            policy_time += time.time() - t

            # step environments
            t = time.time()
            next_obs, reward, done, env_info = self.env.step(action)

            ts += 1

            env_time += time.time() - t

            new_samples = 0

            # append new samples to running paths
            if isinstance(reward, np.ndarray):
                reward = reward[0]
            running_paths["observations"].append(obs)
            running_paths["context"].append(z)
            running_paths["actions"].append(action)
            running_paths["rewards"].append(reward)
            running_paths["dones"].append(done)
            running_paths["env_infos"].append(env_info)
            running_paths["agent_infos"].append(agent_info)

            # if running path is done, add it to paths and empty the running path
            if done or ts >= self.max_path_length:
                paths.append(dict(
                    observations=np.asarray(running_paths["observations"]),
                    actions=np.asarray(running_paths["actions"]),
                    rewards=np.asarray(running_paths["rewards"]),
                    dones=np.asarray(running_paths["dones"]),
                    env_infos=utils.stack_tensor_dict_list(running_paths["env_infos"]),
                    agent_infos=utils.stack_tensor_dict_list(running_paths["agent_infos"]),
                ))
                new_samples += len(running_paths["rewards"])
                running_paths = _get_empty_running_paths_dict()

            if done or ts >= self.max_path_length:
                next_obs = self.env.reset()
                self.infer_posterior(context)
                next_z = self.sample_z()
                ts = 0

            if log: pbar.update(new_samples)
            n_samples += new_samples
            obs = next_obs
            z = self.z
            in_ = np.cat([obs, z), dim=1)

        if log: pbar.stop()

        self.total_timesteps_sampled += self.total_samples
        if log:
            logger.logkv(log_prefix + "PolicyExecTime", policy_time)
            logger.logkv(log_prefix + "EnvExecTime", env_time)

        return paths



