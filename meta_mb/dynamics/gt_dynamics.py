import numpy as np
from meta_mb.logger import logger
import os
from itertools import accumulate
from meta_mb.samplers.vectorized_env_executor import IterativeEnvExecutor, ParallelActionDerivativeExecutor
import copy
from math import ceil
import matplotlib.pyplot as plt


class GTDynamics():
    """
    Ground truth dynamics model which is STATELESS compared to simulator environments.
    """

    def __init__(self,
                 name,
                 env,
                 num_rollouts,
                 horizon,
                 max_path_length,
                 discount=1,
                 n_parallel=1,
                 verbose=False
                 ):
        self.name = name
        self.env = env
        self.num_envs = num_rollouts
        self.max_path_length = max_path_length
        self.horizon = horizon
        self.discount = discount
        self.n_parallel = n_parallel
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # self.vec_env = IterativeEnvExecutor(env, num_rollouts, max_path_length)
        # self.single_env = IterativeEnvExecutor(env, 1, max_path_length)
        self._env = copy.deepcopy(env)

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, prefix=''):
        pass

    # def predict(self, obs, act):
    #     """
    #
    #     :param obs: (batch_size, obs_space_dims)
    #     :param act: (batch_size, obs_space_dims)
    #     :return:
    #     """
    #     if obs is None:
    #         next_obs, rewards, dones, env_infos = self.vec_env.step(act)
    #     else:
    #         assert obs.shape[0] == act.shape[0]
    #         assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
    #         assert act.ndim == 2 and act.shape[1] == self.action_space_dims
    #
    #         obs_reset = self.vec_env.reset_hard(buffer=dict(observations=obs))
    #         print(obs_reset)
    #         print(obs)
    #         next_obs, rewards, dones, env_infos = self.vec_env.step(act)
    #
    #     return next_obs, rewards, dones, env_infos

    # def predict_open_loop(self, init_obs, tau):
    #     self._env.reset_from_obs_hard(init_obs)
    #     obs_hall, obs_hall_mean, obs_hall_std, reward_hall = [], None, None, []
    #     for act in tau:
    #         next_obs, reward, _, _= self._env.step(act)
    #         obs_hall.append(next_obs)
    #         reward_hall.append(reward)
    #
    #     return obs_hall, obs_hall_mean, obs_hall_std, reward_hall
    #
    # def predict_open_loop(self, init_obs, tau):
    #     assert init_obs.shape == (self.obs_space_dims,)
    #     assert len(tau) == self.max_path_length
    #
    #     _ = self.single_env.reset_from_obs_hard(init_obs[None])
    #     obs_hall, obs_hall_mean, obs_hall_std, reward_hall = [], None, None, []
    #     for act in tau:
    #         next_obs, reward, _, _= self.single_env.step(act[None])
    #         next_obs, reward = next_obs[0], reward[0]
    #         obs_hall.append(next_obs)
    #         # obs_hall_mean.append(next_obs)
    #         # obs_hall_std.append(np.zeros_like(next_obs))
    #         reward_hall.append(reward)
    #
    #     return obs_hall, obs_hall_mean, obs_hall_std, reward_hall

    # def get_derivative(self, tau, init_obs=None):
    #     """
    #     Assume s_0 is the reset state.
    #     :param tau: (horizon, batch_size, action_space_dims)
    #     :tf_loss: scalar Tensor R
    #     :return: dR/da_i for i in range(action_space_dims)
    #     """
    #     assert tau.shape == (self.horizon, self.num_envs, self.action_space_dims)
    #     # compute R
    #     # returns = self._compute_returns(tau, init_obs)
    #     # derivative = np.zeros_like(tau)
    #     #
    #     # # perturb tau
    #     # for i in range(self.horizon):
    #     #     for j in range(self.action_space_dims):
    #     #         delta = np.zeros_like(tau)
    #     #         delta[i, :, j] = eps
    #     #         new_returns = self._compute_returns(tau + delta, init_obs=init_obs)
    #     #         derivative[i, :, j] = (new_returns - returns)/eps
    #     #
    #     # return derivative, returns
    #     return self.deriv_env.get_derivative(tau, init_obs)

    def plot_rollout(self, tau, init_obs, global_step):
        if init_obs is None:
            self._env.reset_hard()
        else:
            self._env.reset_hard_from_obs(init_obs)

        obs_array, reward_array, act_norm_array = [], [], []
        for act in tau:
            next_obs, reward, _, _ = self._env.step(act)
            obs_array.append(next_obs)
            reward_array.append(reward)
            act_norm_array.append(np.linalg.norm(act))

        x = np.arange(self.horizon)
        obs_array = np.transpose(np.asarray(obs_array))  # (obs_dims, horizon)
        act_array = np.transpose(np.asarray(tau))  # (act_dims, horizon)

        n_subplots = self.obs_space_dims + self.action_space_dims + 2
        nrows = ceil(np.sqrt(n_subplots))
        ncols = ceil(n_subplots/nrows)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(70, 30))
        axes = axes.flatten()

        for i in range(self.obs_space_dims):  # split by observation space dimension
            ax = axes[i]
            ax.plot(x, obs_array[i], label=f'obs_gt')

        for i in range(self.action_space_dims):
            ax = axes[i+self.obs_space_dims]
            ax.plot(x, act_array[i], label=f'act_{i}', color='r')

        ax = axes[self.obs_space_dims+self.action_space_dims]
        # ax.plot(x, reward_array, label='reward_gt')
        ax.plot(x, act_norm_array, label='act_norm')
        ax.legend()

        ax = axes[self.obs_space_dims+self.action_space_dims+1]
        ax.plot(x, list(accumulate(reward_array)), label='reward_gt')
        # ax.plot(x, list(accumulate(loss_reward)), label='reward_planning')
        ax.legend()

        fig.suptitle(f'{global_step}')

        # plt.show()
        if not hasattr(self, 'save_dir'):
            self.save_dir = os.path.join(logger.get_dir(), 'dyn_vs_env')
            os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, f'{global_step}.png'))
        logger.log('plt saved to', os.path.join(self.save_dir, f'{global_step}.png'))

def _compute_returns(vec_env, tau, init_obs=None):
    """
    Assume s_0 is the reset state.
    :param tau: (horizon, batch_size, action_space_dims)
    :return: total rewards of shape (batch_size,)
    """
    returns = np.zeros(shape=(tau.shape[1],))
    if init_obs is None:
        obs = vec_env.reset_hard()
    else:
        obs = vec_env.reset_from_obs_hard(init_obs)

    # print(f'starting at {obs}')

    for act in tau:
        _, rewards, _, _ = vec_env.step(act)
        returns += rewards
    return returns

if __name__ == "__main__":
    from meta_mb.envs.mb_envs import InvertedPendulumEnv, HalfCheetahEnv
    env = InvertedPendulumEnv()
    # env = HalfCheetahEnv()
    num_rollouts = 3
    horizon = 4
    max_path_length = 20
    eps = 1e-5
    vec_env = IterativeEnvExecutor(env, num_rollouts, max_path_length)
    gt = GTDynamics('gt', env, num_rollouts, horizon, max_path_length, n_parallel=4, verbose=True)

    tau_shape = (horizon, num_rollouts, env.action_space.shape[0])

    tau = np.random.uniform(env.action_space.low, env.action_space.high, size=tau_shape)
    da, returns = gt.get_derivative(tau)
    # print(da)

    # numeric test
    for test_eps in [1e-7, 1e-4, 1e-2, 1e-1]:
        delta = np.random.randint(low=0, high=2, size=tau_shape) * test_eps

        old_returns = _compute_returns(vec_env, tau)
        new_returns = _compute_returns(vec_env, tau + delta)
        delta_returns = new_returns - old_returns

        # approx_delta_returns = [np.inner(delta[i, j, :], tau[i, j, :]) for i in range(num_rollouts)]
        approx_delta_returns = [np.sum(delta[:, i, :] * da[:, i, :]) for i in range(num_rollouts)]

        error = np.abs(np.asarray(delta_returns) - np.asarray(approx_delta_returns))
        print(f'test_eps, delta_norm = {test_eps}, {np.linalg.norm(delta)}')
        # print(f'actual, approx delta returns = {delta_returns}, {approx_delta_returns}')
        print(f'error stats: avg, max = {np.mean(error), np.max(error)}')
        print(f'error pct = {np.mean(error/old_returns)}')
        print()



