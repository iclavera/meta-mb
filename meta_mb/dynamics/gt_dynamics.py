import numpy as np
from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.vectorized_env_executor import ParallelEnvExecutor, IterativeEnvExecutor


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

        if self.n_parallel > 1:
            self.vec_env = ParallelEnvExecutor(env, n_parallel, num_rollouts, max_path_length)
        else:
            self.vec_env = IterativeEnvExecutor(env, num_rollouts, max_path_length)

        self.single_env = IterativeEnvExecutor(env, 1, max_path_length)

    def fit(self, obs, act, obs_next, epochs=1000, compute_normalization=True,
            valid_split_ratio=None, rolling_average_persitency=None, verbose=False, log_tabular=False, prefix=''):
        pass

    def predict(self, obs, act):
        """

        :param obs: (batch_size, obs_space_dims)
        :param act: (batch_size, obs_space_dims)
        :return:
        """
        if obs is None:
            next_obs, rewards, dones, env_infos = self.vec_env.step(act)
        else:
            assert obs.shape[0] == act.shape[0]
            assert obs.ndim == 2 and obs.shape[1] == self.obs_space_dims
            assert act.ndim == 2 and act.shape[1] == self.action_space_dims

            obs_reset = self.vec_env.reset_hard(buffer=dict(observations=obs))
            print(obs_reset)
            print(obs)
            next_obs, rewards, dones, env_infos = self.vec_env.step(act)

        return next_obs, rewards, dones, env_infos

    def predict_open_loop(self, init_obs, tau):
        assert init_obs.shape == (self.obs_space_dims,)
        assert len(tau) == self.max_path_length

        _ = self.single_env.reset_from_obs_hard(init_obs[None])
        obs_hall, obs_hall_mean, obs_hall_std, reward_hall = [], [], [], []
        for act in tau:
            next_obs, reward, _, _= self.single_env.step(act[None])
            next_obs, reward = next_obs[0], reward[0]
            obs_hall.append(next_obs)
            obs_hall_mean.append(next_obs)
            obs_hall_std.append(np.zeros_like(next_obs))
            reward_hall.append(reward)

        return obs_hall, obs_hall_mean, obs_hall_std, reward_hall

    def get_derivative(self, tau, init_obs=None, eps=1e-6):
        """
        Assume s_0 is the reset state.
        :param tau: (horizon, batch_size, action_space_dims)
        :tf_loss: scalar Tensor R
        :return: dR/da_i for i in range(action_space_dims)
        """
        assert tau.shape == (self.horizon, self.num_envs, self.action_space_dims)
        # compute R
        returns = self._compute_returns(tau, init_obs)
        derivative = np.zeros_like(tau)

        # perturb tau
        for i in range(self.horizon):
            for j in range(self.action_space_dims):
                delta = np.zeros_like(tau)
                delta[i, :, j] = eps
                new_returns = self._compute_returns(tau + delta, init_obs=init_obs)
                derivative[i, :, j] = (new_returns - returns)/eps

        return derivative, returns

    def _compute_returns(self, tau, init_obs=None):
        """
        Assume s_0 is the reset state.
        :param tau: (horizon, batch_size, action_space_dims)
        :return: total rewards of shape (batch_size,)
        """
        returns = np.zeros(shape=(self.num_envs,))
        if init_obs is None:
            # obs = self.vec_env.reset(dict(observations=np.zeros((self.num_envs, self.obs_space_dims))))
            # obs = self.vec_env.reset()
            obs = self.vec_env.reset_hard()
        else:
            obs = self.vec_env.reset_from_obs_hard(init_obs)

        for t in range(self.horizon):
            obs, rewards, dones, env_infos = self.vec_env.step(actions=tau[t])
            returns += self.discount ** t * np.asarray(rewards)

        return returns

if __name__ == "__main__":
    from meta_mb.envs.mb_envs import InvertedPendulumEnv, HalfCheetahEnv
    env = InvertedPendulumEnv()
    # env = HalfCheetahEnv()
    num_rollouts = 20
    horizon = 15
    max_path_length = 200
    eps = 1e-5
    gt = GTDynamics('gt', env, num_rollouts, horizon, max_path_length)

    tau_shape = (horizon, num_rollouts, env.action_space.shape[0])

    tau = np.random.uniform(env.action_space.low, env.action_space.high, size=tau_shape)
    da = gt.get_derivative(tau, eps=eps)
    # print(da)

    # numeric test
    for test_eps in [1e-7, 1e-4, 1e-2, 1e-1]:
        delta = np.random.randint(low=0, high=2, size=tau_shape) * test_eps

        old_returns = gt._compute_returns(tau)
        new_returns = gt._compute_returns(tau + delta)
        delta_returns = new_returns - old_returns

        # approx_delta_returns = [np.inner(delta[i, j, :], tau[i, j, :]) for i in range(num_rollouts)]
        approx_delta_returns = [np.sum(delta[:, i, :] * da[:, i, :]) for i in range(num_rollouts)]

        error = np.abs(np.asarray(delta_returns) - np.asarray(approx_delta_returns))
        print(f'test_eps, delta_norm = {test_eps}, {np.linalg.norm(delta)}')
        # print(f'actual, approx delta returns = {delta_returns}, {approx_delta_returns}')
        print(f'error stats: avg, max = {np.mean(error), np.max(error)}')
        print(f'error pct = {np.mean(error/old_returns)}')
        print()



