from meta_mb.utils.serializable import Serializable
from meta_mb.samplers.vectorized_env_executor import ParallelPolicyGradUpdateExecutor, ParallelActionDerivativeExecutor
from meta_mb.optimizers.gt_optimizer import GTOptimizer
import matplotlib.pyplot as plt
import numpy as np
from meta_mb.logger import logger
import os


class GTMPCController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            eps,
            num_rollouts=None,
            reward_model=None,
            discount=1,
            method_str='opt_policy',
            n_parallel=1,
            dyn_pred_str='rand',
            initializer_str='uniform',
            reg_coef=1,
            reg_str=None,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            num_opt_iters=8,
            opt_learning_rate=1e-3,
            clip_norm=-1,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.1,
            num_particles=20,
            verbose=True,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.method_str = method_str
        self.dyn_pred_str = dyn_pred_str
        self.initializer_str = initializer_str
        self.reg_coef = reg_coef
        assert 0 <= self.reg_coef <= 1
        self.reg_str = reg_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_cem_iters = num_cem_iters
        self.num_opt_iters = num_opt_iters
        self.opt_learning_rate = opt_learning_rate
        self.eps = eps
        self.num_envs = num_rollouts
        self.percent_elites = percent_elites
        self.num_elites = int(percent_elites * n_candidates)
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha
        self.num_particles = num_particles
        self.clip_norm = clip_norm

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        assert len(env.observation_space.shape) == 1
        assert len(env.action_space.shape) == 1
        self.obs_space_dims = env.observation_space.shape[0]
        self.action_space_dims = env.action_space.shape[0]

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        if self.method_str == 'opt_policy':
            self.policy = ParallelPolicyGradUpdateExecutor(
                env, n_parallel, num_rollouts, horizon, eps,
                opt_learning_rate, num_opt_iters,
                discount, verbose,
            )

        elif self.method_str == 'opt_act':
            if initializer_str == 'uniform':
                self.tau_mean_val = np.random.uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high,
                    size=(self.horizon, self.num_envs, self.action_space_dims),
                )
            elif initializer_str == 'zeros':
                self.tau_mean_val = np.zeros(
                    (self.horizon, self.num_envs, self.action_space_dims),
                )
                self.tau_mean_val = np.clip(np.random.normal(self.tau_mean_val, scale=0.05), self.env.action_space.low, self.env.action_space.high)
            else:
                raise NotImplementedError('initializer_str must be uniform or zeros')

            self.tau_std_val = 0.05 * np.ones((self.action_space_dims,))
            self.deriv_env = ParallelActionDerivativeExecutor(env, n_parallel, horizon, num_rollouts, eps, discount, verbose)
            self.tau_optimizer = GTOptimizer(alpha=self.opt_learning_rate)

            # plotting
            # (num_opt_iters, batch_size)
            self.returns_array_first_rollout, self.grad_norm_first_rollout, self.tau_norm_first_rollout = [], [], []

        else:
            raise NotImplementedError

        self._local_step = 0
        self.save_dir = os.path.join(logger.get_dir(), 'grads_global_norm')
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def vectorized(self):
        return True

    def get_actions(self, observations, deterministic, plot_first_rollout):
        if self.method_str == 'opt_policy':
            return self.get_rollouts_w_policy(observations, deterministic, plot_first_rollout)
        else:
            return self.get_rollouts(observations, deterministic, plot_first_rollout)

    def get_rollouts_w_policy(self, init_obs_array=None, deterministic=False, plot_first_rollout=False):
        assert deterministic
        info = self.policy.do_gradient_steps(init_obs_array)

        if plot_first_rollout:
            logger.log(info['old_return'][0, :])  # report return for the first rollout over optimization iterations
            if init_obs_array is None:
                init_obs = None
            else:
                init_obs = init_obs_array[0]
            self.policy.plot_first_rollout(init_obs, self._local_step)

            # self.plot_info()

        self._local_step += 1
        return np.asarray(info['old_return'])[:, -1]

    def get_rollouts(self, init_obs_array=None, deterministic=False, plot_first_rollout=False):
        """

        :param observations:
        :param deterministic:
        :param plot_first_rollout:
        :return: rollouts = num_envs experts = (horizon==max_path_length, batch_size==num_envs, act_dims)
                 returns_array = [()] * num_opt_iters
        """
        if deterministic:
            tau = self.tau_mean_val
        else:
            tau = self.tau_mean_val + np.random.normal(size=np.shape(self.tau_mean_val)) * self.tau_std_val
            tau = np.clip(tau, a_min=self.env.action_space.low, a_max=self.env.action_space.high)

        returns_array, grad_norm_first_rollout, tau_norm_first_rollout = [], [], []
        for itr in range(self.num_opt_iters):
            grad_tau, returns = self.deriv_env.get_derivative(tau, init_obs_array=init_obs_array)
            tau += self.tau_optimizer.compute_delta_var(grad_tau)  # Adam optimizer
            # regularization
            # clipping and regularization needs to be modified if not (low, high) = (-1, 1)
            if self.reg_str == 'poly':
                tau = np.clip(tau, a_min=self.env.action_space.low - self.reg_coef, a_max=self.env.action_space.high + self.reg_coef)
                tau -= self.reg_coef * tau**5
            elif self.reg_str == 'scale':
                _max_abs = np.max(np.abs(tau), axis=-1)
                tau /= _max_abs
            elif self.reg_str == 'tanh':
                tau = np.tanh(tau)
            else:
                raise NotImplementedError
            returns_array.append(returns)
            grad_norm_first_rollout.append(np.linalg.norm(grad_tau[:, 0, :], axis=-1))  # list item = (horizon,)
            tau_norm_first_rollout.append(np.linalg.norm(tau[:, 0, :], axis=-1))

        returns_array = np.vstack(returns_array)  # (num_opt_iters, batch_size)

        self.returns_array_first_rollout.extend(returns_array[:, 0])  # [()] * trainer_iterations
        self.grad_norm_first_rollout.extend(grad_norm_first_rollout)  # [(horizon,)] * trainer_iterations
        self.tau_norm_first_rollout.extend(tau_norm_first_rollout)  # [(horizon,)] * trainer_iterations

        if plot_first_rollout:
            logger.log(returns_array[:, 0])  # report return for the first rollout over optimization iterations
            if init_obs_array is None:
                init_obs = None
            else:
                init_obs = init_obs_array[0]
            self.dynamics_model.plot_rollout(tau[:, 0, :], init_obs, self._local_step)

            self.plot_info()

        # Do not rotate
        # if self.initializer_str == 'uniform':
        #     self.tau_mean_val = np.concatenate([
        #         tau[1:],
        #         np.random.uniform(low=self.env.action_space.low, high=self.env.action_space.high, size=(1, self.num_envs, self.action_space_dims)),
        #     ], axis=0)
        # else:
        #     self.tau_mean_val = np.concatenate([
        #         tau[1:],
        #         np.zeros((1, self.num_envs, self.action_space_dims)),
        #     ], axis=0)

        self.tau_mean_val = tau
        # adapt tau_std_val here
        self._local_step += 1
        return returns_array[-1, :]  # total rewards for all envs in the batch, with the latest policy

    def plot_info(self):
        # x: iterations, y: stats average over batch
        x = np.arange(len(self.returns_array_first_rollout))  # plot every 10 steps along the path
        returns = self.returns_array_first_rollout
        # idx_horizon_array = np.arange(0, self.horizon, 10)
        grad_norm = np.stack(self.grad_norm_first_rollout, axis=-1)  # (horizon, trainer_iterations)
        tau_norm = np.stack(self.tau_norm_first_rollout, axis=-1)  # (horizon, trainer_iterations)

        self.returns_array_first_rollout = []
        self.grad_norm_first_rollout = []
        self.tau_norm_first_rollout = []

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))
        axes = axes.flatten()

        ax = axes[0]
        ax.plot(x, returns)
        ax.set_xlabel('trainer_iterations')
        ax.set_ylabel('sum_rewards_for_first_rollout')

        # ax = axes[1]
        # for idx in idx_horizon_array:
        #     ax.plot(x, grad_norm[idx, :], label=idx)
        # ax.set_xlabel('trainer_iterations')
        # ax.set_ylabel('grad_norm')
        # ax.legend(title='path_length_so_far')
        #
        # ax = axes[2]
        # tau_norm[]
        # for idx in idx_horizon_array:
        #     ax.plot(x, tau_norm[idx, :], label=idx)
        # ax.set_xlabel('trainer_iterations')
        # ax.set_ylabel('avg_action_norm')
        # ax.legend(title='path_length_so_far')

        ax = axes[1]
        ax.plot(np.arange(0, self.horizon), np.mean(grad_norm, axis=-1))
        ax.set_xlabel('horizon')
        ax.set_ylabel(f'avg_grad_norm_over_{len(x)}_iteration')

        ax = axes[2]
        ax.plot(np.arange(0, self.horizon), np.mean(tau_norm, axis=-1))
        ax.set_xlabel('horizon')
        ax.set_ylabel(f'avg_action_norm_over_{len(x)}_iteration')

        fig.suptitle(f'{self._local_step}')
        fig.savefig(os.path.join(self.save_dir, f'{self._local_step}.png'))
        logger.log('plt saved to', os.path.join(self.save_dir, f'{self._local_step}.png'))

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        if self.method_str == 'opt_act':
            assert len(dones) == self.num_envs
            if self.initializer_str == 'uniform':
                self.tau_mean_val = np.random.uniform(
                    low=self.env.action_space.low,
                    high=self.env.action_space.high,
                    size=(self.horizon, self.num_envs, self.action_space_dims),
                )
            elif self.initializer_str == 'zeros':
                self.tau_mean_val = np.zeros(
                    (self.horizon, self.num_envs, self.action_space_dims),
                )

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
