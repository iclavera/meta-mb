from meta_mb.utils.serializable import Serializable
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
            num_rollouts=None,
            reward_model=None,
            discount=1,
            method_str='opt_policy',
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
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.method_str = method_str
        self.dyn_pred_str = dyn_pred_str
        self.initializer_str = initializer_str
        self.reg_coef = reg_coef
        self.reg_str = reg_str
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.num_cem_iters = num_cem_iters
        self.num_opt_iters = num_opt_iters
        self.opt_learning_rate = opt_learning_rate
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
        else:
            raise NotImplementedError('initializer_str must be uniform or zeros')

        self.tau_std_val = 0.05 * np.ones((self.action_space_dims,))
        self._local_step = 0

        # plotting
        # (num_opt_iters, batch_size)
        self.returns_array_first_rollout, self.grad_norm_first_rollout, self.tau_norm_first_rollout = [], [], []
        self.save_dir = os.path.join(logger.get_dir(), 'grads_global_norm')
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]
        return self.get_actions(observation)

    def get_rollouts(self, observations, deterministic=False, plot_first_rollout=False):
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
            grad_tau, returns = self.dynamics_model.get_derivative(tau, init_obs=observations)
            tau += self.opt_learning_rate * grad_tau
            tau = np.clip(tau, a_min=self.env.action_space.low, a_max=self.env.action_space.high)
            returns_array.append(returns)
            grad_norm_first_rollout.append(np.linalg.norm(grad_tau[:, 0, :], axis=-1))  # list item = (horizon,)
            tau_norm_first_rollout.append(np.linalg.norm(tau[:, 0, :], axis=-1))

        returns_array = np.vstack(returns_array)  # (num_opt_iters, batch_size)

        self.returns_array_first_rollout.extend(returns_array[:, 0])  # [()] * trainer_iterations
        self.grad_norm_first_rollout.extend(grad_norm_first_rollout)  # [(horizon,)] * trainer_iterations
        self.tau_norm_first_rollout.extend(tau_norm_first_rollout)  # [(horizon,)] * trainer_iterations

        if plot_first_rollout:
            logger.log(returns_array[:, 0])  # report return for the first rollout over optimization iterations
            if observations is None:
                init_obs = None
            else:
                init_obs = observations[0]
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

        self._local_step += 1
        return tau, returns_array[-1, :]  # total rewards for all envs in the batch, with the latest policy

    def get_actions(self, observations, deterministic=False, return_first_info=False, log_grads_for_plot=False):
        raise NotImplementedError
    #
    # def get_random_action(self, n):
    #     return np.random.uniform(low=self.env.action_space.low,
    #                              high=self.env.action_space.high, size=(n,) + self.env.action_space.low.shape)
    #
    #
    # def get_sinusoid_actions(self, action_space, t):
    #     actions = np.array([])
    #     delta = t/action_space
    #     for i in range(action_space):
    #         #actions = np.append(actions, 0.5 * np.sin(i * delta)) #two different ways of sinusoidal sampling
    #         actions = np.append(actions, 0.5 * np.sin(i * t))
    #     #for i in range(3, len(actions)): #limit movement to first 3 joints
    #     #    actions[i] = 0
    #     return actions
    #
    # def predict_open_loop(self, init_obs, tau):
    #     return self.dynamics_model.predict_open_loop(init_obs, tau)
    #
    def plot_info(self):
        # x: iterations, y: stats average over batch
        x = np.arange(len(self.returns_array_first_rollout))  # plot every 10 steps along the path
        returns = self.returns_array_first_rollout
        idx_horizon_array = np.arange(0, self.horizon, 10)
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
        ax.set_ylabel('one_step_reward_for_first_expert')

        ax = axes[1]
        for idx in idx_horizon_array:
            ax.plot(x, grad_norm[idx, :], label=idx)
        ax.set_xlabel('trainer_iterations')
        ax.set_ylabel('grad_norm')
        ax.legend(title='path_length_so_far')

        ax = axes[2]
        for idx in idx_horizon_array:
            ax.plot(x, tau_norm[idx, :], label=idx)
        ax.set_xlabel('trainer_iterations')
        ax.set_ylabel('tau_norm')
        ax.legend(title='path_length_so_far')

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
