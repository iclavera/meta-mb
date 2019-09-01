from meta_mb.utils.serializable import Serializable
from meta_mb.policies.planners.gt_ilqr_planner import GTiLQRPlanner
from meta_mb.policies.planners.mb_gt_ilqr_planner import MBGTiLQRPlanner
import numpy as np
from meta_mb.logger import logger


class GTiLQRController(Serializable):
    def __init__(
            self,
            env,
            eps,
            discount,
            n_parallel,
            initializer_str,
            horizon,
            num_ilqr_iters=100,
            verbose=True,
            dynamics_model=None,
    ):
        Serializable.quick_init(self, locals())
        self.discount = discount
        self.initializer_str = initializer_str
        self.horizon = horizon
        self.num_ilqr_iters = num_ilqr_iters
        self.eps = eps
        self.env = env

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        if dynamics_model is None:
            self.planner = GTiLQRPlanner(env, n_parallel, horizon, eps, self._init_u_array(), verbose=verbose)
        else:
            self.planner = MBGTiLQRPlanner(env, dynamics_model, n_parallel, horizon, eps, self._init_u_array(), verbose=verbose)

    @property
    def vectorized(self):
        return True

    def get_action(self, obs):
        for itr in range(self.num_ilqr_iters):
            optimized_action, backward_accept, forward_accept, planner_returns_log, reward_array = self.planner.update_x_u_for_one_step(obs=obs)

            if backward_accept and forward_accept:
                old_returns, new_returns, diff = planner_returns_log
                logger.logkv('Itr', itr)
                logger.logkv('PlannerPrevReturn', old_returns)
                logger.logkv('PlannerReturn', new_returns)
                logger.logkv('ExpectedDiff', diff)
                logger.logkv('ActualDiff', new_returns - old_returns)
                logger.dumpkvs()

        # shift
        u_new = None  # self._generate_new_u()
        self.planner.shift_u_array(u_new)

        return optimized_action, []

    def _init_u_array(self):
        if self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise NotImplementedError
        return init_u_array

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        """
        This funciton is called by sampler at the start of each trainer iteration.
        :param dones:
        :return:
        """
        pass

    # hack for gt-style mb-ilqr
    def get_actions(self, obs):
        action, _ = self.get_action(obs[0])
        return action[None], []

    def warm_reset(self, u_array):
        pass
        # logger.log('planner resets with collected samples...')
        if u_array is None or np.sum(np.abs(u_array) >= np.mean(self.act_high)) > 0.8 * (self.horizon*self.act_dim):
            u_array = None
        else:
            u_array = u_array[:self.horizon,0, :]
        self.planner.reset_u_array(u_array=u_array)

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
