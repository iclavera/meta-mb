from meta_mb.logger import logger
import scipy.linalg as sla
import numpy as np
import tensorflow as tf
import time
import copy
from meta_mb.policies import utils


class iLQRPlanner(object):
    def __init__(self, env, dynamics_model, num_envs, horizon, initializer_str,
                 num_ilqr_iters, reg_str='V', use_hessian_f=False, discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0,
                 alpha_init=1.0, alpha_decay_factor=3.0,
                 c_1=1e-7, max_forward_iters=10, max_backward_iters=10,
                 n_candidates=1000, num_cem_iters=5, cem_deterministic_policy=True, cem_alpha=0.1, percent_elites=0.1,
                 verbose=False):
        self._env = copy.deepcopy(env)
        self.dynamics_model = dynamics_model
        self.num_envs = num_envs
        self.horizon = horizon
        self.initializer_str = initializer_str
        self.num_ilqr_iters = num_ilqr_iters
        self.reg_str = reg_str
        self.use_hessian_f = use_hessian_f
        self.discount = discount
        self.act_dim = env.action_space.shape[0]
        self.obs_dim = env.observation_space.shape[0]
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_init = mu_init
        self.alpha_init = alpha_init
        self.alpha_decay_factor = alpha_decay_factor
        self.delta_0 = delta_0
        self.delta_init = delta_init
        self.c_1 = c_1
        self.cg_iters = max_backward_iters
        self.max_forward_iters = max_forward_iters
        self.verbose = verbose
        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        # cem
        self.n_candidates = n_candidates
        self.num_cem_iters = num_cem_iters
        self.cem_deterministic_policy = cem_deterministic_policy
        self.cem_alpha = cem_alpha
        self.num_elites = int(percent_elites * n_candidates)

        build_time = time.time()
        self.u_ph_ha = tf.placeholder(
            dtype=tf.float32, shape=(self.horizon, self.act_dim), name='u_ha',
        )
        self.x_ph_o = tf.placeholder(
            dtype=tf.float32, shape=(self.obs_dim,), name='x_o',
        )
        self.u_val_hna = None
        self.opt_u_var_ha, self.opt_disc_reward_var, self.eff_diff_var, self.opt_accept_var = self._build_action_graph()

        self.x_ph_no = tf.placeholder(
            dtype=tf.float32, shape=(self.num_envs, self.obs_dim,), name='x_no',
        )
        self.cem_opt_u_var_ha = self._build_cem_graph()
        logger.log('TimeBuildGraph', build_time - time.time())

    def _build_cem_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        n_candidates = self.n_candidates
        num_envs = self.num_envs
        alpha = self.cem_alpha

        mean = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high + self.act_low) / 2
        var = tf.ones(shape=[horizon, num_envs, 1, act_dim]) * (self.act_high - self.act_low) / 16

        init_obs = tf.reshape(
            tf.tile(tf.expand_dims(self.x_ph_no, axis=1), [1, n_candidates, 1]),
            shape=(num_envs*n_candidates, obs_dim),
        )

        for itr in range(self.num_cem_iters):
            lb_dist, ub_dist = mean - self.act_low, self.act_high - mean
            constrained_var = tf.minimum(tf.minimum(tf.square(lb_dist / 2), tf.square(ub_dist / 2)), var)
            std = tf.sqrt(constrained_var)
            act = mean + tf.random.normal(shape=[horizon, num_envs, n_candidates, act_dim]) * std
            act = tf.clip_by_value(act, self.act_low, self.act_high)
            act = tf.reshape(act, shape=(horizon, num_envs*n_candidates, act_dim))
            disc_rewards = tf.zeros((num_envs*n_candidates,))

            obs = init_obs
            for t in range(horizon):
                next_obs = self.dynamics_model.predict_sym(obs, act[t])
                rewards = self._env.tf_reward(obs, act[t], next_obs)
                disc_rewards += self.discount**t * rewards
                obs = next_obs

            # Re-fit belief to the best ones
            disc_rewards = tf.reshape(disc_rewards, (num_envs, n_candidates))
            _, indices = tf.nn.top_k(disc_rewards, k=self.num_elites, sorted=False)
            act = tf.reshape(act, shape=(horizon, num_envs, n_candidates, act_dim))
            act = tf.transpose(act, (1, 2, 3, 0))  # (num_envs, n_candidates, act_dim, horizon)
            elite_actions = tf.batch_gather(act, indices)
            elite_actions = tf.transpose(elite_actions, (3, 0, 1, 2))  # (horizon, num_envs, n_candidates, act_dim)
            elite_mean, elite_var = tf.nn.moments(elite_actions, axes=[2], keep_dims=True)
            mean = mean * alpha + elite_mean * (1 - alpha)
            var = var * alpha + elite_var * (1 - alpha)

        optimized_actions_mean = mean[:, :, 0, :]
        if self.cem_deterministic_policy:
            return optimized_actions_mean
        else:
            optimized_actions_var = var[:, :, 0, :]
            return optimized_actions_mean + \
                   tf.random.normal(shape=tf.shape(optimized_actions_mean)) * tf.sqrt(optimized_actions_var)

    def _build_action_graph(self):
        """

        :return: build action graph for one environment
        """
        num_models = self.dynamics_model.num_models
        horizon = self.horizon
        obs_dim, act_dim = self.obs_dim, self.act_dim

        '''--------------- Collect u_ha, x_ho, gradients, J_val ----------------------'''

        u_ha = self.u_ph_ha
        x_ho = []
        gradients_list = [tf.TensorArray(dtype=tf.float32, size=horizon, clear_after_read=True) for _ in range(10)]
        x = self.x_ph_o
        disc_reward = tf.zeros(())

        for i in range(horizon):
            if self.use_hessian_f:
                u = u_ha[i, :]
                x_u_concat = tf.concat([x, u], axis=0)
                x, u = x_u_concat[:obs_dim], x_u_concat[-act_dim:]
                x_prime = self.dynamics_model.predict_sym(
                    x[None], u[None], pred_type=tf.random.uniform(shape=(), maxval=num_models, dtype=tf.int32)
                )[0]

                hess = utils.hessian_wrapper(x_prime, x_u_concat, obs_dim, obs_dim+act_dim)
                f_x = utils.jacobian_wrapper(x_prime, x, obs_dim, obs_dim)
                f_u = utils.jacobian_wrapper(x_prime, u, obs_dim, act_dim)
                f_xx = hess[:, :obs_dim, :obs_dim]
                f_uu = hess[:, -act_dim:, -act_dim:]
                f_ux = (hess[:, -act_dim:, :obs_dim] + tf.transpose(hess[:, :obs_dim, -act_dim:], perm=[0, 2, 1])) * 0.5

            else:
                u = u_ha[i, :]
                x_prime = self.dynamics_model.predict_sym(
                    x[None], u[None], pred_type=tf.random.uniform(shape=(), maxval=num_models, dtype=tf.int32)
                )[0]

                # compute gradients
                f_x = utils.jacobian_wrapper(x_prime, x, obs_dim, obs_dim)
                f_u = utils.jacobian_wrapper(x_prime, u, obs_dim, act_dim)
                f_xx = None
                f_uu = None
                f_ux = None

            # store
            df = [f_x, f_u, f_xx, f_uu, f_ux]
            dl = list(self._env.tf_dl(x, u, x_prime))
            for grad_idx, grad in enumerate(df + dl):
                if grad is not None:
                    grad = tf.dtypes.cast(grad, tf.float32)
                    gradients_list[grad_idx] = gradients_list[grad_idx].write(i, grad)
            reward = self._env.tf_reward(x, u, x_prime)
            disc_reward += self.discount**i * reward
            x_ho.append(x)
            x = x_prime
        J_val = -disc_reward

        ''' -------------------- Backward Pass ---------------------------------'''

        def body(prev_reject, i, open_k_array, closed_K_array, delta_J_1, delta_J_2, V_prime_x, V_prime_xx):
            f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = [gradients_list[grad_idx].read(i) for grad_idx in range(10)]
            Q_x = l_x + tf.linalg.matvec(tf.transpose(f_x), V_prime_x)
            Q_u = l_u + tf.linalg.matvec(tf.transpose(f_u), V_prime_x)
            if self.use_hessian_f:
                Q_xx = l_xx + tf.transpose(f_x) @ V_prime_xx @ f_x + tf.tensordot(V_prime_x, f_xx, axes=1)
                Q_xx = (Q_xx + tf.transpose(Q_xx)) * 0.5
                Q_uu = l_uu + tf.transpose(f_u) @ V_prime_xx @ f_u + tf.tensordot(V_prime_x, f_uu, axes=1)
                Q_uu = (Q_uu + tf.transpose(Q_uu)) * 0.5
                Q_ux = l_ux + tf.transpose(f_u) @ V_prime_xx @ f_x + tf.tensordot(V_prime_x, f_ux, axes=1)
            else:
                Q_xx = l_xx + tf.transpose(f_x) @ V_prime_xx @ f_x
                Q_uu = l_uu + tf.transpose(f_u) @ V_prime_xx @ f_u
                Q_ux = l_ux + tf.transpose(f_u) @ V_prime_xx @ f_x

            # test methods to deal with non-P.D. Q_uu
            # 1) trust tf_cg even if it does not converge
            # accept_k, k = utils.tf_cg(f_Ax=lambda k: tf.linalg.matvec(Q_uu, k), b=-Q_u, cg_iters=self.cg_iters, residual_tol=1e-10)
            # accept_K, K = utils.tf_cg(f_Ax=lambda K: Q_uu @ K, b=-Q_ux, cg_iters=self.cg_iters, residual_tol=1e-10)
            # # set new_reject = False and do the next step in backward pass if and only if both cg converge
            # reject = tf.logical_not(tf.logical_and(accept_k, accept_K))

            # 2) use Q_uu_reg = Q_uu + mu * id
            Q_uu_reg = Q_uu + tf.eye(act_dim,) * 1e-5
            # debugging: log min, max eigen values
            if self.verbose:
                eig_vals = tf.linalg.eigvalsh(Q_uu_reg)
                Q_uu_reg = tf.Print(Q_uu_reg, data=['eig_vals', tf.reduce_min(eig_vals), tf.reduce_max(eig_vals)])
            accept, Q_uu_reg_inv = utils.tf_cg(f_Ax=lambda Q: Q @ Q_uu_reg, b=tf.eye(act_dim,), cg_iters=self.cg_iters, residual_tol=1e-10)
            k = - tf.linalg.matvec(Q_uu_reg_inv, Q_u)
            K = - Q_uu_reg_inv @ Q_ux
            reject = tf.logical_not(accept)

            # 3) use adaptive mu as in the original paper

            open_k_array = open_k_array.write(i, k)
            closed_K_array = closed_K_array.write(i, K)
            delta_J_1 += tf.tensordot(k, Q_u, axes=1)
            delta_J_2 += tf.tensordot(k, tf.linalg.matvec(Q_uu, k), axes=1)

            V_x = Q_x + tf.linalg.matvec(tf.transpose(K) @ Q_uu, k) + tf.linalg.matvec(tf.transpose(K), Q_u) + tf.linalg.matvec(tf.transpose(Q_ux), k)
            V_xx = Q_xx + tf.transpose(K) @ Q_uu @ K + tf.transpose(K) @ Q_ux + tf.transpose(Q_ux) @ K
            # V_xx = (V_xx + tf.transpose(V_xx)) * 0.5

            return (reject, i-1, open_k_array, closed_K_array, delta_J_1, delta_J_2, V_x, V_xx)

        # define variables for backward while loop
        V_prime_xx, V_prime_x = tf.zeros((obs_dim, obs_dim)), tf.zeros((obs_dim,))
        open_k_array = tf.TensorArray(
            dtype=tf.float32, size=horizon, element_shape=(act_dim,), tensor_array_name='open_k', clear_after_read=False,
        )
        closed_K_array = tf.TensorArray(
            dtype=tf.float32, size=horizon, element_shape=(act_dim, obs_dim), tensor_array_name='closed_K', clear_after_read=False,
        )

        # test conditions
        # 1) while conjugate gradient descent not rejected, do another step within the backward pass, i = horizon-1, ..., 0
        # cond = lambda prev_reject, *args: tf.logical_not(prev_reject)
        # 2) always accept conjugate gradient
        cond = lambda *args: True
        loop_vars = (False, horizon-1, open_k_array, closed_K_array, tf.zeros(()), tf.zeros(()), V_prime_x, V_prime_xx)
        cg_reject, _, open_k_array, closed_K_array, delta_J_1, delta_J_2, _, _ = tf.while_loop(cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=horizon)

        # debug logging
        # if self.verbose:
        #     delta_J_1 = tf.Print(delta_J_1, data=['cj_reject', cg_reject])

        ''' ----------------------------- Forward Pass --------------------------------'''

        def body(alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha):
            x = self.x_ph_o
            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            disc_reward = tf.zeros(())
            opt_u_ha = [None] * horizon
            for i in range(horizon):
                u = u_ha[i] + alpha * open_k_array.read(i) + tf.linalg.matvec(closed_K_array.read(i), (x - x_ho[i]))
                u = self._activate_u(u)
                opt_u_ha[i] = u

                x_prime = self.dynamics_model.predict_sym(x[None], u[None], pred_type=tf.random.uniform(shape=(), maxval=num_models, dtype=tf.int32))[0]
                reward = self._env.tf_reward(x, u, x_prime)
                disc_reward += self.discount**i * reward
                x = x_prime
            opt_J_val = -disc_reward
            opt_u_ha= tf.stack(opt_u_ha)

            next_alpha = alpha / self.alpha_decay_factor
            return (next_alpha, opt_J_val, delta_J_alpha, opt_u_ha)

        # if the previous iteration converges, break the while loop
        # test conditions
        # 1) 0 > delta_J and J - opt_J > c_1 * (-delta_J)
        cond = lambda alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha: tf.math.logical_not(
            tf.math.logical_and(tf.greater(0., prev_delta_J_alpha), tf.greater(J_val-prev_opt_J_val, -self.c_1*prev_delta_J_alpha))
        )
        # 2) J - opt_J > 0, where opt_J is computed by randomly selected models
        # cond = lambda alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha: tf.math.logical_not(
        #     tf.greater(J_val, prev_opt_J_val)
        # )
        # 3) J - opt_J > 0, where opt_J is computed individually with all models and take a percentile
        # this is too slow
        # def compute_J_val_m(prev_opt_u_ha):
        #     x_mo = tf.stack([self.x_ph_o for _ in range(num_models)], axis=0)
        #     disc_reward_m = tf.zeros((num_models,))
        #     for i in range(horizon):
        #         u_ma = tf.stack([prev_opt_u_ha[i] for _ in range(num_models)], axis=0)
        #         x_prime_mo = self.dynamics_model.predict_batches_sym(x_mo, u_ma)
        #         reward_m = self._env.tf_reward(x_mo, u_ma, x_prime_mo)
        #         disc_reward_m += self.discount**i * reward_m
        #         x_mo = x_prime_mo
        #     return -disc_reward_m
        #
        # def cond(alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha):
        #     # q > 70 gives more aggressive updates
        #     prev_opt_J_val_m = compute_J_val_m(prev_opt_u_ha)
        #     num_forward_accept = tf.reduce_sum(tf.dtypes.cast(tf.greater(J_val_m, prev_opt_J_val_m), tf.float32))
        #     if self.verbose:
        #         num_forward_accept = tf.Print(num_forward_accept, data=['num forward accept', num_forward_accept])
        #     forward_accept = tf.greater(tf.reduce_sum(num_forward_accept), num_models*self.c_1)
        #     return tf.math.logical_not(forward_accept)

        # if self.verbose:
        #     u_ha = tf.Print(u_ha, data=['starting loop with delta_J_1', delta_J_1, 'delta_J_2', delta_J_2])
        # J_val_m = compute_J_val_m(u_ha)
        loop_vars = (self.alpha_init, J_val, 0., u_ha)
        terminal_loop_vars = tf.while_loop(
            cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=self.max_forward_iters
        )
        # if forward_stop is True, forward pass is accepted, otherwise set opt_J_val, opt_u_ha back to J_val, u_ha and no optimization is carried out
        # terminal_loop_vars: loop vars right before convergence or maximum iteration is reached
        forward_accept = tf.math.logical_not(cond(*terminal_loop_vars))
        _, opt_J_val, _, opt_u_ha = tf.cond(
            pred=forward_accept,
            true_fn=lambda: terminal_loop_vars,
            false_fn=lambda: loop_vars,
        )

        return opt_u_ha, -opt_J_val, J_val-opt_J_val, forward_accept

    def _activate_u(self, u):
        # u = tf.clip_by_value(u, self.act_low, self.act_high)
        scale = (self.act_high - self.act_low) * 0.5  # + 1e-8
        loc = (self.act_high + self.act_low) * 0.5
        u = tf.tanh((u-loc)/scale) * scale + loc
        return u

    def get_actions(self, obs_no, verbose=True):
        if self.u_val_hna is None:
            self._reset(obs_no)
        sess = tf.get_default_session()

        t = time.time()
        for env_idx in range(self.num_envs):
            u_val_ha = self.u_val_hna[:, env_idx, :]
            x_val_o = obs_no[env_idx, :]
            for itr in range(self.num_ilqr_iters):
                u_val_ha, opt_disc_reward, eff_diff, opt_accept = sess.run(
                    [self.opt_u_var_ha, self.opt_disc_reward_var, self.eff_diff_var, self.opt_accept_var],
                    feed_dict={self.u_ph_ha: u_val_ha, self.x_ph_o: x_val_o}
                )
                if verbose:
                    logger.logkv(f'RetDyn-{env_idx}-{itr}', opt_disc_reward)
                    logger.logkv(f'RetImpv-{env_idx}-{itr}', eff_diff)

                # break ilqr loop once optimization fails
                # if not opt_accept:
                #     break
            self.u_val_hna[:, env_idx, :] = u_val_ha

        if verbose:
            logger.logkv('iLQRTime', time.time() - t)
            logger.dumpkvs()

        optimized_actions = self.u_val_hna[0]  # (num_envs, act_dim)

        # shift u_array and perm
        self._shift()  # FIXME: alternative ways to initialize u_new

        return optimized_actions, []

    def _shift(self):
        #  shift u_array
        if self.initializer_str == 'cem':
            self.u_val_hna = None
        else:
            if self.initializer_str == 'uniform':
                u_new = np.random.uniform(low=self.act_low, high=self.act_high, size=(self.num_envs, self.act_dim))
            elif self.initializer_str == 'zeros':
                u_new = np.random.normal(scale=0.1, size=(self.num_envs, self.act_dim))
                u_new = np.clip(u_new, a_min=self.act_low, a_max=self.act_high)
            else:
                raise ValueError
            self.u_val_hna = np.concatenate([self.u_val_hna[1:, :, :], u_new[None]])

    def _reset(self, obs):
        if self.initializer_str == 'cem':
            logger.log('initialize with cem...')
            sess = tf.get_default_session()
            init_u_array, = sess.run([self.cem_opt_u_var_ha], feed_dict={self.x_ph_o: obs})
        elif self.initializer_str == 'uniform':
            init_u_array = np.random.uniform(low=self.act_low, high=self.act_high,
                                             size=(self.horizon, self.num_envs, self.act_dim))
        elif self.initializer_str == 'zeros':
            init_u_array = np.random.normal(scale=0.1, size=(self.horizon, self.num_envs, self.act_dim))
            init_u_array = np.clip(init_u_array, a_min=self.act_low, a_max=self.act_high)
        else:
            raise ValueError

        self.u_val_hna = init_u_array