from meta_mb.logger import logger
import tensorflow as tf
import numpy as np
import time
import copy
from meta_mb.policies import utils


class iLQR(object):
    def __init__(self, env, dynamics_model, policy, horizon, initializer_str,
                 num_ilqr_iters, reg_str='V', use_hessian_f=False, discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0,
                 alpha_init=1.0, alpha_decay_factor=3.0, policy_damping_factor=1e0,
                 c_1=0.3, max_forward_iters=10, max_backward_iters=20, policy_buffer_size=10,
                 verbose=True):
        self._env = copy.deepcopy(env)
        self.dynamics_model = dynamics_model
        self.policy = policy
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
        self.policy_damping_factor = policy_damping_factor
        self.delta_0 = delta_0
        self.delta_init = delta_init
        self.c_1 = c_1
        self.cg_iters = max_backward_iters
        self.max_forward_iters = max_forward_iters
        self.policy_buffer_size = policy_buffer_size
        self.verbose = verbose

        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        build_time = time.time()
        self.x_ph_o = tf.placeholder(
            dtype=tf.float32, shape=(self.obs_dim,), name='x_o',
        )
        self.opt_params_values_var, self.opt_u_var_ha, self.opt_disc_reward_var, self.eff_diff_var, self.opt_accept_var = self._build_action_graph()

        self.x_ph_no = tf.placeholder(
            dtype=tf.float32, shape=(1, self.obs_dim,), name='x_no',
        )
        logger.log('TimeBuildGraph', build_time - time.time())

    def _build_action_graph(self):
        """

        :return: build action graph for one environment
        """
        num_models = self.dynamics_model.num_models
        horizon = self.horizon
        obs_dim, act_dim = self.obs_dim, self.act_dim
        policy = self.policy
        policy_params = policy.get_params()
        policy_params_values = list(policy_params.values())
        policy_params_values_flatten = utils.flatten_params_sym(policy_params)
        policy_params_values_flatten_unflatten = utils.unflatten_params_sym(policy_params_values_flatten, policy_params)

        '''--------------- Compute gradients, x_ho, disc_reward ----------------------'''

        u_ha, x_ho = [], []
        # gradients_list = [tf.TensorArray(dtype=tf.float32, size=horizon, clear_after_read=True) for _ in range(10)]
        gradients_list = [[None for _ in range(horizon)] for _ in range(10)]
        x = self.x_ph_o
        disc_reward = tf.zeros(())

        for i in range(horizon):
            if i == 0:
                u = policy.get_actions_sym(x, policy_params_values_flatten_unflatten)
            else:
                u = policy.get_actions_sym(x)

            if self.use_hessian_f:
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
                    # gradients_list[grad_idx] = gradients_list[grad_idx].write(i, grad)
                    gradients_list[grad_idx][i] = grad
            reward = self._env.tf_reward(x, u, x_prime)
            disc_reward += self.discount**i * reward
            u_ha.append(u)
            x_ho.append(x)
            x = x_prime
        J_val = -disc_reward

        ''' ------------------------- Compute policy gradients --------------------------'''

        u = u_ha[0]
        u_theta = utils.jacobian_wrapper(u, x=policy_params_values_flatten, dim_y=act_dim)
        u_theta_theta = utils.hessian_wrapper(u, x=policy_params_values_flatten, dim_y=act_dim)
        # u_theta = utils.flatten_jac_sym(u_theta, act_dim)
        # flatten first, then call hessian_wrapper
        # u_theta_theta = utils.hessian_wrapper(u, x=policy_params_values, dim_y=act_dim)
        # u_theta_theta = utils.flatten_hess_sym(u_theta_theta, act_dim)

        ''' -------------------- Backward Pass ---------------------------------'''

        # define variables for backward while loop
        V_prime_xx, V_prime_x = tf.zeros((obs_dim, obs_dim)), tf.zeros((obs_dim,))
        open_k_array = tf.TensorArray(
            dtype=tf.float32, size=horizon, element_shape=(act_dim,), tensor_array_name='open_k', clear_after_read=False,
        )
        k_first = None
        closed_K_array = tf.TensorArray(
            dtype=tf.float32, size=horizon, element_shape=(act_dim, obs_dim), tensor_array_name='closed_K', clear_after_read=False,
        )
        delta_J_1, delta_J_2 = tf.zeros(()), tf.zeros(())
        backward_reject = False

        for i in range(horizon-1, -1, -1):
            # f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = [gradients_list[grad_idx].read(i) for grad_idx in range(10)]
            f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = [gradients_list[grad_idx][i] for grad_idx in range(10)]

            if i == 0:
                Q_u = l_u + tf.linalg.matvec(tf.transpose(f_u), V_prime_x)
                if self.use_hessian_f:
                    Q_uu = l_uu + tf.transpose(f_u) @ V_prime_xx @ f_u + tf.tensordot(V_prime_x, f_uu, axes=1)
                    Q_uu = (Q_uu + tf.transpose(Q_uu)) * 0.5
                else:
                    Q_uu = l_uu + tf.transpose(f_u) @ V_prime_xx @ f_u
                Q_theta = tf.linalg.matvec(tf.transpose(u_theta), Q_u)
                Q_theta_theta = tf.transpose(u_theta) @ Q_uu @ u_theta + tf.tensordot(Q_u, u_theta_theta, axes=1)  # FIXME

                if self.verbose:
                    eig_vals = tf.linalg.eigvalsh(Q_theta_theta)
                    Q_theta_theta = tf.Print(Q_theta_theta, data=['eig_vals_theta', tf.reduce_min(eig_vals), tf.reduce_max(eig_vals)])
                Q_theta_theta_reg = Q_theta_theta + tf.eye(tf.shape(Q_theta)[0]) * self.policy_damping_factor
                accept, k = utils.tf_cg(f_Ax=lambda k: tf.linalg.matvec(Q_theta_theta_reg, k), b=-Q_theta, cg_iters=self.cg_iters, residual_tol=1e-10)

                if self.verbose:
                    k = tf.Print(k, data=['backward accept', accept])

                k_first = k
                delta_J_1 += tf.tensordot(k, Q_theta, axes=1)
                delta_J_2 += tf.tensordot(k, tf.linalg.matvec(Q_theta_theta, k), axes=1)

            else:
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

                # use conjugate gradient method to solve for k, K
                if self.verbose:
                    eig_vals = tf.linalg.eigvalsh(Q_uu)
                    Q_uu = tf.Print(Q_uu, data=['eig_vals_u', tf.reduce_min(eig_vals), tf.reduce_max(eig_vals)])
                Q_uu_reg = Q_uu + tf.eye(act_dim) * 1e-3
                accept, Q_uu_reg_inv = utils.tf_cg(f_Ax=lambda Q: Q @ Q_uu_reg, b=tf.eye(act_dim), cg_iters=self.cg_iters, residual_tol=1e-10)
                k = - tf.linalg.matvec(Q_uu_reg_inv, Q_u)
                K = - Q_uu_reg_inv @ Q_ux

                # accept_k, k = utils.tf_cg(f_Ax=lambda k: tf.linalg.matvec(Q_uu_reg, k), b=-Q_u, cg_iters=self.cg_iters, residual_tol=1e-10)
                # accept_K, K = utils.tf_cg(f_Ax=lambda K: Q_uu_reg @ K, b=-Q_ux, cg_iters=self.cg_iters, residual_tol=1e-10)  # TODO: b=-Q_ux_reg?
                #
                # if self.verbose:
                #     k = tf.Print(k, data=['backward accept', tf.logical_and(accept_k, accept_K), 'i', i])

                open_k_array = open_k_array.write(i, k)
                closed_K_array = closed_K_array.write(i, K)
                delta_J_1 += tf.tensordot(k, Q_u, axes=1)
                delta_J_2 += tf.tensordot(k, tf.linalg.matvec(Q_uu, k), axes=1)

                V_x = Q_x + tf.linalg.matvec(tf.transpose(K) @ Q_uu, k) + tf.linalg.matvec(tf.transpose(K), Q_u) + tf.linalg.matvec(tf.transpose(Q_ux), k)
                V_xx = Q_xx + tf.transpose(K) @ Q_uu @ K + tf.transpose(K) @ Q_ux + tf.transpose(Q_ux) @ K
                V_xx = (V_xx + tf.transpose(V_xx)) * 0.5

                # prepare for next iteration
                V_prime_x, V_prime_xx = V_x, V_xx

            backward_reject = tf.logical_or(backward_reject, tf.logical_not(accept))

        ''' ----------------------------- Forward Pass --------------------------------'''

        def body(alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha, prev_opt_policy_param_values):
            x = self.x_ph_o
            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            disc_reward = tf.zeros(())
            opt_u_ha = [None] * horizon
            opt_policy_params = utils.unflatten_params_sym(policy_params_values_flatten + alpha * k_first, policy_params)

            for i in range(horizon):
                if i == 0:
                    u = policy.get_actions_sym(x, opt_policy_params)
                else:
                    u = u_ha[i] + alpha * open_k_array.read(i) + tf.linalg.matvec(closed_K_array.read(i), (x - x_ho[i]))

                u = self._activate_u(u)
                opt_u_ha[i] = u

                x_prime = self.dynamics_model.predict_sym(x[None], u[None], pred_type=tf.random.uniform(shape=(), maxval=num_models, dtype=tf.int32))[0]
                reward = self._env.tf_reward(x, u, x_prime)
                disc_reward += self.discount**i * reward
                x = x_prime

            opt_J_val = -disc_reward

            next_alpha = alpha / self.alpha_decay_factor
            return (next_alpha, opt_J_val, delta_J_alpha, opt_u_ha, list(opt_policy_params.values()))

        # # if the previous iteration improves, break the while loop
        # # 1)
        # cond = lambda alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha, prev_opt_policy_params: tf.math.logical_not(
        #     tf.greater(J_val, prev_opt_J_val)
        # )
        # 2) stop if backward pass rejected or linear search stopping condition satisfied
        # line search condition could be true only if backward pass is accepted
        line_search_accept = lambda alpha, opt_J_val, delta_J_alpha, *args: tf.math.logical_and(
            tf.greater(0., delta_J_alpha), tf.greater(J_val-opt_J_val, -self.c_1*delta_J_alpha)
        )
        cond = lambda *args: tf.math.logical_not(tf.math.logical_or(
            backward_reject,
            line_search_accept(*args),
        ))

        # loop_vars: loop vars with opt_J_val:= J_val, prev_delta_J_alpha:= 0, opt_u_ha:= u_ha
        loop_vars = (self.alpha_init, J_val, 0., u_ha, policy_params_values)
        # terminal_loop_vars: loop vars right before convergence or maximum iteration is reached
        terminal_loop_vars = tf.while_loop(
            cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=self.max_forward_iters
        )
        # if forward_stop is True, forward pass is accepted, update policy parameters
        # otherwise set opt_J_val, opt_u_ha back to J_val, u_ha
        forward_accept = line_search_accept(*terminal_loop_vars)
        _, opt_J_val, _, opt_u_ha, opt_policy_params_values = tf.cond(
            pred=forward_accept,
            true_fn=lambda: terminal_loop_vars,
            false_fn=lambda: loop_vars,
        )

        if self.verbose:
            opt_J_val = tf.Print(opt_J_val, data=['forward accepted', forward_accept])

        return opt_policy_params_values, opt_u_ha, -opt_J_val, J_val-opt_J_val, forward_accept

    def _activate_u(self, u):
        # u = tf.clip_by_value(u, self.act_low, self.act_high)
        scale = (self.act_high - self.act_low) * 0.5  # + 1e-8
        loc = (self.act_high + self.act_low) * 0.5
        return tf.tanh((u-loc)/scale) * scale + loc

    def optimize_policy(self, samples_data, log=True, prefix='', verbose=True):
        t = time.time()
        observations = samples_data['observations']
        idx = np.random.choice(len(observations), self.policy_buffer_size)
        sess = tf.get_default_session()

        for obs in observations[idx]:
            for itr in range(self.num_ilqr_iters):
                opt_params_values, opt_disc_reward, eff_diff, opt_accept = sess.run(
                    [self.opt_params_values_var, self.opt_disc_reward_var, self.eff_diff_var, self.opt_accept_var], feed_dict={self.x_ph_o: obs}
                )
                if verbose:
                    logger.logkv(f'{prefix}RetDyn-{itr}', opt_disc_reward)
                    logger.logkv(f'{prefix}RetImpv-{itr}', eff_diff)

                if opt_accept:
                    self.policy.set_params(opt_params_values)
                else:
                    pass
                    # break

            if verbose:
                logger.logkv(prefix + 'OptPolicyTime', time.time() - t)
                logger.dumpkvs()
