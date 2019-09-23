from meta_mb.logger import logger
import tensorflow as tf
import numpy as np
import time
import copy
from meta_mb.policies import utils


class GPS(object):
    def __init__(self, env, dynamics_model, policy, horizon, initializer_str,
                 num_ilqr_iters, reg_str='V', use_hessian_f=False, discount=1,
                 mu_min=1e-6, mu_max=1e10, mu_init=1e-5, delta_0=2, delta_init=1.0,
                 alpha_init=1.0, alpha_decay_factor=3.0, policy_damping_factor=1e0,
                 c_1=0.3, max_forward_iters=10, max_backward_iters=20, policy_buffer_size=10,
                 use_hessian_policy=False, damping_str='Q', learning_rate=1e-3,
                 batch_size=10, num_gradient_steps=5,
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
        self.damping_str = damping_str
        self.delta_0 = delta_0
        self.delta_init = delta_init
        self.c_1 = c_1
        self.cg_iters = max_backward_iters
        self.max_forward_iters = max_forward_iters
        self.policy_buffer_size = policy_buffer_size
        self.use_hessian_policy = use_hessian_policy
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_gradient_steps = num_gradient_steps
        self.verbose = verbose

        self.act_low, self.act_high = env.action_space.low, env.action_space.high

        assert policy_buffer_size % dynamics_model.num_models == 0

        build_time = time.time()

        '''------------- Build graph to rollout trajectory under policy and dynamics model ---------------'''

        self.obs_ph_no = tf.placeholder(
            dtype=tf.float32, shape=(None, self.obs_dim), name='obs',
        )
        self.acts_var_nha = self._build_action_graph()

        '''---------------------------------  Build graph for iLQR optimization ---------------------------'''

        self.x_ph_o = tf.placeholder(
            dtype=tf.float32, shape=(self.obs_dim,), name='x_o',
        )
        self.u_ph_ha = tf.placeholder(
            dtype=tf.float32, shape=(horizon, self.act_dim), name='u_ha',
        )
        self.opt_u_var_ha, self.opt_x_var_ho, self.opt_disc_reward_var, self.eff_diff_var, self.opt_accept_var \
            = self._build_opt_graph()

        '''------------------------------- Build graph for supervised learning ----------------------------'''

        self.exp_acts_ph_na = tf.placeholder(
            dtype=tf.float32, shape=(None, self.act_dim), name='exp_acts',
        )
        self.exp_obs_ph_no = tf.placeholder(
            dtype=tf.float32, shape=(None, self.obs_dim), name='exp_obs',
        )
        self.loss, self.train_op = self._build_policy_loss()

        logger.log('TimeBuildGraph', build_time - time.time())

    def _build_action_graph(self):
        x_no = self.obs_ph_no
        u_hna = []
        for i in range(self.horizon):
            u_na = self.policy.get_actions_sym(x_no)
            u_hna.append(u_na)
            x_no = self.dynamics_model.predict_sym(x_no, u_na)

        u_hna = tf.stack(u_hna, axis=0)
        u_nha = tf.transpose(u_hna, perm=[1, 0, 2])
        return u_nha

    def _build_opt_graph(self):
        """

        :return: build action graph for one environment
        """
        num_models = self.dynamics_model.num_models
        horizon = self.horizon
        obs_dim, act_dim = self.obs_dim, self.act_dim

        '''--------------- Compute gradients, x_ho, disc_reward ----------------------'''

        x = self.x_ph_o
        u_ha = self.u_ph_ha
        x_ho = []
        # gradients_list = [tf.TensorArray(dtype=tf.float32, size=horizon, clear_after_read=True) for _ in range(10)]
        gradients_list = [[None for _ in range(horizon)] for _ in range(10)]
        disc_reward = tf.zeros(())

        for i in range(horizon):
            u = u_ha[i, :]

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
            x_ho.append(x)
            x = x_prime

        if self.verbose:
            print_ops = [tf.print('\n----disc_reward before opt', disc_reward)]
        else:
            print_ops = []
        with tf.control_dependencies(print_ops):
            J_val = -disc_reward
            x_ho = tf.stack(x_ho, axis=0)

        '''-------------------- Backward Pass ---------------------------------'''

        # define variables for backward while loop
        V_prime_xx, V_prime_x = tf.zeros((obs_dim, obs_dim)), tf.zeros((obs_dim,))
        open_k_array = tf.TensorArray(
            dtype=tf.float32, size=horizon, element_shape=(act_dim,), tensor_array_name='open_k', clear_after_read=False,
        )
        closed_K_array = tf.TensorArray(
            dtype=tf.float32, size=horizon, element_shape=(act_dim, obs_dim), tensor_array_name='closed_K', clear_after_read=False,
        )
        delta_J_1, delta_J_2 = tf.zeros(()), tf.zeros(())
        backward_reject = False

        for i in range(horizon-1, -1, -1):
            # f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = [gradients_list[grad_idx].read(i) for grad_idx in range(10)]
            f_x, f_u, f_xx, f_uu, f_ux, l_x, l_u, l_xx, l_uu, l_ux = [gradients_list[grad_idx][i] for grad_idx in range(10)]

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
            # if self.verbose:
            #     eig_vals = tf.linalg.eigvalsh(Q_uu)
            #     Q_uu = tf.Print(Q_uu, data=['eig_vals_u', tf.reduce_min(eig_vals), tf.reduce_max(eig_vals)])
            Q_uu_reg = Q_uu + tf.eye(act_dim) * self.mu_init
            Q_ux_reg = Q_ux
            accept, Q_uu_reg_inv = utils.tf_cg(f_Ax=lambda Q: Q @ Q_uu_reg, b=tf.eye(act_dim), cg_iters=self.cg_iters, residual_tol=1e-10)
            k = - tf.linalg.matvec(Q_uu_reg_inv, Q_u)
            K = - Q_uu_reg_inv @ Q_ux_reg

            open_k_array = open_k_array.write(i, k)
            closed_K_array = closed_K_array.write(i, K)
            delta_J_1 += tf.tensordot(k, Q_u, axes=1)
            delta_J_2 += tf.tensordot(k, tf.linalg.matvec(Q_uu, k), axes=1)

            V_x = Q_x + tf.linalg.matvec(tf.transpose(K) @ Q_uu, k) + tf.linalg.matvec(tf.transpose(K), Q_u) + tf.linalg.matvec(tf.transpose(Q_ux), k)
            V_xx = Q_xx + tf.transpose(K) @ Q_uu @ K + tf.transpose(K) @ Q_ux + tf.transpose(Q_ux) @ K
            V_xx = (V_xx + tf.transpose(V_xx)) * 0.5

            # prepare for next iteration
            V_prime_x, V_prime_xx = V_x, V_xx

            # TODO: if ever rejected, break out of the for loop
            backward_reject = tf.logical_or(backward_reject, tf.logical_not(accept))

        if self.verbose:
            backward_reject = tf.Print(backward_reject, data=['backward accept', tf.logical_not(backward_reject)])

        '''----------------------------- Forward Pass --------------------------------'''

        def body(alpha, prev_opt_J_val, prev_delta_J_alpha, prev_opt_u_ha, prev_opt_x_ho):
            x = self.x_ph_o
            delta_J_alpha = alpha * delta_J_1 + 0.5 * alpha**2 * delta_J_2
            disc_reward = tf.zeros(())
            opt_u_ha = [None] * horizon
            opt_x_ho = [None] * horizon

            for i in range(horizon):
                open_term = alpha * open_k_array.read(i)
                closed_term = tf.linalg.matvec(closed_K_array.read(i), x - x_ho[i])
                if self.verbose:
                    print_ops = [] #[tf.print('i', i, 'alpha', alpha, 'open', open_term, 'closed', closed_term, 'K', closed_K_array.read(i))]
                else:
                    print_ops = []
                with tf.control_dependencies(print_ops):
                    u = u_ha[i] + open_term + closed_term

                u = self._activate_u(u)
                opt_u_ha[i] = u
                opt_x_ho[i] = x

                x_prime = self.dynamics_model.predict_sym(x[None], u[None], pred_type=tf.random.uniform(shape=(), maxval=num_models, dtype=tf.int32))[0]
                reward = self._env.tf_reward(x, u, x_prime)
                disc_reward += self.discount**i * reward
                x = x_prime

            opt_J_val = -disc_reward
            opt_u_ha = tf.stack(opt_u_ha, axis=0)
            opt_x_ho = tf.stack(opt_x_ho, axis=0)

            next_alpha = alpha / self.alpha_decay_factor
            return (next_alpha, opt_J_val, delta_J_alpha, opt_u_ha, opt_x_ho)

        # break if backward pass is rejected or linear search condition is satisfied
        line_search_accept = lambda alpha, opt_J_val, delta_J_alpha, *args: tf.math.logical_and(
            tf.greater(0., delta_J_alpha), tf.greater(J_val-opt_J_val, -self.c_1*delta_J_alpha)
        )
        cond = lambda *args: tf.math.logical_not(tf.math.logical_or(
            backward_reject,
            line_search_accept(*args),
        ))

        # loop_vars: loop vars with opt_J_val:= J_val, prev_delta_J_alpha:= 0, opt_u_ha:= u_ha, opt_x_ho:= x_ho
        loop_vars = (self.alpha_init, J_val, 0., u_ha, x_ho)
        # terminal_loop_vars: loop vars right before convergence or maximum iteration is reached
        terminal_loop_vars = tf.while_loop(
            cond=cond, body=body, loop_vars=loop_vars, maximum_iterations=self.max_forward_iters
        )
        # if forward_stop is True, forward pass is accepted
        # otherwise set opt_J_val, opt_u_ha back to J_val, u_ha
        forward_accept = line_search_accept(*terminal_loop_vars)
        _, opt_J_val, _, opt_u_ha, opt_x_ho = tf.cond(
            pred=forward_accept,
            true_fn=lambda: terminal_loop_vars,
            false_fn=lambda: loop_vars,
        )

        if self.verbose:
            opt_J_val = tf.Print(opt_J_val, data=['accept', forward_accept, 'disc_reward', -opt_J_val, 'diff', J_val-opt_J_val])

        return opt_u_ha, opt_x_ho, -opt_J_val, J_val-opt_J_val, forward_accept

    def _activate_u(self, u):
        # u = tf.clip_by_value(u, self.act_low, self.act_high)
        scale = (self.act_high - self.act_low) * 0.5  # + 1e-8
        loc = (self.act_high + self.act_low) * 0.5
        return tf.tanh((u-loc)/scale) * scale + loc

    def _build_policy_loss(self):
        acts = self.policy.get_actions_sym(self.exp_obs_ph_no)
        loss = tf.losses.mean_squared_error(labels=self.exp_acts_ph_na, predictions=acts)
        # if self.verbose:
        with tf.control_dependencies([tf.print('loss before', loss)]):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            train_op = optimizer.minimize(loss, var_list=self.policy.get_params())

        return loss, train_op

    def optimize_policy(self, samples_data, log=True, prefix='', verbose=True):
        t = time.time()
        sess = tf.get_default_session()

        '''------------------- Sample initial observation from data buffer ---------------------'''

        observations = samples_data['observations']
        idx = np.random.choice(len(observations), self.policy_buffer_size)
        init_obs = observations[idx]
        u_nha, = sess.run([self.acts_var_nha], feed_dict={self.obs_ph_no: init_obs})

        '''--------------------- Collect experts trajectories ----------------------------------'''

        obs_buffer, acts_buffer = [], []
        for traj_idx in range(self.policy_buffer_size):
            accept = False
            x_o, u_ha = init_obs[traj_idx, :], u_nha[traj_idx, :, :]
            for itr in range(self.num_ilqr_iters):
                opt_accept, u_ha, x_ho, disc_reward, eff_diff = sess.run(
                    [self.opt_accept_var, self.opt_u_var_ha, self.opt_x_var_ho, self.opt_disc_reward_var, self.eff_diff_var],
                    feed_dict={self.u_ph_ha: u_ha, self.x_ph_o: x_o}
                )

                if not opt_accept:
                    pass
                    # break
                else:
                    accept = True

            if accept:
                obs_buffer.append(x_ho)
                acts_buffer.append(u_ha)

        '''--------------------- Policy gradient update to match expert trajectories ----------'''

        obs_buffer = np.concatenate(obs_buffer, axis=0)
        acts_buffer = np.concatenate(acts_buffer, axis=0)

        for _ in range(self.num_gradient_steps):

            np.random.shuffle(obs_buffer)
            np.random.shuffle(acts_buffer)

            i = 0
            while i < len(obs_buffer):
                acts = acts_buffer[i:i+self.batch_size]
                obs = obs_buffer[i:i+self.batch_size]
                _, = sess.run(
                    [self.train_op],
                    feed_dict={self.exp_acts_ph_na: acts, self.exp_obs_ph_no: obs},
                )
                i += self.batch_size

        if log or verbose:
            logger.logkv(prefix + 'OptPolicyTime', time.time() - t)
            logger.dumpkvs()
