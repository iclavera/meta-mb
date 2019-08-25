import numpy as np
import time
from meta_mb.logger import logger
import tensorflow as tf
from meta_mb.policies import utils


class CollocationProblem(object):
    def __init__(self, env, dynamics_model, horizon):
        self.env = env
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_dim = int(np.prod(env.action_space.shape))
        self.init_obs_val = None

        self.init_obs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(self.obs_dim,),
        )
        self.inputs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=((self.horizon-1)*self.obs_dim + self.horizon*self.act_dim, )
        )
        self.sess = tf.get_default_session()
        self._build_graph()

    # def _df_sym(self, obs, acts, next_obs):
    #     """
    #
    #     :param obs: (horizon, obs_dim)
    #     :param acts: (horizon, act_dim)
    #     :param next_obs: (horizon, obs_dim)
    #     """
    #     # jac_f_x: (horizon, obs_dim, obs_dim)
    #     jac_f_x = utils.gradients_wrapper(next_obs, obs, self.obs_dim, self.obs_dim, [obs, acts])
    #
    #     # jac_f_u: (horizon, obs_dim, act_dim)
    #     jac_f_u = utils.gradients_wrapper(next_obs, acts, self.obs_dim, self.act_dim, [obs, acts])
    #
    #     return jac_f_x, jac_f_u

    def _build_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        # init_obs_ph, inputs_ph = self.init_obs_ph, self.inputs_ph
        init_obs_ph, inputs_ph = tf.stop_gradient(self.init_obs_ph), tf.stop_gradient(self.inputs_ph)

        x_array_drop_first = tf.reshape(
            inputs_ph[:(horizon-1)*obs_dim],
            shape=(horizon-1, obs_dim,),
        )
        x_array = tf.concat([init_obs_ph[None], x_array_drop_first], axis=0)
        u_array = tf.reshape(
            inputs_ph[-horizon*act_dim:],
            shape=(horizon, act_dim),
        )

        # build objective
        self.tf_objective = -tf.reduce_sum(self.env.tf_reward(obs=x_array, acts=u_array, next_obs=None), axis=0)

        # build constraints, (horizon-1, obs_dim) => ((horizon-1) * obs_dim,)
        x_target_array = self.dynamics_model.predict_sym(obs_ph=x_array, act_ph=u_array)
        self.tf_constraints = tf.reshape(x_array_drop_first - x_target_array[:-1], (-1,))

        # self.debugger_array = [x_target_array, x_array_drop_first]

        # # build gradient for objective
        # dr_dx = self.env.tf_deriv_reward_obs(obs=x_array, acts=u_array, batch_size=horizon)
        # dr_du = self.env.tf_deriv_reward_act(obs=x_array, acts=u_array, batch_size=horizon)
        # self.tf_gradient = tf.concat([tf.reshape(-dr_dx[1:], (-1,)), tf.reshape(-dr_du, (-1,))], axis=0)
        #
        # # build jacobian matrix for constraints
        # jac_f_x, jac_f_u = self._df_sym(obs=x_array, acts=u_array, next_obs=x_target_array)
        # jac_c_x = tf.eye(num_rows=(horizon-1)*obs_dim, num_columns=(horizon-1)*obs_dim)
        # for i in range(1, horizon-1):
        #     jac_c_x += tf.pad(
        #         -jac_f_x[i],
        #         paddings=tf.constant([[i*obs_dim, (horizon-i-2)*obs_dim], [(i-1)*obs_dim, (horizon-i-1)*obs_dim]]),
        #     )
        # jac_c_u = tf.zeros(((horizon-1)*obs_dim, horizon*act_dim,))
        # for i in range(horizon-1):
        #     jac_c_u += tf.pad(
        #         -jac_f_u[i],
        #         paddings=tf.constant([[i*obs_dim, (horizon-i-2)*obs_dim], [i*act_dim, (horizon-i-1)*act_dim]]),
        #     )
        # jac_c_inputs = tf.concat([jac_c_x, jac_c_u], axis=1)
        # assert jac_c_inputs.get_shape().as_list() == [(horizon-1)*obs_dim, (horizon-1)*obs_dim+horizon*act_dim]
        # self.tf_jacobian = jac_c_inputs

        # build gradient for objective
        t = time.time()
        self.tf_gradient, = tf.gradients(ys=[self.tf_objective], xs=[inputs_ph])
        logger.log(f'compute tf_gradient takes {time.time() - t}')

        # build jacobian matrix for constraints
        t = time.time()
        self.tf_jacobian = utils.jacobian_wrapper(y=self.tf_constraints, x=inputs_ph, dim_y=(horizon - 1) * obs_dim)
        logger.log(f'compute tf_jacobian takes {time.time() - t}')

    def set_init_obs(self, obs):
        self.init_obs_val = obs

    def get_inputs(self, x, u):
        assert x.shape == (self.horizon-1, self.obs_dim)
        assert u.shape == (self.horizon, self.act_dim)
        return np.concatenate([x.ravel(), u.ravel()])

    def get_x_u(self, inputs):
        x = inputs[:(self.horizon-1)*self.obs_dim].reshape(self.horizon-1, self.obs_dim)
        u = inputs[-self.horizon*self.act_dim:].reshape(self.horizon, self.act_dim)
        return x, u

    def objective(self, x):
        """
        The callback for calculating the objective
        :param x: concat(x_2, ..., x_T, u_1, u_2..., u_T)
        :return:
        """
        # x_target_array, x_array_drop_first = self.sess.run(self.debugger_array, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})
        # for i in range(self.horizon-1):
        #     logger.log(f"at {i}, target_obs = {x_target_array[i]}, obs = {x_array_drop_first[i]}")
        return self.sess.run(self.tf_objective, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})

    def constraints(self, x):
        return self.sess.run(self.tf_constraints, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})

    def gradient(self, x):
        return self.sess.run(self.tf_gradient, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})

    def jacobian(self, x):
        return self.sess.run(self.tf_jacobian, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})
