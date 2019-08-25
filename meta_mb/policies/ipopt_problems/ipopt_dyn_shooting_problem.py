import numpy as np
import tensorflow as tf


class ShootingProblem(object):
    def __init__(self, env, dynamics_model, horizon):
        self.env = env
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.init_obs_val = None

        self.init_obs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(self.obs_dim,),
        )
        self.inputs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=horizon*self.act_dim
        )
        self.sess = tf.get_default_session()
        self._build_graph()

    def _build_graph(self):
        obs_dim, act_dim = self.obs_dim, self.act_dim
        horizon = self.horizon
        # init_obs_ph, inputs_ph = tf.stop_gradient(self.init_obs_ph), tf.stop_gradient(self.inputs_ph)
        init_obs_ph, inputs_ph = self.init_obs_ph, self.inputs_ph

        u_array = tf.reshape(inputs_ph, shape=(horizon, act_dim))

        # build objective
        returns = tf.zeros(())
        obs = init_obs_ph
        for i in range(self.horizon):
            act = u_array[i]
            next_obs = self.dynamics_model.predict_sym(obs_ph=obs[None], act_ph=act[None])[0]
            reward = self.env.tf_reward(obs=obs, acts=act, next_obs=next_obs)
            returns += reward
            obs = next_obs
        self.tf_objective = -returns

        # build gradient
        self.tf_gradient, = tf.gradients(ys=[self.tf_objective], xs=[inputs_ph])

        # # build gradient
        # dr_dx = self.env.tf_deriv_reward_obs(obs=x_array, acts=u_array, batch_size=horizon)
        # dr_du = self.env.tf_deriv_reward_act(obs=x_array, acts=u_array, batch_size=horizon)
        # self.tf_gradient = tf.concat([tf.reshape(-dr_dx[1:], (-1,)), tf.reshape(-dr_du, (-1,))], axis=0)

    def set_init_obs(self, obs):
        self.init_obs_val = obs

    def get_u(self, inputs):
        return inputs.reshape(self.horizon, self.act_dim)

    def get_inputs(self, u):
        return u.ravel()  # WARNING: not copied

    def objective(self, x):
        return self.sess.run(self.tf_objective, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})

    def gradient(self, x):
        return self.sess.run(self.tf_gradient, feed_dict={self.inputs_ph: x, self.init_obs_ph: self.init_obs_val})
