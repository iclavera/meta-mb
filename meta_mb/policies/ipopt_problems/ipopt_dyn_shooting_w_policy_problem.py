import numpy as np
import tensorflow as tf
from meta_mb.policies import utils
import time
from meta_mb.logger import logger


class ShootingWPolicyProblem(object):
    def __init__(self, env, num_envs, dynamics_model, policy, horizon):
        self.env = env
        self.num_envs = num_envs
        self.dynamics_model = dynamics_model
        self.policy = policy
        self.horizon = horizon
        self.act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        self.init_obs_val = None

        self.init_obs_ph = tf.placeholder(
            dtype=tf.float32,
            shape=(self.num_envs, self.obs_dim),
        )
        self.sess = tf.get_default_session()
        t = time.time()
        self._build_graph()
        logger.log(f"building graph takes {time.time() - t}")

    def _build_graph(self):
        policy = self.policy
        init_obs_ph = tf.stop_gradient(self.init_obs_ph)

        # build objective
        returns = tf.zeros((self.num_envs,))
        raw_actions_array = []  # (list): horizon * (num_envs * act_dim)
        obs = init_obs_ph
        for i in range(self.horizon):
            acts, _ = policy.get_actions_sym(obs)
            raw_actions_array.append(tf.reshape(acts, (-1,)))
            acts = tf.clip_by_value(acts, self.act_low, self.act_high)
            next_obs = self.dynamics_model.predict_sym(obs_ph=obs, act_ph=acts)
            rewards = self.env.tf_reward(obs=obs, acts=acts, next_obs=next_obs)
            returns += rewards
            obs = next_obs
        self.tf_objective = tf.reduce_mean(-returns, axis=0)

        # build gradient
        policy_vars = list(policy.get_params().values())
        gradients_array = tf.gradients(ys=[self.tf_objective], xs=policy_vars)
        self.tf_gradient = tf.concat(
            list(map(lambda arr: tf.reshape(arr, (-1,)), gradients_array)), axis=0,
        )

        # build constraints
        self.tf_constraints = tf.concat(raw_actions_array, axis=0)

        # build jacobian
        jacobian_array = utils.jacobian_wrapper(y=self.tf_constraints, x=policy_vars,
                                                  dim_y=self.horizon*self.num_envs*self.act_dim)
        self.tf_jacobian = tf.concat(
            list(map(lambda arr: tf.reshape(arr, (-1,)), jacobian_array)), axis=0
        )
        print(f"==== shape = {self.tf_jacobian.get_shape()}")

    def set_init_obs(self, obs):
        self.init_obs_val = obs

    def objective(self, x):
        self.policy.set_raveled_params(x)
        return self.sess.run(self.tf_objective, feed_dict={self.init_obs_ph: self.init_obs_val})

    def gradient(self, x):
        self.policy.set_raveled_params(x)
        return self.sess.run(self.tf_gradient, feed_dict={self.init_obs_ph: self.init_obs_val})

    def constraints(self, x):
        self.policy.set_raveled_params(x)
        return self.sess.run(self.tf_constraints, feed_dict={self.init_obs_ph: self.init_obs_val})

    def jacobian(self, x):
        self.policy.set_raveled_params(x)
        return self.sess.run(self.tf_jacobian, feed_dict={self.init_obs_ph: self.init_obs_val})

