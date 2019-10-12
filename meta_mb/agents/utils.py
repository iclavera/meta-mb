import tensorflow as tf


class GoalSampler(object):
    def __init__(
            self,
            env,
            policy,
            q_ensemble,
            num_target_goals,
    ):
        self.env = env
        self.policy = policy
        self.q_ensemble = q_ensemble
        self.num_target_goals = num_target_goals

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.goal_dim = env.goal_dim

        self.target_goal_ph_no = tf.placeholder(dtype=tf.float32, shape=(num_target_goals, self.obs_dim), name='target_goal')
        self.min_q_var = self._build()

    def _build(self):
        ob_no = tf.stack([self.env.start_state for _ in range(self.num_target_goals)], axis=0)
        dist_info_sym = self.policy.distribution_info_sym(tf.concat([ob_no, self.target_goal_ph_no], axis=1))
        actions_var_na, _ = self.policy.distribution.sample_sym(dist_info_sym)
        input_q_fun = tf.concat([ob_no, actions_var_na, self.target_goal_ph_no], axis=1)

        # agent_q_var = (num_q, num_target_goals)
        min_q_var = tf.stack([tf.reshape(q.value_sym(input_var=input_q_fun), (-1,)) for q in self.q_ensemble], axis=0)
        return tf.reduce_min(min_q_var, axis=0)

    def compute_min_q(self, target_goals):
        feed_dict = {self.target_goal_ph_no: target_goals}
        sess = tf.get_default_session()
        min_q, = sess.run([self.min_q_var], feed_dict=feed_dict)
        return min_q
