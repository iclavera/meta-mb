import tensorflow as tf


class GoalSampler(object):
    def __init__(
            self,
            env,
            policy,
            q_ensemble,
            num_target_goals,
            num_goals,
    ):
        self.env = env
        self.policy = policy
        self.q_ensemble = q_ensemble
        self.num_target_goals = num_target_goals
        self.num_goals = num_goals

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.goal_dim = env.goal_dim

        self.target_goal_ph_no = tf.placeholder(dtype=tf.float32, shape=(num_target_goals, self.obs_dim), name='target_goal')
        self.observations_ph_no = tf.placeholder(dtype=tf.float32, shape=(num_target_goals, self.obs_dim), name='ob')
        self.log_q_var = self._build()

    def _build(self):
        dist_info_sym = self.policy.distribution_info_sym(tf.concat([self.observations_ph_no, self.target_goal_ph_no], axis=-1))
        actions_var_na, _ = self.policy.distribution.sample_sym(dist_info_sym)
        input_q_fun = tf.concat([self.observations_ph_no, actions_var_na, self.target_goal_ph_no], axis=-1)

        # agent_q_var = (num_q, num_target_goals)
        q_var = tf.stack([q.value_sym(input_var=input_q_fun) for q in self.q_ensemble], axis=0)
        log_q_var = tf.log(tf.reduce_min(q_var, axis=0))
        return log_q_var
        # opponent_q_var = (num_agents-1, num_q, num_target_goals)
        # opponent_q_var = [tf.stack([q.value_sym(input_var=input_q_fun) for q in ensemble], axis=0) for ensemble in self.opponent_q_function_ensembles]
        # opponent_q_var = tf.stack(opponent_q_var, axis=0)

        # opponent_log_q_var = tf.reduce_min(tf.log(opponent_q_var), axis=-2)
        # log_q_var = (num_agents, num_target_goals)
        # log_q_var = tf.concat([tf.expand_dims(agent_log_q_var, axis=0), opponent_log_q_var], axis=0)
        # take the max across agents
        # log_q_var = tf.reduce_max(log_q_var, axis=0)
        #
        # goal_dist = tf.distributions.Categorical(logits=tf.expand_dims(log_q_var-agent_log_q_var, axis=0), name='goal_dist')
        # goal_var = tf.concat([goal_dist.sample() for _ in range(self.num_goals)], axis=0)
        # return goal_var

    def compute_log_q(self, target_goals, init_obs_no):
        feed_dict = {self.target_goal_ph_no: target_goals, self.observations_ph_no: init_obs_no}
        sess = tf.get_default_session()
        log_q = sess.run(self.log_q_var, feed_dict=feed_dict)
        return log_q
