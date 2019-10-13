import tensorflow as tf
from meta_mb.logger import logger
import numpy as np


class GoalBuffer(object):
    def __init__(
            self,
            env,
            agent_index,
            policy,
            q_ensemble,
            eval_goals,
            num_target_goals,
            max_buffer_size,
            eps
    ):
        self.env = env
        self.agent_index = agent_index
        self.policy = policy
        self.q_ensemble = q_ensemble
        self.num_target_goals = num_target_goals
        self.max_buffer_size = max_buffer_size
        self.eps = eps

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.goal_dim = env.goal_dim

        self.target_goal_ph_no = tf.placeholder(dtype=tf.float32, shape=(num_target_goals, self.obs_dim), name='target_goal')
        self.min_q_var = self._build()

        self.buffer = env.sample_goals(max_buffer_size)
        self.eval_buffer = eval_goals

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

    def refresh(self, target_goals, agent_q, max_q, q_list, log=True):
        """
        g ~ (1 - eps) * P + eps * U
        :param alpha:
        :param target_goals:
        :param agent_q:
        :param max_q:
        :param q_list:
        :param num_samples:
        :return:
        """
        samples = []
        samples.extend(target_goals[np.where(agent_q == max_q)])
        logger.logkv('q_leading-pct', len(samples) / len(agent_q))

        p = np.exp(np.max(q_list[:self.agent_index] + q_list[self.agent_index+1:], axis=0) - agent_q)
        p /= np.sum(p)
        p = np.ones(len(target_goals)) / len(target_goals)
        p /= np.sum(p)  # fix numeric error
        indices = np.random.choice(len(target_goals), size=self.max_buffer_size - len(samples), replace=True, p=p)
        samples.extend(target_goals[indices])

        self.buffer = samples

    def get_batch_generator(self, eval, batch_size):
        if eval:
            assert len(self.eval_buffer) % batch_size == 0, f"buffer size = {len(self.eval_buffer)}"
            num_batches = len(self.eval_buffer) // batch_size
            return iter(np.split(np.asarray(self.eval_buffer), num_batches))

        assert self.max_buffer_size % batch_size == 0, f"buffer size = {self.max_buffer_size}"
        num_batches = self.max_buffer_size // batch_size
        return iter(np.split(np.asarray(self.buffer), num_batches))
