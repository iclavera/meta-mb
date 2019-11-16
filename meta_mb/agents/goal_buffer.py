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
            max_buffer_size,
            alpha,
            sample_rule,
            curiosity_percentage,
    ):
        self.env = env
        self.agent_index = agent_index
        self.policy = policy
        self.q_ensemble = q_ensemble
        self.max_buffer_size = max_buffer_size
        self.alpha = alpha

        self.obs_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.goal_dim = env.goal_dim

        self.goal_ph = tf.placeholder(dtype=tf.float32, shape=(None, self.goal_dim), name='goal')
        self.min_q_var = self._build()

        self.buffer = env.sample_goals(mode=None, num_samples=max_buffer_size)
        self.eval_buffer = env.eval_goals

        self.sample_rule = sample_rule
        self.curiosity_percentage = curiosity_percentage

    def _build(self):
        ob_no = tf.tile(self.env.init_obs[None], (tf.shape(self.goal_ph)[0], 1))
        dist_info_sym = self.policy.distribution_info_sym(tf.concat([ob_no, self.goal_ph], axis=1))
        act_na, _ = self.policy.distribution.sample_sym(dist_info_sym)
        input_q_fun = tf.concat([ob_no, act_na, self.goal_ph], axis=1)

        # agent_q_var = (num_q, num_target_goals)
        q_vals = tf.stack([tf.reshape(q.value_sym(input_var=input_q_fun), (-1,)) for q in self.q_ensemble], axis=0)
        return tf.reduce_min(q_vals, axis=0)

    def compute_min_q(self, target_goals):
        feed_dict = {self.goal_ph: target_goals}
        sess = tf.get_default_session()
        min_q, = sess.run([self.min_q_var], feed_dict=feed_dict)
        return min_q

    def refresh(self, mc_goals, proposed_goals, q_list, log=True):
        """
        g ~ (1 - alpha) * P + alpha * U
        U = X_E, where E is the target region in the maze, X is the indicator function
        if alpha = 1, g ~ U, target_goals should all lie in E
        otherwise target_goals lie in anywhere of the maze
        :param mc_goals:
        :param reused_goals: goals from previous refresh iteration, to be appended to the goal buffer
        :param q_list:
        :param log:
        :return:
        """

        """--------------------- alpha = 1, g ~ U or g ~ X ---------------------"""

        if self.alpha == 1:
            # uniform sampling, all sample_goals come from env.target_goals
            self.buffer = mc_goals[np.random.choice(len(mc_goals), size=self.max_buffer_size, replace=True)]
            return

        assert mc_goals.shape == (q_list.shape[1], self.goal_dim)

        """--------------- alpha < 1, g ~ (1-alpha) * P + alpha * U ------------------"""

        # for maze env
        # _target_goals_ind_list = self.env._target_goals_ind.tolist()
        # mask = np.array(list(map(lambda ind: ind.tolist() in _target_goals_ind_list, self.env._get_index(sample_goals))), dtype=np.int)
        # assert np.sum(mask) > 0
        # if np.sum(mask) > 0:
        #     u = mask / np.sum(mask)
        # else:
        #     u = np.zeros_like(mask)

        samples = []

        """--------------------- sample with curiosity -------------------"""
        # if the current agent has the max q value, add the goal to the buffer,
        # because it might be an overestimate due to distribution mismatch
        # agent_q = q_list[self.agent_index, :]
        # max_q, min_q = np.max(q_list, axis=0), np.min(q_list, axis=0)
        # kth = int(len(agent_q) * (1 - self.curiosity_percentage))  # drop k goals with low disagreement
        # curiosity_mask = np.full_like(agent_q, fill_value=True)
        # curiosity_mask[np.argpartition(max_q - min_q, kth=kth)[:kth]] = False
        # samples.extend(mc_goals[np.logical_and(agent_q == max_q, curiosity_mask)])

        """------------ sample if the current agent proposed a goal in the previous iteration  -------------"""
        samples.extend(proposed_goals)

        if log:
            logger.logkv('q_leading-pct', len(samples) / len(q_list[0]))

        if self.sample_rule == 'softmax':

            """-------------- sample with softmax -------------"""

            log_p = np.max(q_list[:self.agent_index] + q_list[self.agent_index+1:], axis=0)  # - agent_q
            p = np.exp(log_p - np.max(log_p))
            p /= np.sum(p)

        elif self.sample_rule == 'norm_diff':

            """------------- sample with normalized difference --------------"""

            max_q = np.max(q_list, axis=0)
            agent_q = q_list[self.agent_index, :]
            p = max_q - agent_q
            if np.sum(p) == 0:
                p = np.ones_like(p) / len(p)
            else:
                p = p / np.sum(p)

        else:
            raise ValueError

        u = np.ones_like(q_list[0])
        u = u / np.sum(u)
        goal_dist = (1 - self.alpha) * p + self.alpha * u
        goal_dist = goal_dist / np.sum(goal_dist)
        indices = np.random.choice(len(mc_goals), size=self.max_buffer_size - len(samples), replace=True, p=goal_dist)
        samples.extend(mc_goals[indices])

        self.buffer = samples

        return indices

    def get_batches(self, eval, batch_size):
        if eval:
            assert len(self.eval_buffer) % batch_size == 0, f"buffer size = {len(self.eval_buffer)}"
            num_batches = len(self.eval_buffer) // batch_size
            return np.split(np.asarray(self.eval_buffer), num_batches)

        assert self.max_buffer_size % batch_size == 0, f"buffer size = {self.max_buffer_size}"
        num_batches = self.max_buffer_size // batch_size
        return np.split(np.asarray(self.buffer), num_batches)
