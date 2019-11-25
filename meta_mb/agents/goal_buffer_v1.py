import tensorflow as tf
from meta_mb.logger import logger
from meta_mb.utils import compile_function
import numpy as np


class GoalBufferV1(object):
    def __init__(
            self,
            env,
            agent_index,
            policy,
            q_ensemble,
            max_buffer_size,
            alpha,
            sampling_rule,
            sess,
            # curiosity_percentage,
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
        self.compute_min_q = lambda goal_ng: sess.run(self.min_q_var, feed_dict={self.goal_ph: goal_ng})

        self.buffer = env.sample_goals(mode=None, num_samples=max_buffer_size)
        self.eval_buffer = env.eval_goals

        self.sampling_rule = sampling_rule
        # self.curiosity_percentage = curiosity_percentage

        # temporary storage
        self.mc_goals = None
        self.proposed_goals = []
        self.q_max = None

    def _build(self):
        ob_no = tf.tile(self.env.init_obs[None], (tf.shape(self.goal_ph)[0], 1))
        dist_info_sym = self.policy.distribution_info_sym(tf.concat([ob_no, self.goal_ph], axis=1))
        act_na, _ = self.policy.distribution.sample_sym(dist_info_sym)
        input_q_fun = tf.concat([ob_no, act_na, self.goal_ph], axis=1)

        # agent_q_var = (num_q, num_target_goals)
        q_vals = tf.stack([tf.reshape(q.value_sym(input_var=input_q_fun), (-1,)) for q in self.q_ensemble], axis=0)
        return tf.reduce_min(q_vals, axis=0)

    def update_info(self, mc_goals, q_list):
        self.mc_goals = mc_goals
        self.q_max = np.max(q_list, axis=0)

    def update_buffer(self, proposed_goals, log=True):
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

        if self.alpha == 1 or self.alpha == -1:
            # uniform sampling, all sample_goals come from env.target_goals
            self.buffer = self.mc_goals[np.random.choice(len(self.mc_goals), size=self.max_buffer_size, replace=True)]
            return None

        """--------------- alpha < 1, g ~ (1-alpha) * P + alpha * U ------------------"""

        agent_q = self.compute_min_q(self.mc_goals)
        self.q_max = np.maximum(self.q_max, agent_q)

        num_proposed_goals = len(proposed_goals)
        num_goals_u = int((self.max_buffer_size - num_proposed_goals) * self.alpha)
        num_goals_p = self.max_buffer_size - num_proposed_goals - num_goals_u


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

        """------------------------ sample with P --------------------"""

        if self.sampling_rule == 'softmax':

            """-------------- sample with softmax -------------"""

            log_diff = self.q_max - agent_q
            p = np.exp(log_diff - np.max(log_diff))
            p /= np.sum(p)

        elif self.sampling_rule == 'norm_diff':

            """------------- sample with normalized difference --------------"""

            diff = self.q_max - agent_q
            if np.sum(diff) == 0:
                p = np.ones_like(diff) / len(diff)
            else:
                p = diff / np.sum(diff)

        else:
            raise ValueError

        indices_p = np.random.choice(len(self.mc_goals), size=num_goals_p, replace=True, p=p)

        """------------------------- sample with U -----------------"""

        u = np.ones_like(agent_q)
        u = u / np.sum(u)
        indices_u = np.random.choice(len(self.mc_goals), size=num_goals_u, replace=True, p=u)

        """----------------------- concatenate sampled goals ---------------------"""

        samples = proposed_goals + list(self.mc_goals[indices_p]) + list(self.mc_goals[indices_u])
        assert len(samples) == self.max_buffer_size
        self.buffer = samples

        if log:
            logger.logkv('PMax', np.max(p))
            logger.logkv('PMin', np.min(p))
            logger.logkv('PStd', np.std(p))
            logger.logkv('PMean', np.mean(p))
            logger.logkv('ProposedGoalsCtr', len(proposed_goals))

        return indices_p

    def get_batches(self, eval, batch_size):
        if eval:
            assert len(self.eval_buffer) % batch_size == 0, f"buffer size = {len(self.eval_buffer)}"
            num_batches = len(self.eval_buffer) // batch_size
            return np.split(np.asarray(self.eval_buffer), num_batches)

        assert self.max_buffer_size % batch_size == 0, f"buffer size = {self.max_buffer_size}"
        num_batches = self.max_buffer_size // batch_size
        return np.split(np.asarray(self.buffer), num_batches)
