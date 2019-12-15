from meta_mb.logger import logger
import numpy as np


class GoalSampler(object):
    def __init__(
            self,
            env,
            greedy_eps,
    ):
        self.env = env
        self.greedy_eps = greedy_eps

        self.goal_dim = env.goal_dim

        # temporary storage
        self.mc_goals = None
        self.goal_dist = None

    def set_goal_dist(self, mc_goals, goal_dist):
        self.mc_goals = mc_goals
        self.goal_dist = goal_dist

    def sample_goals(self, init_obs_no, log=True, log_prefix=''):
        if self.goal_dist is None:
            # assert self.greedy_eps == 1
            indices = np.random.choice(len(self.mc_goals), size=len(init_obs_no))
            return self.mc_goals[indices]

        # assume no goal buffer for now
        samples = np.empty((len(init_obs_no), self.goal_dim))
        greedy_mask = np.random.binomial(1, self.greedy_eps, len(init_obs_no)).astype(np.bool)
        size_u_samples = np.sum(greedy_mask)
        size_p_samples = len(init_obs_no) - size_u_samples

        p_indices = np.random.choice(len(self.mc_goals), size=size_p_samples, p=self.goal_dist)  # FIXME: replace = False?
        samples[np.logical_not(greedy_mask)] = self.mc_goals[p_indices]

        u_indices = np.random.choice(len(self.mc_goals), size=size_u_samples)
        samples[greedy_mask] = self.mc_goals[u_indices]

        return samples
