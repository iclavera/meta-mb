import numpy as np


class ValueEnsembleWrapper(object):
    def __init__(self, env, size, vfun_list, num_mc_goals):
        self.env = env
        self.size = size
        self.vfun_list = vfun_list
        self.num_mc_goals = num_mc_goals

    def sample_goals(self, init_obs_no):
        if self.size == 0:  # baseline
            return self.env.sample_goals(mode=None, num_samples=len(init_obs_no))

        mc_goals = self.env.sample_goals(mode=None, num_samples=self.num_mc_goals)
        input_obs = np.repeat(init_obs_no, repeats=self.num_mc_goals).reshape((-1, self.env.obs_dim))
        input_goal = np.tile(mc_goals, [len(init_obs_no), 1])
        values = []
        for vfun in self.vfun_list:
            values.append(vfun.compute_values(input_obs, input_goal).flatten())

        goal_distribution = np.var(values, axis=0)
        goal_distribution /= np.sum(goal_distribution)
        indices = np.random.choice(self.num_mc_goals, size=len(init_obs_no), replace=True, p=goal_distribution)
        samples = mc_goals[indices]
        return samples