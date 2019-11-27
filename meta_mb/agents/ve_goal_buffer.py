from meta_mb.logger import logger
import numpy as np


class GoalBuffer(object):
    def __init__(
            self,
            env,
            max_buffer_size,
    ):
        self.max_buffer_size = max_buffer_size

        self.buffer = env.sample_goals(mode=None, num_samples=max_buffer_size)
        self.eval_buffer = env.eval_goals

        self.mc_goals = None
        self.goal_distribution = None

    def update_buffer(self, mc_goals, int_rewards, log=True):
        if mc_goals is not None:
            self.mc_goals = mc_goals
            self.goal_distribution = int_rewards

        indices = np.random.choice(len(self.mc_goals), size=self.max_buffer_size, p=self.goal_distribution)
        samples = self.mc_goals[indices]
        self.buffer = samples

        if log:
            logger.logkv('PMax', np.max(self.goal_distribution))
            logger.logkv('PMin', np.min(self.goal_distribution))
            logger.logkv('PStd', np.std(self.goal_distribution))
            logger.logkv('PMean', np.mean(self.goal_distribution))

    def get_batches(self, eval, batch_size):
        if eval:
            assert len(self.eval_buffer) % batch_size == 0, f"buffer size = {len(self.eval_buffer)}"
            num_batches = len(self.eval_buffer) // batch_size
            return np.split(np.asarray(self.eval_buffer), num_batches)

        assert self.max_buffer_size % batch_size == 0, f"buffer size = {self.max_buffer_size}"
        num_batches = self.max_buffer_size // batch_size
        return np.split(np.asarray(self.buffer), num_batches)
