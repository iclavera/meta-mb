from meta_mb.samplers.base import SampleProcessorfrom meta_mb.utils import utilsfrom meta_mb.logger import loggerimport numpy as npclass ModelSampleProcessor(SampleProcessor):    def __init__(            self,            reward_fn,            achieved_goal_fn,            baseline=None,            discount=0.99,            gae_lambda=1,            normalize_adv=False,            positive_adv=False,    ):        self.reward_fn = reward_fn        self.achieved_goal_fn = achieved_goal_fn        self.baseline = baseline        self.discount = discount        self.gae_lambda = gae_lambda        self.normalize_adv = normalize_adv        self.positive_adv = positive_adv    def process_samples(self, paths, replay_strategy=None, replay_k=4, log=False, log_prefix='', log_all=False):        """        Processes sampled paths. This involves:            - computing discounted rewards (returns)            - fitting baseline estimator using the path returns and predicting the return baselines            - estimating the advantages using GAE (+ advantage normalization id desired)            - stacking the path data            - logging statistics of the paths        Args:            paths_meta_batch (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)            log (str): indicates whether to log            log_prefix (str): prefix for the logging keys        Returns:            (list of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)        """        if replay_strategy == 'future':            # future_p = 1 - (1. / (1 + replay_k))            pass        else:            replay_k = None        samples_data, paths = self._compute_samples_data(paths, replay_k)        # 8) log statistics if desired        self._log_path_stats(paths, log=log, log_prefix=log_prefix, log_all=log_all)        return samples_data    def _compute_samples_data(self, paths, replay_k):        assert type(paths) == list        # 1) compute discounted rewards (returns)        for path in paths:            path['returns'] = utils.discount_cumsum(path["rewards"], self.discount)        # 4) stack path data        if replay_k is not None:            goals, observations, next_observations, actions, rewards, dones, returns, time_steps, env_infos, agent_infos = \                self._concatenate_path_data_with_hindsight(paths, replay_k)        else:            goals, observations, next_observations, actions, rewards, dones, returns, time_steps, env_infos, agent_infos = \                self._concatenate_path_data(paths)        # 6) create samples_data object        samples_data = dict(            goals=goals,            observations=observations,            next_observations=next_observations,            actions=actions,            rewards=rewards,            dones=dones,            returns=returns,            advantages=returns, # FIXME: Hack for SVG            time_steps=time_steps,            env_infos=env_infos,            agent_infos=agent_infos,        )        return samples_data, paths    def _concatenate_path_data_with_hindsight(self, paths, replay_k):        # stack paths        goals = np.stack([path["goals"][:-1] for path in paths])        observations = np.stack([path["observations"][:-1] for path in paths])        next_observations = np.stack([path["observations"][1:] for path in paths])        actions = np.stack([path["actions"][:-1] for path in paths])        rewards = np.stack([path["rewards"][:-1] for path in paths])        dones = np.stack([path["dones"][:-1] for path in paths])        returns = np.stack([path["returns"][:-1] for path in paths])        time_steps = np.stack([np.arange(len(path["observations"][:-1])) for path in paths])        '''------------------------------ HER -------------------------'''        num_paths, horizon, _ = np.shape(goals)        # index of path: (num_paths * horizon)        idx_0 = np.repeat(np.arange(num_paths), repeats=horizon)        # index of timestep: (horizon * num_paths)        idx_1 = np.tile(np.arange(horizon), reps=num_paths)        # sample timesteps with probability replay_k / horizon, shape = (horizon * num_paths, horizon)        base_mask = np.random.random(size=(len(idx_1), horizon)) < 2*replay_k/horizon        # use 2*replay/horizon above because only sample goals        # achieved after OR THE SAME AS the current timestep in the same trajectory        future_mask = idx_1[..., np.newaxis] <= np.arange(horizon)[np.newaxis, ...]        # all index arrays have shape (num_paths * horizon * replay_k,) in expectation        future_idx_0, future_idx_1 = np.where(np.logical_and(base_mask, future_mask))        idx_0, idx_1 = idx_0[future_idx_0], idx_1[future_idx_0]        # _goals is set to be achieved goal from future or current timestep        _goals = self.achieved_goal_fn(next_observations[idx_0, future_idx_1])        assert _goals.shape == (len(idx_0), goals.shape[-1]), _goals.shape        _observations = observations[idx_0, idx_1]        _next_observations = next_observations[idx_0, idx_1]        _actions = actions[idx_0, idx_1]        _rewards = self.reward_fn(_observations, _actions, _next_observations, _goals)        logger.logkv('AvgReward', np.mean(rewards))        logger.logkv('HindsightAvgReward', np.mean(_rewards))        '''------------- concatenate original samples and relabeled samples -----------------'''        new_goals = np.concatenate([*goals, _goals])        new_observations = np.concatenate([*observations, _observations])        new_next_observations = np.concatenate([*next_observations, _next_observations])        new_actions = np.concatenate([*actions, _actions])        new_rewards = np.concatenate([*rewards, _rewards])        new_dones = np.concatenate([*dones, dones[idx_0, idx_1]])        new_returns = np.concatenate([*returns, returns[idx_0, idx_1]])        new_time_steps = np.concatenate([*time_steps, time_steps[idx_0, idx_1]])        new_env_infos = None        new_agent_infos = None        return new_goals, new_observations, new_next_observations, new_actions, new_rewards, new_dones, new_returns, new_time_steps, new_env_infos, new_agent_infos    def _concatenate_path_data(self, paths):        goals = np.concatenate([path["goals"][:-1] for path in paths])        observations = np.concatenate([path["observations"][:-1] for path in paths])        next_observations = np.concatenate([path["observations"][1:] for path in paths])        actions = np.concatenate([path["actions"][:-1] for path in paths])        rewards = np.concatenate([path["rewards"][:-1] for path in paths])        dones = np.concatenate([path["dones"][:-1] for path in paths])        returns = np.concatenate([path["returns"][:-1] for path in paths])        time_steps = np.concatenate([np.arange(len(path["observations"][:-1])) for path in paths])        env_infos = utils.concat_tensor_dict_list([path["env_infos"] for path in paths], end=-1)        agent_infos = utils.concat_tensor_dict_list([path["agent_infos"]for path in paths], end=-1)        return goals, observations, next_observations, actions, rewards, dones, returns, time_steps, env_infos, agent_infos