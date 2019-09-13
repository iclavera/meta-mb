from meta_mb.utils.serializable import Serializable
import numpy as np
from pdb import set_trace as st


class MPCController(Serializable):
    def __init__(
            self,
            name,
            env,
            dynamics_model,
            reward_model=None,
            discount=1,
            use_cem=False,
            n_candidates=1024,
            horizon=10,
            num_cem_iters=8,
            Qs=None,
            percent_elites=0.1,
            use_reward_model=False,
            alpha=0.1,
            num_particles=20,
    ):
        Serializable.quick_init(self, locals())
        self.dynamics_model = dynamics_model
        self.reward_model = reward_model
        self.discount = discount
        self.n_candidates = n_candidates
        self.horizon = horizon
        self.use_cem = use_cem
        self.num_cem_iters = num_cem_iters
        self.percent_elites = percent_elites
        self.env = env
        self.use_reward_model = use_reward_model
        self.alpha = alpha
        self.num_particles = num_particles
        self.Qs = Qs

        self.unwrapped_env = env
        while hasattr(self.unwrapped_env, '_wrapped_env'):
            self.unwrapped_env = self.unwrapped_env._wrapped_env

        # make sure that enc has reward function
        assert hasattr(self.unwrapped_env, 'reward'), "env must have a reward function"

    @property
    def vectorized(self):
        return True

    def get_action(self, observation):
        if observation.ndim == 1:
            observation = observation[None]

        if self.use_cem:
            action = self.get_cem_action(observation)
        else:
            action = self.get_rs_action(observation)

        return action, dict()

    def get_actions(self, observations):
        if self.use_cem:
            actions = self.get_cem_action(observations)
        else:
            actions = self.get_rs_action(observations)

        #for i in range(3, len(actions)): #limit movement to first 3 joints
        #    actions[i] = 0
        return actions, dict()

    def get_random_action(self, n):
        return np.random.uniform(low=self.env.action_space.low,
                                 high=self.env.action_space.high, size=(n,) + self.env.action_space.low.shape)

    def get_sinusoid_actions(self, action_space, t):
        actions = np.array([])
        delta = t/action_space
        for i in range(action_space):
            #actions = np.append(actions, 0.5 * np.sin(i * delta)) #two different ways of sinusoidal sampling
            actions = np.append(actions, 0.5 * np.sin(i * t))
        #for i in range(3, len(actions)): #limit movement to first 3 joints
        #    actions[i] = 0
        return actions

    def get_cem_action(self, observations, mean=None, std=None):

        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        act_dim = self.env.action_space.shape[0]

        num_elites = max(int(self.n_candidates * self.percent_elites), 1)
        clip_low = np.array([self.env.action_space.low])
        clip_high = np.array([self.env.action_space.high])


        for i in range(self.num_cem_iters):
            z = np.random.normal(size=(n, m, act_dim))
            actions = []
            a_stack = []
            for t in range(h+1):
                a = mean[t] + z * std[t]
                a = np.clip(a, clip_low, clip_high)
                a_stack.append(a.copy())
                a = a.reshape((n * m, 1, act_dim))
                a = np.transpose(a, (1, 0, 2))
                actions.append(a)
            # h*(n*m*)*dim
            actions = np.array(actions)
            a_stacked = np.concatenate(a_stack, axis = -1)
            returns = np.zeros((n * m * self.num_particles,))

            cand_a = actions[0].reshape((m, n, -1))
            observation = np.repeat(observations, n * self.num_particles, axis=0)
            for t in range(h):
                a_t = np.repeat(actions[t], self.num_particles, axis=0)
                a_t = np.tanh(np.reshape(a_t, (-1, act_dim)))
                next_observation = self.dynamics_model.predict(observation, a_t, deterministic=False)
                rewards = self.unwrapped_env.reward(observation, a_t, next_observation)
                returns += self.discount ** t * rewards
                observation = next_observation
            returns = np.reshape(returns, (-1, 1))
            action = np.repeat(actions[h], self.num_particles, axis=0)
            action = np.tanh(np.reshape(action, (-1, act_dim)))
            input_q_fun = np.concatenate([observation, action], axis=-1)
            # TODO: add done function here
            next_q_values = [(self.discount ** (h + 1)) * Q.value(input_q_fun) for Q in self.Qs]
            q_values_var = [returns + next_q_values[j] for j in range(2)]
            min_q_val_var = np.min(q_values_var, axis=0)

            returns = np.mean(np.split(min_q_val_var.reshape(m, n * self.num_particles),
                                       self.num_particles, axis=-1), axis=0)   # TODO: Make sure this reshaping works
            elites_idx = ((-returns).argsort(axis=-1) < num_elites).T
            elites = a_stacked[elites_idx]
            elites = np.reshape(elites, (-1, h+1, act_dim))
            for t in range(h+1):
                mean[t] = mean[t] * self.alpha + (1 - self.alpha) * np.mean(elites, axis=0)[t]
                sd = np.std(elites, axis=0)[t]
                lb_dist, ub_dist = mean[t] - self.env.action_space.low, self.env.action_space.high - mean[t]
                std[t] = np.minimum(np.minimum(lb_dist/2, ub_dist/2), sd)

        return cand_a[range(m), np.argmax(returns, axis=1)]

    def get_rs_action(self, observations):
        n = self.n_candidates
        m = len(observations)
        h = self.horizon
        returns = np.zeros((n * m,))

        a = self.get_random_action(h * n * m).reshape((h, n * m, -1))

        cand_a = a[0].reshape((m, n, -1))
        observation = np.repeat(observations, n, axis=0)
        for t in range(h):
            next_observation = self.dynamics_model.predict(observation, a[t])
            if self.use_reward_model:
                assert self.reward_model is not None
                rewards = self.reward_model.predict(observation, a[t], next_observation)
            else:
                rewards = self.unwrapped_env.reward(observation, a[t], next_observation)
            returns += self.discount ** t * rewards
            observation = next_observation
        returns = returns.reshape(m, n)
        return cand_a[range(m), np.argmax(returns, axis=1)]

    def get_params_internal(self, **tags):
        return []

    def reset(self, dones=None):
        pass

    def log_diagnostics(*args):
        pass

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
