import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time
from gym import spaces, Env
import tensorflow as tf
from pdb import set_trace as st

class DoubleIntegratorEnv(Env):
    """
    state: [pos, vel]
    """
    def __init__(self, discount=0.99):
        self._state = np.zeros((2,))
        self.dt = 0.05
        self.max_path_length = 200
        self._fig = None
        self.discount = discount
        self.vectorized = True
        self.action_space = spaces.Box(low=np.array((-1,)), high=np.array((1,)), dtype=np.float64)
        self.observation_space = spaces.Box(low=np.array((-1, -1)), high=np.array((1e6, 1e6)), dtype=np.float64)

    def step(self, action):
        next_state = self._state + np.array([self._state[1], action[0]]) * self.dt
        reward = -0.05 * (self._state[0] ** 2 + self._state[1] ** 2 + action ** 2)
        next_state = np.clip(next_state, self.observation_space.low, self.observation_space.high)
        done = False
        # done = False
        env_info = dict()
        self._state = next_state

        return next_state.copy(), reward, done, env_info

    def reset(self):
        self._states = None
        # self._state = np.random.uniform(low=-2, high=2, size=2)
        self._state = np.ones((2,))
        return self._state.copy()

    def reward(self, observation, action, next_obs = None):
        return -0.5 * ((observation[:, 0] ** 2 + observation[:, 1] ** 2)[:, None] + action ** 2)

    def tf_reward(self, observation, action, next_obs = None):
        return tf.reshape(-0.5 * (tf.expand_dims(observation[:, 0] ** 2 + observation[:, 1] ** 2, axis = -1) + action ** 2), [-1])

    def tf_termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = tf.tile(tf.constant([False]), [tf.shape(obs)[0]])
        # done = tf.logical_or(tf.reduce_any(next_obs < self.observation_space.low, axis = -1),
        #                      tf.reduce_any(next_obs > self.observation_space.high, axis = -1))
        done = done[:,None]
        return done

    def termination_fn(self, obs, act, next_obs):
        assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 2
        done = np.array([False]).repeat(obs.shape[0])
        # done = np.any(next_obs < self.observation_space.low, axis = -1) or np.any(next_obs > self.observation_space.high, axis = -1)
        done = done[:,None]
        return done



    def set_state(self, state):
        self._state = state

    def vec_step(self, actions):
        next_states = self._states + np.stack([self._states[:, 1], actions[:, 0]], axis=-1) * self.dt
        rewards = -0.5 * (self._states[:, 0] ** 2 + self._states[:, 1] ** 2 + actions[:, 0] ** 2)
        dones = np.sum([(next_states[:, i] < l) + (next_states[:, i] > h) for i, (l, h)
                        in enumerate(zip(self.observation_space.low, self.observation_space.high))], axis=0).astype(np.bool)
        env_infos = dict()
        self._states = next_states
        rewards[dones] /= (1 - self.discount)
        return next_states, rewards, dones, env_infos

    def vec_set_state(self, states):
        self._states = states

    def vec_reset(self, num_envs=None):
        if num_envs is None:
            assert self._num_envs is not None
            num_envs = self._num_envs
        else:
            self._num_envs = num_envs
        self._states = np.random.uniform(low=-2, high=2, size=(num_envs, 2))
        return self._states

    def render(self, mode='human', iteration=None):
        if self._fig is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(111)
            self._agent_render, = self._ax.plot(self._state[0], 0, 'ro')
            self._goal_render, = self._ax.plot(0, 'y*')
            self._ax.set_xlim(-4.5, 4.5)
            self._ax.set_ylim(-.5, .5)
            self._ax.set_aspect('equal')
            self._canvas = FigureCanvas(self._fig)

        self._agent_render.set_data(self._state[0], 0)
        if iteration is not None:
            self._ax.set_title('Iteration %d' % iteration)
        self._canvas.draw()
        self._canvas.flush_events()
        if mode == 'rgb_array':
            width, height = self._fig.get_size_inches() * self._fig.get_dpi()
            image = np.fromstring(self._canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
            return image

    def close(self):
        plt.close()
        self._fig = None
