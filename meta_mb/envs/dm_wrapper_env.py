from meta_mb.utils.serializable import Serializable
import numpy as np
import gym
import tensorflow as tf
from collections.abc import Iterable


class DeepMindWrapper(Serializable):
  """Wraps a DM Control environment into a Gym interface."""

  metadata = {'render.modes': ['rgb_array']}
  reward_range = (-np.inf, np.inf)

  def __init__(self, env, render_size=(64, 64), camera_id=0):
    Serializable.quick_init(self, locals())
    self._env = env
    self._render_size = render_size
    self._camera_id = camera_id

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    components = {}
    for key, value in self._env.observation_spec().items():
      components[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    return gym.spaces.Dict(components)

  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return gym.spaces.Box(
        action_spec.minimum, action_spec.maximum, dtype=np.float32)

  def step(self, action):
    time_step = self._env.step(action)
    obs = dict(time_step.observation)
    reward = time_step.reward or 0
    done = time_step.last()
    info = {'discount': time_step.discount}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    return dict(time_step.observation)

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    del args  # Unused
    del kwargs  # Unused
    return self._env.physics.render(
        *self._render_size, camera_id=self._camera_id)

  def reset_from_obs(self, observation):
    # support only one observation
    self._env._physics.reset_from_obs(observation)

  def tf_reward(self, obs, act, nextobs):
    # support vector
    return self._env._physics.reward(obs, act, nextobs)


class ConcatObservation(Serializable):
  """Select observations from a dict space and concatenate them."""

  def __init__(self, env, keys):
    Serializable.quick_init(self, locals())
    self._env = env
    self._keys = keys

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def observation_space(self):
    spaces = self._env.observation_space.spaces
    spaces = [spaces[key] for key in self._keys]
    low = np.concatenate([space.low if space.low.ndim > 0 else space.low[None] for space in spaces], 0)
    high = np.concatenate([space.low if space.high.ndim >  0 else space.high[None] for space in spaces], 0)
    dtypes = [space.dtype for space in spaces]
    if not all(dtype == dtypes[0] for dtype in dtypes):
      message = 'Spaces must have the same data type; are {}.'
      raise KeyError(message.format(', '.join(str(x) for x in dtypes)))
    return gym.spaces.Box(low, high, dtype=dtypes[0])

  def step(self, action):
    obs, reward, done, info = self._env.step(action)
    obs = self._select_keys(obs)
    return obs, reward, done, info

  def reset(self):
    obs = self._env.reset()
    obs = self._select_keys(obs)
    return obs

  def _select_keys(self, obs):
    return np.concatenate([obs[key] if obs[key].ndim > 0 else obs[key][None] for key in self._keys], 0)

  def render(self, *args, **kwargs):
      return self._env.render(*args, **kwargs)


class ActionRepeat(Serializable):
  """Repeat the agent action multiple steps."""

  def __init__(self, env, amount):
    Serializable.quick_init(self, locals())
    self._env = env
    self._amount = amount

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    done = False
    total_reward = 0
    current_step = 0
    while current_step < self._amount and not done:
      observ, reward, done, info = self._env.step(action)
      total_reward += reward
      current_step += 1
    return observ, total_reward, done, info

  def render(self, *args, **kwargs):
      return self._env.render(*args, **kwargs)

  @property
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
      return self._env.action_space

  def tf_reward(self, obs, acts, next_obs):
    return tf.zeros_like(acts)[:, 0]

  def reward(self, obs, acts, next_obs):
    return 0