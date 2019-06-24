import numpy as np
from gym.envs.mujoco import mujoco_env
from meta_mb.meta_envs.base import RandomEnv
from gym import utils
import os


class BlueEnv(RandomEnv, utils.EzPickle): 
    def __init__(self, arm='right', log_rand=0, actions=None):
        utils.EzPickle.__init__(**locals())

        assert arm in ['left', 'right']
        xml_file = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'assets', 'blue_' + arm + '_v2.xml')

        self.goal = np.zeros((3,))
        self._arm = arm

        if actions is not None:
            self.actions = actions
            self.path_len = 0
            self.max_path_len = len(actions)

        max_torques = np.array([5, 5, 4, 4, 3, 2, 1])
        self._low = -max_torques
        self._high = max_torques

        RandomEnv.__init__(self, log_rand, xml_file, 20)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat[:-3],
            self.get_body_com("right_gripper_link"),
            self.ee_position - self.goal,
        ])

    def step(self, action):
        done = False
        if (hasattr(self, "actions")):
            action = self.actions[self.path_len]
            self.path_len += 1
            if(self.path_len == self.max_path_len):
                done = True

        self.do_simulation(action, self.frame_skip)
        #self.correction() # Use for v2 arms
        vec = self.ee_position - self.goal
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(action/(2 * self._high)).sum()
        reward = reward_dist + 0.5 * 0.1 * reward_ctrl
        observation = self._get_obs()
        info = dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return observation, reward, done, info

    def reward(self, obs, act, obs_next):
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_ctrl = -0.5 * 0.1 * np.sum(np.square(act / (2 * self._high)), axis=1)
            reward_dist = -np.linalg.norm(obs_next[:, -3:], axis=1)
            reward = reward_dist + reward_ctrl
            return np.clip(reward, -1e2, 1e2)

        elif obs.ndim == 1:
            return self.reward(obs[None], act[None], obs_next[None])[0]

        else:
            raise NotImplementedError

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        #self.goal = np.random.uniform(low=[-0.75, -0.25, 0.25], high=[-0.25, 0.25, 0.5])
        self.goal = np.array([-0.55, -0.1, 0.21]) #fixed goal
        qpos[-3:] = self.goal
        qvel[-3:] = 0

        self.set_state(qpos, qvel)
        observation = self._get_obs()
        return observation

    def correction(self):
        try:
            qpos = self.sim.data.qpos.flat
            qvel = self.sim.data.qvel.flat[:-3]
            shoulder_lift_joint = qpos[1]
            if shoulder_lift_joint > -0.01:
                correction = 0.5 * abs(shoulder_lift_joint) + shoulder_lift_joint
            qpos[1] = correction
            qvel[1] = qvel[1] * 0.9
            print("correction")
            self.set_state(qpos, qvel)
        except:
            pass

    @property
    def ee_position(self):
        return (self.get_body_com(self._arm + '_r_finger_tip_link')
                + self.get_body_com(self._arm + '_l_finger_tip_link'))/2

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 2
        self.viewer.cam.elevation = -20
        self.viewer.cam.type = 0


if __name__ == "__main__":
    env = BlueEnv('right')
    while True:
        env.reset()
        for _ in range(200):
            action = env.action_space.sample()
            env.step(action)
            env.render()
