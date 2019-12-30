import numpy as np

from gym.utils.ezpickle import EzPickle
from meta_mb.envs.robotics import hand_env
from meta_mb.envs.robotics.utils import robot_get_obs


FINGERTIP_SITE_NAMES = [
    'robot0:S_fftip',
    'robot0:S_mftip',
    'robot0:S_rftip',
    'robot0:S_lftip',
    'robot0:S_thtip',
]


DEFAULT_INITIAL_QPOS = {
    'robot0:WRJ1': -0.16514339750464327,
    'robot0:WRJ0': -0.31973286565062153,
    'robot0:FFJ3': 0.14340512546557435,
    'robot0:FFJ2': 0.32028208333591573,
    'robot0:FFJ1': 0.7126053607727917,
    'robot0:FFJ0': 0.6705281001412586,
    'robot0:MFJ3': 0.000246444303701037,
    'robot0:MFJ2': 0.3152655251085491,
    'robot0:MFJ1': 0.7659800313729842,
    'robot0:MFJ0': 0.7323156897425923,
    'robot0:RFJ3': 0.00038520700007378114,
    'robot0:RFJ2': 0.36743546201985233,
    'robot0:RFJ1': 0.7119514095008576,
    'robot0:RFJ0': 0.6699446327514138,
    'robot0:LFJ4': 0.0525442258033891,
    'robot0:LFJ3': -0.13615534724474673,
    'robot0:LFJ2': 0.39872030433433003,
    'robot0:LFJ1': 0.7415570009679252,
    'robot0:LFJ0': 0.704096378652974,
    'robot0:THJ4': 0.003673823825070126,
    'robot0:THJ3': 0.5506291436028695,
    'robot0:THJ2': -0.014515151997119306,
    'robot0:THJ1': -0.0015229223564485414,
    'robot0:THJ0': -0.7894883021600622,
}


POINTS_PER_DIM = 20


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class HandReachEnv(hand_env.HandEnv, EzPickle):
    def __init__(
        self, distance_threshold=0.01, n_substeps=20, relative_control=False,
        initial_qpos=DEFAULT_INITIAL_QPOS, reward_type='sparse',
    ):
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type

        hand_env.HandEnv.__init__(
            self, 'hand/reach.xml', n_substeps=n_substeps, initial_qpos=initial_qpos,
            relative_control=relative_control)
        EzPickle.__init__(self)

        # recover achieved goal from observation
        self._goal_size = len(self._get_achieved_goal())

    def _get_achieved_goal(self):
        goal = [self.sim.data.get_site_xpos(name) for name in FINGERTIP_SITE_NAMES]
        return np.array(goal).flatten()

    # Customized methods
    # ----------------------------
    def get_achieved_goal(self, next_obs):
        achieved_goal = next_obs[:, -self._goal_size:]
        return achieved_goal

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv methods
    # ----------------------------

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()

        self.initial_goal = self._get_achieved_goal().copy()
        self.palm_xpos = self.sim.data.body_xpos[self.sim.model.body_name2id('robot0:palm')].copy()

    def _get_obs(self):
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        achieved_goal = self._get_achieved_goal().ravel()
        observation = np.concatenate([robot_qpos, robot_qvel, achieved_goal])
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = self.np_random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        if self.np_random.uniform() < 0.1:
            # With some probability, ask all fingers to move back to the origin.
            # This avoids that the thumb constantly stays near the goal position already.
            goal = self.initial_goal.copy()
        return goal.flatten()

    def _sample_goals(self, num_samples):

        thumb_idx = 4
        finger_idx_vec = self.np_random.choice(4, size=num_samples)

        # Pick a meeting point above the hand.
        meeting_pos_vec = (self.palm_xpos + np.array([0.0, -0.09, 0.05]))[np.newaxis, ...]
        meeting_pos_vec += self.np_random.normal(scale=0.005, size=[num_samples, 3])

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        goal_vec = np.tile(goal[np.newaxis, ...], reps=[num_samples, 1, 1])
        for idx in [thumb_idx, finger_idx_vec]:
            offset_direction = (meeting_pos_vec - goal[idx, :])
            offset_direction /= np.linalg.norm(offset_direction, axis=-1, keepdims=True)
            goal_vec[np.arange(num_samples), idx, :] = meeting_pos_vec - 0.005 * offset_direction

        goal_vec = goal_vec.reshape((num_samples, self._goal_size))

        # With some probability, ask all fingers to move back to the origin.
        # This avoids that the thumb constantly stays near he goal position already.
        goal_vec[self.np_random.uniform(size=num_samples) < 0.1, :, :] = self.initial_goal
        return goal_vec

    def _sample_eval_goals(self):
        thumb_name = 'robot0:S_thtip'
        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        goals = []

        for finger_idx, finger_name in enumerate(FINGERTIP_SITE_NAMES):
            if finger_name == thumb_name:
                continue

            # Pick a meeting point above the hand.
            meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
            x = np.linspace(-0.01, 0.01, num=POINTS_PER_DIM//2)
            y = np.linspace(-0.01, 0.01, num=POINTS_PER_DIM//2)
            xx, yy = np.meshgrid(x, y)
            meeting_pos_offset = np.zeros((POINTS_PER_DIM*POINTS_PER_DIM//4, 3))
            meeting_pos_offset[:, 0] = xx.ravel()
            meeting_pos_offset[:, 1] = yy.ravel()
            # meeting_pos_vec has shape (POINTS_PER_DIM*POINTS_PER_DIM//4, 3)
            # z-position is fixed
            # x, y-positions are spread over the grid of two standard deviation 0.01
            meeting_pos_vec = meeting_pos[np.newaxis, ...] + meeting_pos_offset

            # Slightly move meeting goal towards the respective finger to avoid that they
            # overlap.
            goal = self.initial_goal.copy().reshape(-1, 3)
            goal_vec = np.tile(goal[np.newaxis, ...], reps=[len(meeting_pos_vec), 1, 1])
            for idx in [thumb_idx, finger_idx]:
                offset_direction = meeting_pos_vec - goal[idx, :][np.newaxis, ...]
                offset_direction /= np.linalg.norm(offset_direction, axis=1, keepdims=True)
                goal_vec[:, idx, :] = meeting_pos_vec - 0.005 * offset_direction
            goals.append(goal_vec.reshape((-1, self._goal_size)))

        return np.concatenate(goals, axis=0)

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _render_callback(self):
        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        goal = self.goal.reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'target{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = goal[finger_idx] - sites_offset[site_id]

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = 'finger{}'.format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]
        self.sim.forward()
