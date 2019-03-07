from meta_mb.envs.cassie.assets.build.cassiemujoco import CassieSim, CassieVis, pd_in_t
from meta_mb.logger import logger
import numpy as np
from gym import spaces
import gym
from meta_mb.envs.cassie.utils.eulerangles import quat2euler

# CASSIE_TORQUE_LIMITS = np.array([4.5*25, 4.5*25, 12.2*16, 12.2*16, 0.9*50]) # ctrl_limit * gear_ratio
# CASSIE_MOTOR_VEL_LIMIT = np.array([2900, 2900, 1300, 1300, 5500]) / 60 / (2*np.pi) # max_rpm / 60 / 2*pi
# P_GAIN_RANGE = [10, 10000]
# D_GAIN_RANGE = [1, 100]
# MODEL_TIMESTEP = 0.001
#
# DEFAULT_P_GAIN = 200
# DEFAULT_D_GAIN = 20
#
# NUM_QPOS = 34
# NUM_QVEL = 32
#
# CTRL_COST_COEF = 0.001
# STABILISTY_COST_COEF = 0.01



class CassieEnv(gym.Env, gym.utils.EzPickle):
    # TODO: add randomization of initial state
    _JOINT_NAMES = ['hipRollDrive', 'hipYawDrive', 'hipPitchDrive', 'kneeDrive', 'shinJoint', 'tarsusJoint',
                    'footDrive']

    def __init__(self, render=False, fix_pelvis=False, frame_skip=20,
                 stability_cost_coef=1e-2, ctrl_cost_coef=1e-3, alive_bonus=1, impact_cost_coef=1e-5,
                 rotation_cost_coef=1e-2, policytask='balancing', ctrl_type='T', apply_forces=False):

        gym.utils.EzPickle.__init__(**locals())
        print('fr_skip:', frame_skip, 'task', policytask)

        self.sim = CassieSim()
        if render:
            self.vis = CassieVis()
            print("here!")
        else:
            self.vis = None

        assert ctrl_type in ['T', 'P', 'V', 'TP', 'TV', 'PV', 'TPV']
        # T: Torque ctrl        # TP: Torque + Position ctrl    # None or all: Torque + Position + Velocity
        # P: Positon ctrl       # TV: Torque + Velocity ctrl
        # V: Velocity ctrl      # PV: Position + Velocity ctr

        self.parameters = {}
        self.set_default_parameters()
        self.fix_pelvis = fix_pelvis
        self.model_timestep = 0.0005 * frame_skip
        self.frame_skip = frame_skip
        self.task = policytask
        self.ctrl_type = ctrl_type
        self._pd_params_to_set = []
        self.apply_forces = apply_forces

        # action and observation space specs
        self.act_dim = len(self.torque_limits)

        self.num_qpos = self.parameters['num_qpos']
        self.num_qvel = self.parameters['num_qvel']
        # self.obs_dim = 66
        self.obs_dim = 40

        # reward function coeffs
        self.stability_cost_coef = stability_cost_coef
        self.ctrl_cost_coef = ctrl_cost_coef
        self.impact_cost_coef = impact_cost_coef
        self.alive_bonus = alive_bonus
        self.rotation_cost_coef = rotation_cost_coef
        self._time_step = 0

        if fix_pelvis: self.sim.hold()

    def _cassie_state_to_obs(self, int_state):
        pelvis_pos_rel_to_r_foot = -np.array(int_state.rightFoot.position).astype(np.float32)

        # pelvis
        pelvis_ori = quat2euler(np.array(int_state.pelvis.orientation).astype(np.float32))
        pelvis_rot_vel = np.array(int_state.pelvis.rotationalVelocity).astype(np.float32)
        pelvis_transl_vel = np.array(int_state.pelvis.translationalVelocity).astype(np.float32)

        # joints
        joint_pos = np.array(int_state.joint.position).astype(np.float32)
        joint_vel = np.array(int_state.joint.velocity).astype(np.float32)

        # motors
        motor_pos = np.array(int_state.motor.position).astype(np.float32)
        motor_vel = np.array(int_state.motor.position).astype(np.float32)

        qpos = np.array([motor_pos[0], motor_pos[1], motor_pos[2], motor_pos[3], joint_pos[0], joint_pos[1], motor_pos[4],
                         motor_pos[5], motor_pos[6], motor_pos[7], motor_pos[8], joint_pos[3], joint_pos[4], motor_pos[9]])

        qvel = np.array([motor_vel[0], motor_vel[1], motor_vel[2], motor_vel[3], joint_vel[0], joint_vel[1], motor_vel[9],
             motor_vel[5], motor_vel[6], motor_vel[7], motor_vel[8], joint_vel[3], joint_vel[4], motor_vel[4]])

        obs = np.concatenate([pelvis_pos_rel_to_r_foot, pelvis_ori, qpos,
                              pelvis_transl_vel, pelvis_rot_vel, qvel])

        # qpos = self.sim.get_state().qpos()
        # qvel = self.sim.get_state().qvel()
        # obs = np.concatenate([qpos, qvel])

        return obs

    def step(self, action):
        assert action.ndim == 1 and action.shape == (self.act_dim,)
        u = self._action_to_pd_u(action * 5)
        if self.apply_forces and self._time_step % 10 == 0:
            self.apply_random_force()
        state, internal_state = self.do_simulation(u, self.frame_skip)
        obs = self._cassie_state_to_obs(internal_state)

        reward, agent_infos = self.reward(internal_state, state, action)

        done = self.done(state)
        return obs, reward, done, agent_infos

    def reset(self):
        self.sim = CassieSim()
        if self.fix_pelvis: self.sim.hold()
        u = self._action_to_pd_u(np.zeros(self.act_dim,))
        internal_state = self.sim.step_pd(u)
        self._time_step = 0
        return self._cassie_state_to_obs(internal_state)

    def do_simulation(self, u, n_frames):
        assert n_frames >= 1
        for _ in range(n_frames):
            internal_state_obj = self.sim.step_pd(u) # step_pd returns state_out_t structure -> however this structure is still not fully understood
        joint_state = self.sim.get_state() # get CassieState object
        return joint_state, internal_state_obj

    def done(self, state):
        pelvis_pos = np.array(state.qpos())
        return pelvis_pos[2] < 0.65

    def reward(self, internal_state, state, action):
        # reward fct
        qvel = np.array(state.qvel())
        pelvis_rot_vel = qvel[3:6]
        pelvis_transl_vel = qvel[:3]

        foot_forces = self.get_foot_forces(internal_state)
        motor_torques = _to_np(internal_state.motor.torque)
        forward_vel = pelvis_transl_vel[0]
        ctrl_cost = self.ctrl_cost_coef * 0.5 * np.mean(np.square(motor_torques/(2 * self.torque_limits)))
        stability_cost = self.stability_cost_coef * 0.5 * np.mean(np.square(pelvis_transl_vel[1:]))  #  quadratic velocity of pelvis in y and z direction ->
        rotation_cost = self.rotation_cost_coef * 0.5 * np.mean(np.square(pelvis_rot_vel))
                                                                                              #  enforces to hold the pelvis in same position while walking
        impact_cost = self.impact_cost_coef * 0.5 * np.sum(np.square(np.clip(foot_forces, -1, 1)))
        if self.task == 'balancing':
            vel_cost = self.stability_cost_coef * forward_vel ** 2
            reward = - vel_cost - ctrl_cost - stability_cost - impact_cost + self.alive_bonus
        elif self.task == 'fixed-vel':
            vel_reward = np.exp(- (2.3 - forward_vel) ** 2)
            reward = vel_reward - ctrl_cost - stability_cost - rotation_cost - impact_cost + self.alive_bonus
        else:
            reward = forward_vel - ctrl_cost - stability_cost - rotation_cost - impact_cost + self.alive_bonus
        return reward, dict(ctrl_cost=ctrl_cost, forward_vel=forward_vel, stability_cost=stability_cost)

    def render(self, *args, **kwargs):
        if self.vis is None:
            print('Setting up cassie visualizer')
            self.setup_cassie_vis()
        self.vis.draw(self.sim)

    def get_foot_forces(self, internal_state):
        left_toe = _to_np(internal_state.leftFoot.toeForce)
        left_heel = _to_np(internal_state.leftFoot.heelForce)
        right_toe = _to_np(internal_state.rightFoot.toeForce)
        right_heel = _to_np(internal_state.rightFoot.heelForce)
        return np.concatenate([left_toe, left_heel, right_toe, right_heel])

    def apply_random_force(self):
        force = np.zeros((6,))
        y_force = np.random.choice([0, 10, 25, 50]) * np.random.choice([-1, 1])
        force[1] = y_force
        self.sim.apply_force(force)

    @property
    def torque_limits(self):
        return np.array([4.5, 4.5, 12.2, 12.2, 0.9] * 2)

    @property
    def dt(self):
        return self.model_timestep

    @property
    def action_space(self):
        return spaces.Box(low=-self.torque_limits, high=self.torque_limits, dtype=np.float32)

    @property
    def observation_space(self):
        obs_limit = np.inf * np.ones(self.obs_dim)
        return spaces.Box(-obs_limit, obs_limit, dtype=np.float32)

    def setup_cassie_vis(self):
        self.vis = CassieVis()

    def _action_to_pd_u(self, action):
        """
        motors:
        0: hip abduction
        1: hip twist
        2: hip pitch -> lift leg up
        3: knee
        4: foot pitch

        Typical pGain ~ 200 [100, 10000]
        Typical dGain ~ 20
        Typical feedforward torque > 0
        """

        u = pd_in_t()
        for i, leg_name in enumerate(['leftLeg', 'rightLeg']):
            leg = getattr(u, leg_name)
            for motor_id in range(5):
                leg.motorPd.torque[motor_id] = action[i * 5 + motor_id]
                leg.motorPd.pGain[motor_id] = 0
                leg.motorPd.dGain[motor_id] = 0
                leg.motorPd.dTarget[motor_id] = 0
                leg.motorPd.pTarget[motor_id] = 0
        return u

    def set_default_parameters(self):
        self.parameters = dict(cassie_torque_limits=np.array([4.5, 4.5, 12.2, 12.2, 0.9] * 2),
                               cassie_motor_vel_limits=np.array([2900, 2900, 1300, 1300, 5500]) / 60 / (2 * np.pi), # max_rpm / 60 / 2*pi
                               p_gain_range=[10, 10000],
                               d_gain_range=[1, 100],
                               pGain=200,
                               dGain=20,
                               num_qpos=34,
                               num_qvel=32,
                               ctrl_cost_coef=0.001,
                               stability_cost_coef=0.01,)

    def log_diagnostics(self, paths, prefix):
        forward_vel = [np.mean(path['env_infos']['forward_vel']) for path in paths]
        ctrl_cost = [np.mean(path['env_infos']['ctrl_cost']) for path in paths]
        stability_cost = [np.mean(path['env_infos']['stability_cost']) for path in paths]
        path_length = [path["observations"].shape[0] for path in paths]

        logger.logkv('AvgForwardVel', np.mean(forward_vel))
        logger.logkv('StdForwardVel', np.std(forward_vel))
        logger.logkv('AvgCtrlCost', np.mean(ctrl_cost))
        logger.logkv('AvgStabilityCost', np.mean(stability_cost))
        logger.logkv('AvgPathLength', np.mean(path_length))


def _to_np(o, dtype=np.float32):
    return np.array([o[i] for i in range(len(o))], dtype=dtype)


if __name__ == '__main__':
    render = True
    env = CassieEnv(render=render, frame_skip=20)
    import time

    for i in range(5):
        obs = env.reset()
        for j in range(50000):
            cum_forward_vel = 0
            act = env.action_space.sample()
            obs, reward, done, info = env.step(act * 10)
            if render:
                env.render()
            time.sleep(.2)
            # if done:
            #     break
