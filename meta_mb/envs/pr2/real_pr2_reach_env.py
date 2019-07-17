import numpy as np
import zmq
from meta_mb.logger import logger
import gym
from gym import spaces
from meta_mb.meta_envs.base import MetaEnv
import tensorflow as tf
from meta_mb.utils.serializable import Serializable
import time


class PR2ReachEnv(MetaEnv, Serializable):
    PR2_GAINS = np.array([3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098])

    def __init__(self,
                 exp_type='shape_mod',
                 torque_penalty=1.25e-1,
                 vel_penalty=1.25e-1,
                 max_torques=[3] * 7):
        Serializable.quick_init(self, locals())

        self.norm = 1000

        self.exp_type = exp_type
        if exp_type == 'shape_joints':
            self.goal = np.array([ 3.48729010e-01,  9.44079044e-02,  1.45437872e+00, -1.55104661e+00,
  1.59658259e+00, -1.73600719e+00, -2.58865643e-01, -3.28908378e-01,  1.70644299e-01,
 -3.33109042e-02, -1.97651754e-01,  1.57414804e-01, -2.82707699e-02,
 -3.05237283e-01,  8.85290046e-02,  3.89394478e-02]
  )

            self.init_qpos = np.array([3.85207921e-01, -1.41945343e-01,  1.64343706e+00, -1.51601210e+00,
                                      1.31405443e+00, -1.54883181e+00])
            self.init_obs = np.array([3.85207921e-01, -1.41945343e-01,  1.64343706e+00, -1.51601210e+00,
  1.31405443e+00, -1.54883181e+00,  1.43069760e-01, 0, 0, 0, 0, 0, 0, 0,
                                       -3.06694218e-01, 1.87049223e-01,  1.12720687e-03,
                                       -2.03442785e-01,  1.59440809e-01, 1.02890217e-02,
                                       -3.07411827e-01,  1.18937711e-01,  7.99029507e-02])
            """
            Init pos
[ 3.08021861e-01 -2.06834587e-01  1.76161857e+00 -1.42335884e+00
  1.25823199e+00 -1.39824744e+00  7.36730329e-02  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  4.44089210e-15
  0.00000000e+00  0.00000000e+00 -2.79665121e-01  1.49428436e-01
 -4.42059122e-03 -1.64933584e-01  1.26288441e-01  9.70061776e-03
 -2.71695377e-01  7.52098626e-02  7.38003843e-02]

 
 Goal
[ 3.48729010e-01,  9.44079044e-02,  1.45437872e+00, -1.55104661e+00,
  1.59658259e+00, -1.73600719e+00, -2.58865643e-01, -3.28908378e-01,  1.70644299e-01,
 -3.33109042e-02, -1.97651754e-01,  1.57414804e-01, -2.82707699e-02,
 -3.05237283e-01,  8.85290046e-02,  3.89394478e-02]
  
  
  Init Pos (Worst Case)
 [3.85207921e-01, -1.41945343e-01,  1.64343706e+00, -1.51601210e+00,
  1.31405443e+00, -1.54883181e+00,  1.43069760e-01, -3.06694218e-01,
  1.87049223e-01,  1.12720687e-03, -2.03442785e-01,  1.59440809e-01,
  1.02890217e-02, -3.07411827e-01,  1.18937711e-01,  7.99029507e-02]



        """

        elif exp_type == 'shape_gripper':
            self.goal = np.array([-3.28908378e-01,  1.70644299e-01, -3.33109042e-02,
                                  -1.97651754e-01,  1.57414804e-01, -2.82707699e-02,
                                  -3.05237283e-01,  8.85290046e-02,  3.89394478e-02])

            self.init_qpos = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                       1.31405443e+00, -1.54883181e+00])
            self.init_obs = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                      1.31405443e+00, -1.54883181e+00, 1.43069760e-01, 0, 0, 0, 0, 0, 0, 0,
                                      -3.06694218e-01, 1.87049223e-01, 1.12720687e-03,
                                      -2.03442785e-01, 1.59440809e-01, 1.02890217e-02,
                                      -3.07411827e-01, 1.18937711e-01, 7.99029507e-02])
        elif exp_type == 'shape_mod':
            self.goal = np.array([-0.40205352,  0.13775634, -0.03672675,
                                  -0.27075515,  0.12438263, -0.02908585,
                                  -0.37856982,  0.05570664,  0.03472728])

            self.init_qpos = np.array([0.26267194,  0.06945243,  1.13799865, -1.73577404,  1.48262084, -1.9809189,
            -1.54233528])

            self.init_obs = np.array([0.26267194,  0.06945243,  1.13799865, -1.73577404,  1.48262084, -1.9809189,
            -1.54233528,  0.,          0.,          0.,          0.,          0.,
             0.,          0.,         -0.4407883,   0.09142632,  0.07738364, -0.2684554,
             0.163703,    0.08128676, -0.33237608,  0.02133017,  0.14725503])

            """
            Shape Mod
            [-0.40205352,  0.13775634, -0.03672675,
             -0.27075515,  0.12438263, -0.02908585,
             -0.37856982,  0.05570664,  0.03472728]
            
            
            Init Pos
            [0.26267194,  0.06945243,  1.13799865, -1.73577404,  1.48262084, -1.9809189,
            -1.54233528,  0.,          0.,          0.,          0.,          0.,
             0.,          0.,         -0.4407883,   0.09142632,  0.07738364, -0.2684554,
             0.163703,    0.08128676, -0.33237608,  0.02133017,  0.14725503]
            
            Goal
            [-0.41822912,  0.15120888, -0.03849079,
             -0.29876662,  0.13038571, -0.02571729,
             -0.40618667,  0.07449564,  0.03775624]
             
             [-0.4357618,   0.14885474, -0.03789454,
              -0.2949085,   0.14328422, -0.02922774,
              -0.40151519,  0.06359246,  0.03629649]
            """


        elif exp_type == 'peg':
             raise NotImplementedError
        elif exp_type == 'bottle':
            self.init_qpos = np.array([0.09536639, -0.05727076,  1.07930878, -1.04463866,  1.71724862, -2.00068935,
                                     -1.04508426])
            self.init_obs = np.array([0.09536639, -0.05727076,  1.07930878, -1.04463866,  1.71724862, -2.00068935,
                                     -1.04508426,  0.,          0.,          0.,          0.,          0.,
                                      0.,          0.,         -0.27317487,  0.06260623,  0.12906069, -0.1010125,
                                      0.14036395,  0.1370379,  -0.16080087, -0.00439068,  0.20217985])
            self.goal = np.array([-0.27337845,  0.0623045,   0.08042455,
                                  -0.10072089,  0.13381461,  0.09027978,
                                  -0.16556836, -0.00824566,  0.15118447])

        elif exp_type == 'reach':
            self.init_qpos = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                       1.31405443e+00, -1.54883181e+00, 1.43069760e-01])

            self.init_obs = np.array([3.85207921e-01, -1.41945343e-01, 1.64343706e+00, -1.51601210e+00,
                                      1.31405443e+00, -1.54883181e+00, 1.43069760e-01, 0, 0, 0, 0, 0, 0, 0,
                                      -3.06694218e-01, 1.87049223e-01, 1.12720687e-03,
                                      -2.03442785e-01, 1.59440809e-01, 1.02890217e-02,
                                      -3.07411827e-01, 1.18937711e-01, 7.99029507e-02])

            self.goal = np.array([-0.30982005,  0.71146246,  0.21908543, -0.14216614,
                                   0.78684261,  0.26139753, -0.20410874,  0.64335638,  0.31437626])

            """
            Reach Goal
            [0.95775596, -0.27941703,  1.13751759 -0.65259902  1.5608301  -1.76946555
             0.16095193, -0.01667959,  0.          0.         -0.00289525  0.
             0.,          0.,         -0.30982005  0.71146246  0.21908543 -0.14216614
             0.78684261,  0.26139753, -0.20410874  0.64335638  0.31437626]
            """
        elif exp_type == 'lego':
            self.init_qpos = np.array([0.32054076, -0.05439454,  1.652898,   -1.2846685,   1.47029941, -1.45872484,
                                      0.53286618])
            self.init_obs = np.array([0.32054076, -0.05439454,  1.652898,   -1.2846685,   1.47029941, -1.45872484,
                                      0.53286618,  0.,          0.,          0.,          0.,          0.,
                                      0.,          0.,         -0.20574476,  0.17039927, -0.03583246, -0.12496017,
                                      0.13846403, -0.01859916, -0.22157128,  0.11702173,  0.03809535])
            self.goal = np.array([-2.08417832e-01,  1.55992459e-01, -9.93656620e-02,
                                  -1.28845066e-01,  1.23824094e-01, -8.64020558e-02,
                                  -2.24913514e-01,  1.03418447e-01, -2.82650796e-02])

        elif exp_type == 'lego_mod':
            self.init_qpos = np.array([0.53634668, -0.07985761,  1.12051997, -1.62009586,
                                       1.32938391, -2.00258634,  0.3364669])

            self.init_obs = np.array([0.53634668, -0.07985761,  1.12051997, -1.62009586,  1.32938391, -2.00258634,
                                      0.33646691,  0.,          0.,          0.,          0.00302298,  0.,
                                      0.,          0.,         -0.33515356,  0.26675175,  0.13111669, -0.25692461,
                                      0.23463285,  0.14815731, -0.35199802,  0.21565351,  0.21306161])

            self.goal = np.array([-0.35562659,  0.1141765,  -0.10301183,
                                  -0.27585707,  0.08202434, -0.09264046,
                                  -0.37196246,  0.06152,    -0.03529104])

            """
            [-0.35562659,  0.1141765,  -0.10301183,
             -0.27585707,  0.08202434, -0.09264046,
             -0.37196246,  0.06152,    -0.03529104]
            
            
            Lego Mod
            Init Pos
             [0.47856077 -0.07918085  1.25008667 -0.94040375  1.70405949 -1.88341541
              1.07703224  0.          0.          0.          0.          0.
              0.          0.         -0.14661003  0.38439563  0.09342872 -0.06981191
              0.35207254  0.10436448 -0.16461223  0.33383388  0.16475069]

            Mid Point
            [ 3.10426153e-01 -6.64916096e-02  1.52942478e+00 -1.71130779e+00
              1.40927073e+00 -1.60739482e+00 -1.42003170e+00  0.00000000e+00
              0.00000000e+00  0.00000000e+00  1.75169657e-09  0.00000000e+00
              0.00000000e+00  0.00000000e+00 -4.28747575e-01  9.45141253e-02
              6.92418854e-03 -2.56404145e-01  1.63419403e-01  9.74860450e-03
             -3.22991869e-01  2.26133317e-02  7.45412423e-02]

            Goal
            [ 2.92518323e-01  5.10953336e-02  1.72634051e+00 -1.67772098e+00
              1.51067668e+00 -1.44950095e+00 -1.41407098e+00  0.00000000e+00
              0.00000000e+00 -1.86963334e-09  0.00000000e+00  0.00000000e+00
              0.00000000e+00  0.00000000e+00 -4.33270007e-01  9.18991318e-02
             -9.43792175e-02 -2.61428022e-01  1.63224017e-01 -9.55503864e-02
             -3.26072146e-01  2.12508344e-02 -3.00473105e-02]
            """


            """
 [0.27154295, -0.31469311,  1.39184034, -1.11210181,  1.24261327, -1.7384872,
  0.5793337,   0.,          0.,          0.,          0.,          0.,
  0.,          0.,         -0.21384546,  0.13586604,  0.17599293, -0.13454249,
  0.10505592,  0.20431746, -0.23049067,  0.08440514,  0.25743059]
  
  
  
  
  Init Pos
 [0.32054076, -0.05439454,  1.652898,   -1.2846685,   1.47029941, -1.45872484,
  0.53286618,  0.,          0.,          0.,          0.,          0.,
  0.,          0.,         -0.20574476,  0.17039927, -0.03583246, -0.12496017,
  0.13846403, -0.01859916, -0.22157128,  0.11702173,  0.03809535]

  Goal
  2.82735342e-01  1.18010958e-01  1.58987917e+00 -1.26932280e+00
  1.68300315e+00 -1.50114604e+00  5.11590314e-01  0.00000000e+00
  1.71434136e-03 -3.19587486e-03  0.00000000e+00  4.51286036e-06
  2.05169215e-12  3.56203955e-11 -2.08417832e-01  1.55992459e-01
 -9.93656620e-02 -1.28845066e-01  1.23824094e-01 -8.64020558e-02
 -2.24913514e-01  1.03418447e-01 -2.82650796e-02]


            """

        else:
            raise NotImplementedError

        """
       [-2.01591996e-01,  8.36417201e-02,  6.44355452e-03,
        -1.25248757e-01,  5.13652953e-02,  2.18814130e-02,
        -2.19660074e-01,  3.36434857e-02,  8.49410456e-02]
        
        
        
 [0.20314838, -0.1314542,   1.45676857, -1.11673447,  1.38387586, -1.70250532,
  0.5676298,   0.00492352,  0.01532753, -0.05476294,  0.,          0.,
  0.,          0.,         -0.20603113,  0.11275647,  0.0637374,  -0.12700435,
  0.08092093,  0.08415633, -0.2226102,   0.06130295,  0.14567541]
  
  
  
 [1.38063295e-01, -9.40695720e-02,  1.58715252e+00, -1.08112088e+00,
  1.35975363e+00, -1.60948325e+00,  5.61364518e-01, -2.07052837e-01,
  8.15807391e-02,  8.37895106e-03, -1.27846596e-01,  4.95414323e-02,
  2.58043992e-02, -2.23361214e-01,  2.98273786e-02,  9.06094677e-02]

"""

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        print("Connecting to the server...")
        self.socket.connect("tcp://127.0.0.1:7777")
        max_torques = np.array(max_torques)
        self.frame_skip = 2
        self.dt = 0.2


        self.vel_penalty = vel_penalty
        self.torque_penalty = torque_penalty
        self.act_dim = 7
        self.obs_dim = 23
        self._t = 0
        self._low, self._high = -max_torques, max_torques

    def step(self, action):
        ob = self.do_simulation(action, self.frame_skip)
        alpha = 1e-5
        reward_vel = -self.vel_penalty * np.square(np.linalg.norm(ob[7:14]))
        reward_ctrl = -self.torque_penalty * np.square(np.linalg.norm(action))
        _norm_end = np.linalg.norm((ob[-9:] - self.goal[-9:]))

        self.norm = _norm_end

        scaled_norm_end = 3 * _norm_end
        if self.exp_type == 'shape_joints':
            norm_end = np.linalg.norm(ob[:7] - self.goal[:7]) + scaled_norm_end
            #norm_end = np.linalg.norm(np.concatenate([ob[:7], ob[-9:]], axis=-1) - self.goal)
        elif self.exp_type != 'shape_joints':
            norm_end = np.linalg.norm((ob[-9:] - self.goal))
        else:
            raise NotImplementedError

        if self.exp_type == 'lego':
            reward_dist = -(np.square(norm_end) + np.log(np.square(norm_end) + alpha))
        else:
            reward_dist = -(np.square(norm_end) + np.log(np.square(norm_end) + alpha))
        reward = reward_vel + reward_dist + reward_ctrl
        done = False
        self._t += 1
        return ob, reward, done, dict(dist=_norm_end, reward_vel=reward_vel, reward_ctrl=reward_ctrl), #reward_vel=reward_vel)

    def do_simulation(self, action, frame_skip):
        assert frame_skip > 0
        if action.ndim == 2:
            action = action.reshape(-1)
        action = np.clip(action, self._low, self._high).astype(np.float32)
        for _ in range(frame_skip):
            md = dict(
                dtype=str(action.dtype),
                cmd="action",
            )
            if self.norm < 0.07 and self.exp_type == 'bottle':
                action[6] = 9
            self.socket.send_json(md, 0 | zmq.SNDMORE)
            self.socket.send(action, 0, copy=True, track=False)
            ob = self._get_obs()
        return ob

    def reward(self, obs, act, obs_next):
        goal = self.goal
        alpha = 1e-5
        assert obs.ndim == act.ndim == obs_next.ndim
        if obs.ndim == 2:
            assert obs.shape == obs_next.shape and act.shape[0] == obs.shape[0]
            reward_vel = -self.vel_penalty * np.square(np.linalg.norm(obs_next[:, 7:14], axis=1))
            reward_ctrl = - self.torque_penalty* np.square(np.linalg.norm(act, axis=1))
            if self.exp_type == 'shape_joints':
                norm_end = np.linalg.norm(np.concatenate([obs_next[:, :7], obs_next[:, -9:]], axis=-1) - goal, axis=1)
            elif self.exp_type != 'shape_joints':
                norm_end = np.linalg.norm((obs_next[:, -9:] - goal), axis=1)
            else:
                raise NotImplementedError
            if self.exp_type == 'lego':
                reward_dist = -(np.square(norm_end) + np.log(0.1 * np.square(norm_end) + alpha))
            else:
                reward_dist = -(np.square(norm_end) + np.log(np.square(norm_end) + alpha))
            reward = reward_vel + reward_dist + reward_ctrl
            return np.clip(reward, -100, 100)
        elif obs.ndim == 1:
            assert obs.shape == obs_next.shape
            return self.reward(obs[None], act[None], obs_next[None])[0]
        else:
            raise NotImplementedError

    def tf_reward(self, obs, acts, next_obs):
        alpha = 1e-5
        reward_vel = - self.vel_penalty * tf.square(tf.linalg.norm(next_obs[:, 7:14], axis=1))
        reward_ctrl = - self.torque_penalty * tf.square(tf.linalg.norm(acts, axis=1))
        scaled_norm_end = tf.linalg.norm(next_obs[:, -9:] - self.goal[-9:], axis=1)
        if self.exp_type == 'shape_joints':
            norm_end = tf.linalg.norm(obs[:, :7] - self.goal[:7], axis=1) + scaled_norm_end
            #norm_end = tf.linalg.norm(
                #(tf.concat([next_obs[:, :7], next_obs[:, -9:]], axis=-1) - self.goal)
                #, axis=1)
        elif self.exp_type != 'shape_joints':
            norm_end = tf.linalg.norm((next_obs[:, -9:] - self.goal), axis=1)
        else:
            raise NotImplementedError

        if self.exp_type == 'lego':
            reward_dist = -(tf.square(norm_end) + tf.log(tf.square(norm_end) + alpha))
        else:
            reward_dist = -(tf.square(norm_end) + tf.log(tf.square(norm_end) + alpha))
        reward = reward_vel + reward_dist + reward_ctrl
        return reward

    def reset(self, *args, **kwargs):
        self._t = 0
        qpos = self.init_qpos
        md = dict(
            dtype=str(qpos.dtype),
            cmd="reset",

        )
        self.socket.send_json(md, 0 | zmq.SNDMORE)  # buffer
        self.socket.send(qpos, 0, copy=True, track=False)
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        msg = self.socket.recv(flags=0, copy=True, track=False)
        buf = memoryview(msg)
        obs = np.frombuffer(buf, dtype=np.float64)
        return obs  # This returns [7 joint angles, 7 joint vel, 3 x-y-z of gripper]

    def log_diagnostics(self, paths, prefix=''):
        dist = [path["env_infos"]['dist'] for path in paths]
        final_dist = [path["env_infos"]['dist'][-1] for path in paths]
        vel = [path["env_infos"]['reward_vel'] for path in paths]
        ctrl = [path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgDistance', np.mean(dist))
        logger.logkv(prefix + 'AvgFinalDistance', np.mean(final_dist))
        logger.logkv(prefix + 'AvgControlCost', np.mean(ctrl))
        logger.logkv(prefix + 'AvgVelocityCost', np.mean(vel))

    @property
    def action_space(self):
        return spaces.Box(low=self._low, high=self._high, dtype=np.float32)

    @property
    def observation_space(self):
        low = np.ones(self.obs_dim) * -1e6
        high = np.ones(self.obs_dim) * 1e6
        return spaces.Box(low=low, high=high, dtype=np.float32)

    @property
    def idx(self):
        return 0


if __name__ == "__main__":
    env = PR2ReachEnv()
    print("reset!")
    obs = env.reset()
    # time.sleep(2.)
    #obs = np.expand_dims(, axis=0)
    #print(env._init_obs)
    print("reset done!")
    i = 0
    while True:
        print("action!")
        a = env.action_space.sample() * 0
        #a[6] = (np.pi/2) * i
        obs, rew, done, info = env.step(a)
        print(obs)

"""
   Init pos: 
   [-7.30998593e-02, -1.15979640e-01,  1.54449814e+00, -1.11630016e+00,
   1.40435373e+00, -1.69883237e+00, -1.76086815e-02, -2.91515588e-01, -9.26095481e-02,
   5.60259465e-02, -2.18762961e-01, -2.50179475e-02,  6.77916120e-02,
  -2.36602619e-01, -9.01772915e-02,  1.27465781e-01]

  Goal pos:
  [ 0.66692461, -0.35267623,  0.80349848, -1.21836351,  1.33325966, -2.00243762,
  0.0725418, -0.30728344,  0.31105323,  0.37427125, -0.23453081,
  0.37864483,  0.38603692, -0.25237047,  0.31348548,  0.44571109]
  
  Peg pos:
  [0.14967022  0.07190568  1.48244082 -0.95690761  
   1.65639349 -1.54128657  0.19380116 -0.1646487   
   0.06718034 -0.03346684 -0.09189608  0.13477194 
  -0.02170117 -0.10973574  0.06961259  0.037973]
  
  
New peg:
 
[ 4.18868008e-01,  2.85507865e-01,  1.45036985e+00, -1.39397039e+00,
  1.82038998e+00, -1.62782107e+00, -8.96121673e-02, -2.65874405e-01,
  1.65092658e-01, -1.06816831e-01, -1.93121779e-01,  2.32684259e-01, 
 -9.50511654e-02, -2.10961437e-01,  1.67524915e-01, -3.53769959e-02]
 
 
Init pos:
[ 0.49970196,  0.15269384,  1.16413649, -1.38325735,  1.74894883, -1.88839618,
 -0.35949815,  0.,          0.,          0.,          0.,          0.,
  0.,          0.,         -0.24655252,  0.24108528,  0.03895606, -0.17379989,
  0.30867688,  0.05072172, -0.19163955,  0.24351754,  0.11039589]
  
Above:
[ 4.25915071e-01,  1.20209263e-01,  1.55604368e+00, -1.43045261e+00,
  1.63180848e+00, -1.55468276e+00, -6.38651381e-01, -3.61566944e-01,
  3.50665027e-01, -1.67848312e-01, -8.69284400e-03,  3.11125629e-01,
 -1.63492694e-01, -3.19882215e-01,  3.74774284e-02, -8.19474818e-02]
 
 Right Above:
 [4.16214996e-01,  2.49470428e-01,  1.42102492e+00, -1.39006158e+00,
  1.74958514e+00, -1.66241084e+00, -1.85462466e-01,  -2.79987771e-01,
  3.67949199e-01, -1.93394366e-01,  9.05306987e-03,  2.27672675e-01, 
 -1.84830483e-01, -3.59725246e-01,  8.55374129e-02, -9.48781656e-02]

 
Inside:
[ 3.91094291e-01,  2.84323537e-01,  1.43738111e+00, -1.35604046e+00,
  1.80245738e+00, -1.66876315e+00, -4.75405913e-01, -3.40473978e-01,
  3.58889626e-01, -2.03525318e-01, -5.31858159e-04,  2.91391339e-01,
 -2.10746410e-01, -3.30802992e-01,  4.84806241e-02, -1.22947107e-01]
 
 
 
 Shape:
 
 [3.23135421e-01,  3.10463368e-02,  1.39809417e+00, -1.79107645e+00,
  1.48499257e+00, -1.78533070e+00, -4.99316088e-02, -2.77133789e-01,  2.60195320e-01,
 -7.51620145e-02, -1.63141711e-01,  3.09072021e-02, -7.61140795e-02,
 -8.50183576e-01,  7.24356002e-02,  4.08848839e-02]
 
 Init Shape:
 
 [0.38719768,  0.2799246,   1.3039659,  -1.94091413,  1.7186948,  -1.89383479,
 -0.41323167,  -0.35821855,  0.3139938,  -0.1234674,  -0.14442993,
  0.11681355, -0.1095298,  -0.82973836, -0.07711106, -0.02973255]

"""





"""
Init Pos
 [0.47856077 -0.07918085  1.25008667 -0.94040375  1.70405949 -1.88341541
  1.07703224  0.          0.          0.          0.          0.
  0.          0.         -0.14661003  0.38439563  0.09342872 -0.06981191
  0.35207254  0.10436448 -0.16461223  0.33383388  0.16475069]
  
Mid Point
[ 3.10426153e-01 -6.64916096e-02  1.52942478e+00 -1.71130779e+00
  1.40927073e+00 -1.60739482e+00 -1.42003170e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  1.75169657e-09  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -4.28747575e-01  9.45141253e-02
  6.92418854e-03 -2.56404145e-01  1.63419403e-01  9.74860450e-03
 -3.22991869e-01  2.26133317e-02  7.45412423e-02]

Goal
[ 2.92518323e-01  5.10953336e-02  1.72634051e+00 -1.67772098e+00
  1.51067668e+00 -1.44950095e+00 -1.41407098e+00  0.00000000e+00
  0.00000000e+00 -1.86963334e-09  0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00 -4.33270007e-01  9.18991318e-02
 -9.43792175e-02 -2.61428022e-01  1.63224017e-01 -9.55503864e-02
 -3.26072146e-01  2.12508344e-02 -3.00473105e-02]


"""