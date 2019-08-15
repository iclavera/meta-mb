import joblib
import tensorflow as tf
import argparse
import numpy as np
import pickle
from meta_mb.samplers.utils import rollout
from meta_mb.envs.blue.real_blue_env import BlueReacherEnv
from meta_mb.envs.blue.real_blue_arm_env import ArmReacherEnv
from meta_mb.envs.blue.full_blue_env import FullBlueEnv
from meta_mb.envs.blue.blue_env import BlueEnv
from meta_mb.envs.blue.mimic_blue_pos_env import MimicBluePosEnv
from meta_mb.envs.blue.mimic_blue_action_env import MimicBlueActEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.meta_envs.rl2_env import rl2env
import time



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str)
    parser.add_argument('--max_path_length', type=int, default=1000,
                        help='Max length of rollout')
    parser.add_argument('--num_rollouts', '-n', type=int, default=10,
                        help='Max length of rollout')
    parser.add_argument('--speedup', type=float, default=1,
                        help='Speedup')
    parser.add_argument('--video_filename', type=str,
                        help='path to the out video file')
    parser.add_argument('--prompt', type=bool, default=False,
                        help='Whether or not to prompt for more sim')
    parser.add_argument('--ignore_done', action='store_true',
                        help='Whether stop animation when environment done or continue anyway')
    parser.add_argument('--stochastic', action='store_true', help='Apply stochastic action instead of deterministic')
    args = parser.parse_args()

    # If the snapshot file use tensorflow, do:
    # import tensorflow as tf
    # with tf.Session():
    #     [rest of the code]

    Environment = ArmReacherEnv
    SimEnv = BlueEnv

    with tf.Session() as sess:
        pkl_path = args.param
        print("Testing policy %s" % pkl_path)
        data = joblib.load(pkl_path)
        policy = data['policy']

        control_freq = 5
        dt = 1.0/control_freq

        env = rl2env(normalize(Environment(side='right', control_freq=control_freq)))
        simenv = SimEnv(arm='right')

        real_rewards = np.array([])
        act_rewards = np.array([])
        pos_rewards = np.array([])
        for i in range(args.num_rollouts):
            path = rollout(env, policy, max_path_length=args.max_path_length, animated=False, speedup=args.speedup,
                           video_filename=args.video_filename, save_video=False, ignore_done=args.ignore_done,
                           stochastic=args.stochastic)

            # Render sim of the task
            simenv.reset()
            for action in path[0]['actions']:
                start_time = time.time()
                simenv.step(action)
                simenv.render()

                end_time = time.time()
                diff = end_time - start_time
                time.sleep(max(0, dt - diff))

            print('rewards:', np.mean(path[0]['rewards']))
            print('actions:', path[0]['actions'])
            print('observations:', path[0]['observations'])

            pickle.dump(path[0]['actions'], open('blue_actions.pkl', 'wb'))
        #print("Real Reward Avg")
        #print(np.mean(real_rewards))
        #print("Act Reward Avg")
        #print(np.mean(act_rewards))
        #print("Pos Reward Avg")
        #print(np.mean(pos_rewards))
