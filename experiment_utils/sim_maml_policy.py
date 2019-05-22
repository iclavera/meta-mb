import joblib
import os
import tensorflow as tf
import argparse
from meta_mb.samplers.utils import rollout
from meta_mb.samplers.meta_samplers.maml_sample_processor import MAMLSampleProcessor
from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from meta_mb.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
import json
import numpy as np
import utils
from meta_mb.envs.normalized_env import normalize
from meta_mb.envs.blue.blue_env import BlueReacherEnv
from meta_mb.envs.pr2.pr2_env import PR2ReacherEnv

from meta_mb.logger import logger
from meta_mb.meta_algos.trpo_maml import TRPOMAML




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

    

    with tf.Session() as sess:
        for f in os.listdir(args.param):
            if f.endswith(".pkl"):
                pkl_path = args.param + f
            elif f.endswith(".json"):
                json_path = args.param + f
        
        with open(json_path, 'r') as f:
            kwargs = json.load(f)




        num_inner_grad_steps = kwargs['num_inner_grad_steps']
        baseline = utils.get_class_name(kwargs['baseline'], '$class')
        hidden_n = getattr(tf, kwargs['hidden_nonlinearity']['function'])
        output_n = kwargs['output_nonlinearity']


        # d = kwargs['env']
        # print(d)
        # e = json.loads(kwargs['env']['$class'])
        # print(e)
       
        # hard coded here
        env = normalize(BlueReacherEnv()) # Wrappers?
        # hard coded here

        sample_processor = MAMLSampleProcessor(
            baseline=baseline(),
            discount=kwargs['discount'],
            gae_lambda=kwargs['gae_lambda'],
            normalize_adv=kwargs['normalize_adv'],
            positive_adv=kwargs['positive_adv'],
        )

        print("Testing policy %s" % pkl_path)
        data = joblib.load(pkl_path)
        policy = data['policy']
        policy.switch_to_pre_update()
        # get params/set params
        test_policy = MetaGaussianMLPPolicy(
            name="test-meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=1,
            hidden_sizes=kwargs['hidden_sizes'],
            # reuse=True,
            learn_std=kwargs['learn_std'],
            hidden_nonlinearity=hidden_n,
            output_nonlinearity=output_n,
        )


        algo = TRPOMAML(
            policy=test_policy,
            step_size=kwargs['step_size'],
            inner_type=kwargs['inner_type'],
            inner_lr=kwargs['inner_lr'],
            meta_batch_size=1,
            num_inner_grad_steps=kwargs['num_inner_grad_steps'],
        )

        uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
        sess.run(tf.variables_initializer(uninit_vars))

        old_params = policy.get_param_values()
        # test_policy.switch_to_pre_update()
        test_policy.set_params(old_params)
        test_policy.switch_to_pre_update()   




        env = data['env']
        for step in range(num_inner_grad_steps):
            paths={0: []}
            for _ in range(kwargs['rollouts_per_meta_task']):

                path = rollout(env, test_policy, max_path_length=args.max_path_length, animated=False, speedup=args.speedup,
                            video_filename=args.video_filename, save_video=False, ignore_done=args.ignore_done,
                            stochastic=True)
                paths[0].append(path)
                path['dones'] = [False] * len(path['rewards'])  #sarah ignore?? 
                print("sum is ", sum(path['rewards']))
                print(np.mean(path['env_infos']['reward_ctrl']))

            samples_data = sample_processor.process_samples(paths, log='all', log_prefix='Step_%d-' % step)
            logger.log("Computing inner policy updates...")

            old = test_policy.get_param_values()
            dif = {}
            algo._adapt(samples_data)
            now = test_policy.get_param_values()
            for i in old:
                dif[i] = np.linalg.norm(old[i] - now[i])
            print("diffs: ", dif)
        
        path = rollout(env, test_policy, max_path_length=args.max_path_length, animated=True, speedup=args.speedup,
                video_filename=args.video_filename, save_video=False, ignore_done=args.ignore_done,
                stochastic=args.stochastic)
