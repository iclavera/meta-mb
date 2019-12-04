from meta_mb.baselines.linear_baseline import LinearFeatureBaseline
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from meta_mb.envs.normalized_env import normalize
from meta_mb.algos.ppo import PPO
from meta_mb.trainers.mf_trainer import Trainer
from meta_mb.samplers.meta_samplers import MAMLSampler
from meta_mb.samplers.single_sample_processor import SingleSampleProcessor
from meta_mb.policies.gaussian_mlp_policy import GaussianMLPPolicy
import os
from meta_mb.logger import logger
import json
import numpy as np

maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    baseline = LinearFeatureBaseline()
    env = normalize(HopperRandParamsEnv())
    obs_dim = np.prod(env.observation_space.shape)
    policy = GaussianMLPPolicy(
            name="meta-policy",
            obs_dim=obs_dim,
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
        envs_per_task=5,
    )

    sample_processor = SingleSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
    )

    algo = PPO(
        policy=policy,
        learning_rate=config['learning_rate'],
        max_epochs=config['max_epochs']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
    )
    trainer.train()


if __name__=="__main__":
    idx = np.random.randint(0, 1000)
    data_path = maml_zoo_path + '/data/rl2/test_%d' % idx
    logger.configure(dir=data_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(maml_zoo_path + "/configs/rl2_config.json", 'r'))
    json.dump(config, open(data_path + '/params.json', 'w'))
    main(config)