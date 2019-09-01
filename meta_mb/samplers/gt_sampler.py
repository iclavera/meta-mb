from meta_mb.utils.serializable import Serializable
import copy
from meta_mb.logger import logger
import time


class GTSampler(Serializable):
    """
    Sampler for ground-truth model

    Args:
        env (meta_mb.meta_envs.base.MetaEnv) : environment object
        policy (meta_mb.policies.bptt_controllers.gt_mpc_controller.*) : policy object
        max_path_length (int): maximum path length
    """
    def __init__(
            self,
            env,
            policy,
            max_path_length,
    ):
        Serializable.quick_init(self, locals())

        self.env = copy.deepcopy(env)
        self.policy = policy
        self.max_path_length = max_path_length

    def update_tasks(self):
        pass

    def obtain_samples(self, log=True, log_prefix='', random=False):
        policy = self.policy
        policy.reset(dones=[True])

        obs = self.env.reset()
        returns = 0
        start_time = time.time()
        for t in range(self.max_path_length):
            policy_time = time.time()
            act, _ = self.policy.get_action(obs)
            policy_time = time.time() - policy_time
            obs, reward, _, _ = self.env.step(act)
            returns += reward

            if log:
                logger.logkv(log_prefix + 'PathLengthSoFar', t)
                logger.logkv(log_prefix + 'Reward', reward)
                logger.logkv(log_prefix + 'SumReward', returns)
                logger.logkv(log_prefix + 'PolicyExecTime', policy_time)
                logger.logkv(log_prefix + 'TotalTime', time.time() - start_time)
                logger.dumpkvs()

        return None

    def __getstate__(self):
        state = dict()
        state['init_args'] = Serializable.__getstate__(self)
        # dumps policy
        state['policy'] = self.policy.__getstate__()
        return state

    def __setstate__(self, state):
        Serializable.__setstate__(self, state['init_args'])
        self.policy = state['policy']
