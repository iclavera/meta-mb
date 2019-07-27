from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv

def make_env(env, distractor=False, ptsize=2):

    if env == 'pt':
        if distractor:
            from meta_mb.envs.mujoco.point_pos_distractor import PointEnv

            raw_env = PointEnv()
        else:
            from meta_mb.envs.mujoco.point_pos import PointEnv

            raw_env = PointEnv(ptsize=ptsize)
        max_path_length = 16
    elif env == 'ip':
        raw_env = InvertedPendulumEnv()
        max_path_length = 16
    elif env == 'cartpole_balance':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole', 'balance')),
                                                 keys=['position', 'velocity']), amount=8)
        max_path_length = 125
    elif env == 'cartpole_balance_norepeat':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole', 'balance')),
                                                 keys=['position', 'velocity']), amount=1)
        max_path_length = 1000
    elif env == 'cartpole_swingup':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole', 'swingup')),
                                                 keys=['position', 'velocity']), amount=8)
        max_path_length = 125
    elif env == 'reacher_easy':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('reacher', 'easy')),
                                                 keys=['position', 'velocity', 'to_target']), amount=4)
        max_path_length = 250
    elif env == 'cheetah_run':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cheetah', 'run')),
                                                 keys=['position', 'velocity']), amount=4)
        max_path_length = 250
    else:
        raise NotImplementedError

    return raw_env, max_path_length
