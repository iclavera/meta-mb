from meta_mb.envs.mujoco.point_pos import PointEnv
from meta_mb.envs.mujoco.inverted_pendulum_env import InvertedPendulumEnv

def make_env(env, render_size=(64, 64), distractor=False, ptsize=2):

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
    elif env == 'cheetah_gym':
        from meta_mb.envs.mb_envs.half_cheetah_dm import HalfCheetahEnv
        raw_env = HalfCheetahEnv()
        max_path_length = 100
    elif env == 'cartpole_balance':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole', 'balance'), render_size=render_size),
                                                 keys=['position', 'velocity']), amount=8)
        max_path_length = 125
    elif env == 'cartpole_balance_distractor':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole_distractor', 'balance_distractor'),
                                                                 render_size=render_size),
                                                 keys=['position', 'velocity']), amount=8)
        max_path_length = 125
    elif env == 'cartpole_swingup':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cartpole', 'swingup'), render_size=render_size),
                                                 keys=['position', 'velocity']), amount=8)
        max_path_length = 125
    elif env == 'reacher_easy':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('reacher', 'easy'), render_size=render_size),
                                                 keys=['position', 'velocity', 'to_target']), amount=4)
        max_path_length = 250
    elif env == 'cheetah_run':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cheetah', 'run'), render_size=render_size),
                                                 keys=['position', 'velocity']), amount=4)
        max_path_length = 250
    elif env == 'cheetah_run_distractor':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('cheetah_distractor', 'run'), render_size=render_size),
                                                 keys=['position', 'velocity']), amount=4)
        max_path_length = 250
    elif env == 'finger_spin':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('finger', 'spin'), render_size=render_size),
                                                 keys=['position', 'velocity', 'touch']), amount=2)
        max_path_length = 500
    elif env == 'walker':
        from dm_control import suite
        from meta_mb.envs.dm_wrapper_env import DeepMindWrapper, ConcatObservation, ActionRepeat
        raw_env = ActionRepeat(ConcatObservation(DeepMindWrapper(suite.load('walker', 'walk'), render_size=render_size),
                                                 keys=['height', 'orientations', 'velocity']), amount=2) #leaving out height
        max_path_length = 125


    else:
        raise NotImplementedError

    return raw_env, max_path_length
