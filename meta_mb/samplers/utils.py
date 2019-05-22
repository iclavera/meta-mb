import numpy as np
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1, save_video=True,
            video_filename='sim_out.mp4', ignore_done=False, stochastic=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = {}
    env_infos = {}
    images = []

    ''' get wrapped env '''
    wrapped_env = env
    while hasattr(wrapped_env, '_wrapped_env'):
        wrapped_env = wrapped_env._wrapped_env

    frame_skip = wrapped_env.frame_skip if hasattr(wrapped_env, 'frame_skip') else 1
    assert hasattr(wrapped_env, 'dt'), 'environment must have dt attribute that specifies the timestep'
    timestep = wrapped_env.dt

    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        if not stochastic:
            a = agent_info['mean']
        next_o, r, d, env_info = env.step(a[0])
        observations.append(o)
        rewards.append(r)
        actions.append(a)

        for k in agent_info.keys():
            if k not in agent_infos:
                agent_infos[k] = []
            agent_infos[k].append(agent_info[k])

        for k in env_info.keys():
            if k not in env_infos:
                env_infos[k] = []
            env_infos[k].append(env_info[k])
        path_length += 1
        if d and not ignore_done: # and not animated:
            break
        o = next_o
        if animated:
            env.render()
            time.sleep(timestep*frame_skip / speedup)
            if save_video:
                from PIL import Image
                image = env.wrapped_env.wrapped_env.get_viewer().get_image()
                pil_image = Image.frombytes('RGB', (image[1], image[2]), image[0])
                images.append(np.flipud(np.array(pil_image)))

    if animated:
        if save_video:
            import moviepy.editor as mpy
            fps = int(speedup/timestep * frame_skip)
            clip = mpy.ImageSequenceClip(images, fps=fps)
            if video_filename[-3:] == 'gif':
                clip.write_gif(video_filename, fps=fps)
            else:
                clip.write_videofile(video_filename, fps=fps)
        #return

    return dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        agent_infos=agent_infos,
        env_infos=env_infos
        )