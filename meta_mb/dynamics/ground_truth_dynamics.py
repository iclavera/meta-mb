import numpy as np
import time
import tensorflow as tf

from multiprocessing import Pool

class GroundTruthDynamics:
    def __init__(self, envs):
        self.n_processes = len(envs)
        self.envs = envs
        if self.n_processes > 1:
            self.predict_mode = 'parallel'
        else:
            self.predict_mode = 'serial'

    def predict(self, obs, act, **kwargs):
        if self.predict_mode == 'serial':
            return self.serial_predict(obs, act)
        if self.predict_mode == 'parallel':
            return self.parallel_predict(obs, act)

    def parallel_predict(self, obs, act):
        start = time.time()
        assert len(obs) % self.n_processes == 0
        def into_chunk(l, i):
            return l[i * self.n_processes : (i+1) * self.n_processes]

        ret = np.zeros_like(obs)
        pool = Pool(self.n_processes)
        for i in range(len(obs) // self.n_processes):
            res = pool.map(self.f, list(zip(np.arange(self.n_processes), into_chunk(obs, i), into_chunk(act, i))))
            ret[i * self.n_processes : (i+1) * self.n_processes] = res

        print(time.time() - start)
        return ret

    def serial_predict(self, obs, act):
        start = time.time()
        ret = np.zeros_like(obs)

        for i, o in enumerate(obs):
            self.envs[0].reset_from_obs(o)
            ret[i] = self.envs[0].step(act[i])[0]

        print(time.time() - start)
        return ret

    def reset_from_obs(self, observations):
        [env.reset_from_obs(o) for (env, o) in zip(self.envs, observations)]


    def f(self, arguments):
        env_index, o, act = arguments
        env = self.envs[env_index]
        env.reset_from_obs(o)
        ret = env.step(act)[0]
        return ret

