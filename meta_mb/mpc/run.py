import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import random
import time
import multiprocessing as mp
import utils
import agents.agents as A
from copy import deepcopy as copy
from settings import USE_GPU, dtype, device, num_cpu
from hp import *
# from envs.particle_env_b import ParticleMazeEnv

agent = A.MPCAgent(problem_params, env_params, mpc_params)
agent.run_lifetime()

