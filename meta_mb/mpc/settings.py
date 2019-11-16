import torch
import multiprocessing as mp

USE_GPU = False
dtype = torch.float32

if USE_GPU and torch.cuda.is_available():
	device = torch.device('cuda')
else:
	device = torch.device('cpu')

num_cpu = min(4, mp.cpu_count())
