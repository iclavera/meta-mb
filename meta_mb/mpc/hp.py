import torch
import torch.nn.functional as F

problem_params = {
	'env_type': 'Hopper-v3', 'T': 10000, 'ep_len': 200,
	'buffer_size': 4000, 'gamma': .99,
	'dir_name': None, 'render_env': False,
	'print_freq': 10, 'img_freq': 10, 'save_freq': 200,
	'print_vars': False, 'do_resets': False,
	'freeze': False
}

env_params = {
	'vel_schedule': [None],
	'rand_vel': False, 'vel_min': -4, 'vel_max': 4,
	'vel_every': 100
}

mpc_params = {
	'mode': 'mppi', 'H': 64, 'num_rollouts': 120, 'mpc_steps': 3,
	'filter_coefs': (1,0.05,0.8,0), 'mppi_temp': 1, 'use_best_plan': True,
	'planner_polyak': 1
}
