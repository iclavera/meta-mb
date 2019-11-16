import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import sys
import multiprocessing as mp
import pickle

def save_particle_image(env, ensemble, states, R, dir_name, ep_obs, t, back_min=None, back_max=None):
	dim_min, dim_max = -.9, .9
	points_per_dim = 20
	wall_points = 100
	wall_width = .02
	num_rollouts, H, N = states.shape

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)

	particle_pos, goal_pos = env.x, env.g[0]

	x = np.linspace(-1, 1, num=wall_points)
	y = np.linspace(-1, 1, num=wall_points)

	for i in range(wall_points):
		for j in range(wall_points):
			k = env.get_index(np.array([x[i], y[j]]))
			if env.grid[k[0],k[1]]:
				ax.add_artist(plt.Rectangle([x[i]-(wall_width/2),y[j]-(wall_width/2)], width=wall_width, height=wall_width, fill=True, color='black'))

	x = np.linspace(dim_min, dim_max, num=points_per_dim)
	y = np.linspace(dim_min, dim_max, num=points_per_dim)

	xx = np.zeros(points_per_dim**2)
	yy = np.zeros(points_per_dim**2)
	z = np.zeros(points_per_dim**2)

	preds = np.zeros((points_per_dim**2, len(ensemble.vals)))

	for i in range(points_per_dim):
		for j in range(points_per_dim):
			xx[i*points_per_dim+j] = x[i]
			yy[i*points_per_dim+j] = y[j]
			state = np.concatenate([[x[i]], [y[j]], goal_pos])
			z[i*points_per_dim+j] = ensemble.get_value(state)
	
	if back_min is not None and back_max is not None:
		cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter', vmin=back_min, vmax=back_max)
	else:
		cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter')
	plt.colorbar(cb, shrink=0.5)

	xx = np.zeros(num_rollouts*H)
	yy = np.zeros(num_rollouts*H)
	z = np.zeros(num_rollouts*H)
	for i in range(num_rollouts):
		for tt in range(H):
			xx[i*H+tt] = states[i,tt,0]
			yy[i*H+tt] = states[i,tt,1]
			z[i*H+tt] = R[i]

	cb = plt.scatter(xx, yy, c=z, s=.1, cmap='magma')
	plt.colorbar(cb, shrink=0.5)

	for i in range(t-1):
		obs_i = ep_obs[i]
		ax.add_artist(plt.Circle(obs_i[:2], radius=0.005, fill=True, color='red'))

	ax.add_artist(plt.Circle(particle_pos, radius=0.025, fill=True, color='red'))
	ax.add_artist(plt.Circle(goal_pos, radius=0.1, fill=False, color='green'))

	plt.title('Ensemble values on grid')

	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	path_name = dir_name + '/grid_' + str(t) + '.png'
	plt.savefig(path_name)
	print('Saved figure to', path_name)
	plt.clf()
	plt.close()

def save_particle_var_image(env, ensemble, states, R, dir_name, ep_obs, t, back_min=None, back_max=None):
	dim_min, dim_max = -.9, .9
	points_per_dim = 20
	wall_points = 100
	wall_width = .02

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)

	particle_pos, goal_pos = env.x, env.g[0]

	x = np.linspace(-1, 1, num=wall_points)
	y = np.linspace(-1, 1, num=wall_points)

	for i in range(wall_points):
		for j in range(wall_points):
			k = env.get_index(np.array([x[i], y[j]]))
			if env.grid[k[0],k[1]]:
				ax.add_artist(plt.Rectangle([x[i]-(wall_width/2),y[j]-(wall_width/2)], width=wall_width, height=wall_width, fill=True, color='black'))

	x = np.linspace(dim_min, dim_max, num=points_per_dim)
	y = np.linspace(dim_min, dim_max, num=points_per_dim)

	xx = np.zeros(points_per_dim**2)
	yy = np.zeros(points_per_dim**2)
	z = np.zeros(points_per_dim**2)

	preds = np.zeros((points_per_dim**2, len(ensemble.vals)))

	for i in range(points_per_dim):
		for j in range(points_per_dim):
			xx[i*points_per_dim+j] = x[i]
			yy[i*points_per_dim+j] = y[j]
			state = np.concatenate([[x[i]], [y[j]], goal_pos])
			z[i*points_per_dim+j] = ensemble.get_var(state)
	
	if back_min is not None and back_max is not None:
		cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter', vmin=back_min, vmax=back_max)
	else:
		cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter')
	plt.colorbar(cb, shrink=0.5)

	if R is not None:
		num_rollouts, H, N = states.shape
		xx = np.zeros(num_rollouts*H)
		yy = np.zeros(num_rollouts*H)
		z = np.zeros(num_rollouts*H)
		for i in range(num_rollouts):
			for tt in range(H):
				xx[i*H+tt] = states[i,tt,0]
				yy[i*H+tt] = states[i,tt,1]
				z[i*H+tt] = R[i]

		cb = plt.scatter(xx, yy, c=z, s=.1, cmap='magma')
		plt.colorbar(cb, shrink=0.5)

	for i in range(t-1):
		obs_i = ep_obs[i]
		ax.add_artist(plt.Circle(obs_i[:2], radius=0.005, fill=True, color='red'))

	ax.add_artist(plt.Circle(particle_pos, radius=0.025, fill=True, color='black'))
	ax.add_artist(plt.Circle(goal_pos, radius=0.1, fill=False, color='green'))

	plt.title('Ensemble stds on grid')

	plt.xlim(-1, 1)
	plt.ylim(-1, 1)

	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	path_name = dir_name + '/grid_var_' + str(t) + '.png'
	plt.savefig(path_name)
	print('Saved figure to', path_name)
	plt.clf()
	plt.close()

def save_particle_image_pol(env, ensemble, dir_name, ep_obs, t, draw_back=True, inc_name=True, back_min=None, back_max=None):
	dim_min, dim_max = -.9, .9
	points_per_dim = 20
	wall_points = 100
	wall_width = .02

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)

	particle_pos, goal_pos = env.x, env.g[0]

	"""
	WALLS
	"""

	x = np.linspace(-1, 1, num=wall_points)
	y = np.linspace(-1, 1, num=wall_points)

	for i in range(wall_points):
		for j in range(wall_points):
			k = env.get_index(np.array([x[i], y[j]]))
			if env.grid[k[0],k[1]]:
				ax.add_artist(plt.Rectangle([x[i]-(wall_width/2),y[j]-(wall_width/2)], width=wall_width, height=wall_width, fill=True, color='black'))

	"""
	BACKGROUND
	"""

	x = np.linspace(dim_min, dim_max, num=points_per_dim)
	y = np.linspace(dim_min, dim_max, num=points_per_dim)

	xx = np.zeros(points_per_dim**2)
	yy = np.zeros(points_per_dim**2)
	z = np.zeros(points_per_dim**2)

	if draw_back and ensemble is not None:
		preds = np.zeros((points_per_dim**2, len(ensemble.models)))
		for i in range(points_per_dim):
			for j in range(points_per_dim):
				xx[i*points_per_dim+j] = x[i]
				yy[i*points_per_dim+j] = y[j]
				state = np.concatenate([[x[i]], [y[j]], goal_pos])
				preds[i*points_per_dim+j] = ensemble.get_preds_np(state)
				z[i*points_per_dim+j] = np.max(preds[i*points_per_dim+j])
	
		if back_min is not None and back_max is not None:
			cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter', vmin=back_min, vmax=back_max)
		else:
			cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter')
	else:
		cb = plt.scatter(xx, yy, c=z, s=100000, marker='s', cmap='winter')

	plt.colorbar(cb, shrink=0.5)

	"""
	TRAJECTORIES
	"""

	# if R is not None and R is not []:
	# 	num_rollouts, _, N = states.shape
	# 	xx = np.zeros(num_rollouts)#*H)
	# 	yy = np.zeros(num_rollouts)#*H)
	# 	z = np.zeros(num_rollouts)#*H)
	# 	for i in range(num_rollouts):
	# 		for tt in range(H):
	# 			if dones[i,tt]:
	# 				break
	# 		xx[i] = np.clip(states[i,tt,0], -1, 1)
	# 		yy[i] = np.clip(states[i,tt,1], -1, 1)
	# 		z[i] = R[i]
	# 	xx[xx < -1] = None
	# 	yy[yy < -1] = None
	# 	z[xx == None] = None

	# 	cb = plt.scatter(xx, yy, c=z, s=2, cmap='magma')
	# 	plt.colorbar(cb, shrink=0.5)

	"""
	PAST HISTORY
	"""

	l_obs = ep_obs[0]
	for i in range(1, t):
		if i < t:
			obs_i = ep_obs[i]
		else:
			obs_i = env.get_obs()
		if (l_obs[0]-obs_i[0])**2 <= .11**2 and (l_obs[1]-obs_i[1])**2 <= .11**2:
			l = mlines.Line2D([l_obs[0], obs_i[0]], [l_obs[1], obs_i[1]], color='red', alpha=0.1)
			ax.add_line(l)
		if i < t:
			l_obs = obs_i
		# ax.add_artist(plt.Circle(obs_i[:2], radius=0.005, fill=True, color='red'))
	
	"""
	POLICY
	"""

	pol_colors = [
		'white', 'bisque', 'thistle', 'lavender', 'azure', 'lightyellow',
		'honeydew', 'mistyrose', 'gainsboro', 'mintcream', 'lavenderblush',
		'seashell', 'wheat', 'ghostwhite', 'aliceblue', 'cornsilk'
	]

	# if pol is not None and pol is not []:
	# 	num_pol, color_ind = pol.shape[0] // H, 0
	# 	traj_done = False
	# 	orig_obs = l_obs
	# 	for i in range(pol.shape[0]):
	# 		# Different policy ensembles
	# 		if i % H == 0:
	# 			l_obs = orig_obs
	# 			color = pol_colors[color_ind % len(pol_colors)]
	# 			color_ind += 1
	# 			traj_done = False
	# 			i += 1
	# 			continue
	# 		if traj_done:
	# 			continue
	# 		pol_i = pol[i]
	# 		if abs(pol_i[0]-l_obs[0]) > .11 or abs(pol_i[1]-l_obs[1]) > .11:
	# 			traj_done = True
	# 			continue
	# 		l = mlines.Line2D([l_obs[0], pol_i[0]], [l_obs[1], pol_i[1]], color=color)
	# 		ax.add_line(l)
	# 		l_obs = pol_i

	"""
	CURRENT STATE
	"""

	ax.add_artist(plt.Circle(goal_pos, radius=0.15, lw=2, fill=False, color='darkorange', zorder=1000))
	#ax.add_artist(plt.Circle(ep_obs[t-1], radius=0.015, fill=True, color='darkorange', zorder=10000))
	ax.add_artist(plt.Circle(particle_pos, radius=0.025, fill=True, color='darkorange', zorder=100000))
	#l = mlines.Line2D([ep_obs[t-1][0], particle_pos[0]], [ep_obs[t-1][1], particle_pos[1]], color='darkorange', zorder=10000)
	#ax.add_line(l)
	
	plt.title('Ensemble values on grid (time %d)' % t)
	plt.axis('equal')

	plt.xlim(-1, 1)
	plt.ylim(-1, 1)

	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	if inc_name:
		path_name = dir_name + '/grid_' + str(t) + '.png'
	else:
		path_name = dir_name + '/grid.png'
	plt.savefig(path_name)
	print('Saved figure to', path_name)
	plt.clf()
	plt.close()

	# return preds

def save_particle_var_image_pol(env, ensemble, states, R, dones, dir_name, ep_obs, t, pol, H, back_min=None, back_max=None, preds=None):
	dim_min, dim_max = -.9, .9
	points_per_dim = 20
	wall_points = 100
	wall_width = .02

	fig, ax = plt.subplots(figsize=(7, 4))
	ax.set_xlim(-1, 1)
	ax.set_ylim(-1, 1)

	particle_pos, goal_pos = env._get_obs(), env.goal

	"""
	WALLS
	"""

	x = np.linspace(-1, 1, num=wall_points)
	y = np.linspace(-1, 1, num=wall_points)

	for i in range(wall_points):
		for j in range(wall_points):
			k = env.get_index(np.array([x[i], y[j]]))
			if env.grid[k[0],k[1]]:
				ax.add_artist(plt.Rectangle([x[i]-(wall_width/2),y[j]-(wall_width/2)], width=wall_width, height=wall_width, fill=True, color='black'))

	"""
	BACKGROUND
	"""

	x = np.linspace(dim_min, dim_max, num=points_per_dim)
	y = np.linspace(dim_min, dim_max, num=points_per_dim)

	xx = np.zeros(points_per_dim**2)
	yy = np.zeros(points_per_dim**2)
	z = np.zeros(points_per_dim**2)

	for i in range(points_per_dim):
		for j in range(points_per_dim):
			xx[i*points_per_dim+j] = x[i]
			yy[i*points_per_dim+j] = y[j]
			# state = np.concatenate([[x[i]], [y[j]], goal_pos])
			state = np.array([x[i], y[j]])
			z[i*points_per_dim+j] = ensemble.get_var(state, goal_pos)

	if back_min is not None and back_max is not None:
		cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter', vmin=back_min, vmax=back_max)
	else:
		cb = plt.scatter(xx, yy, c=z, s=200, marker='s', cmap='winter')
	plt.colorbar(cb, shrink=0.5)

	"""
	TRAJECTORIES
	"""

	if R is not None:
		num_rollouts, _, N = states.shape
		xx = np.zeros(num_rollouts)#*H)
		yy = np.zeros(num_rollouts)#*H)
		z = np.zeros(num_rollouts)#*H)
		for i in range(num_rollouts):
			for tt in range(H):
				if dones[i,tt]:
					break
			xx[i] = states[i,tt,0]
			yy[i] = states[i,tt,1]
			z[i] = R[i]
		xx[xx < -1] = None
		yy[yy < -1] = None
		z[xx == None] = None

		cb = plt.scatter(xx, yy, c=z, s=2, cmap='magma')
		plt.colorbar(cb, shrink=0.5)

	"""
	PAST HISTORY
	"""

	l_obs = ep_obs[0]
	for i in range(1, t+1):
		if i < t:
			obs_i = ep_obs[i]
		else:
			obs_i = env.get_obs()
		if (l_obs[0]-obs_i[0])**2 <= .11**2 and (l_obs[1]-obs_i[1])**2 <= .11**2:
			l = mlines.Line2D([l_obs[0], obs_i[0]], [l_obs[1], obs_i[1]], color='red', alpha=0.1)
			ax.add_line(l)
		if i < t:
			l_obs = obs_i
		# ax.add_artist(plt.Circle(obs_i[:2], radius=0.005, fill=True, color='red'))
	
	"""
	POLICY
	"""

	pol_colors = [
		'white', 'bisque', 'thistle', 'lavender', 'azure', 'lightyellow', 
		'honeydew', 'mistyrose', 'gainsboro', 'mintcream', 'lavenderblush',
		'seashell', 'wheat', 'ghostwhite', 'aliceblue', 'cornsilk'
	]
	num_pol, color_ind = pol.shape[0] // H, 0
	traj_done = False
	for i in range(pol.shape[0]):
		# Different policy ensembles
		if i % H == 0:
			l_obs = env.get_obs()
			color = pol_colors[color_ind % len(pol_colors)]
			color_ind += 1
			traj_done = False
		if traj_done:
			continue
		pol_i = pol[i]
		if abs(pol_i[0]-l_obs[0]) > .11 or abs(pol_i[1]-l_obs[1]) > .11:
			traj_done = True
			continue
		l = mlines.Line2D([l_obs[0], pol_i[0]], [l_obs[1], pol_i[1]], color=color)
		ax.add_line(l)
		l_obs = pol_i
		
	"""
	CURRENT STATE
	"""

	ax.add_artist(plt.Circle(goal_pos, radius=0.15, lw=2, fill=False, color='darkorange', zorder=1000))
	#ax.add_artist(plt.Circle(ep_obs[t-1], radius=0.015, fill=True, color='darkorange', zorder=10000))
	ax.add_artist(plt.Circle(particle_pos, radius=0.025, fill=True, color='darkorange', zorder=100000))
	#l = mlines.Line2D([ep_obs[t-1][0], particle_pos[0]], [ep_obs[t-1][1], particle_pos[1]], color='darkorange', zorder=10000)
	#ax.add_line(l)
	
	plt.title('Ensemble stds on grid (time %d)' % t)
	plt.axis('equal')

	plt.xlim(-1, 1)
	plt.ylim(-1, 1)

	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	path_name = dir_name + '/grid_var_' + str(t) + '.png'
	plt.savefig(path_name)
	print('Saved figure to', path_name)
	plt.clf()
	plt.close()

	return preds

def save_image_obj_pol(env, ensemble, states, R, dones, dir_name, ep_obs, t, pol, H, back_min=None, back_max=None):
	image_obj = (
		env, ensemble, states, R, dones,
		ep_obs, t, pol, H, back_min, back_max
	)

	if not os.path.isdir(dir_name):
		os.makedirs(dir_name)
	file_name = dir_name + '/imgobj_' + str(t) + '.pkl'
	af = open(file_name, 'wb')
	pickle.dump(image_obj, af)
	af.close()
	print('Saved image object to', file_name)

def gen_image(dir_name, t_i):
	af = open(dir_name + '/image_objs/imgobj_' + str(t_i) + '.pkl', 'rb')
	image_obj = pickle.load(af)
	af.close()
	env, ensemble, states, R, dones, ep_obs, t, pol, H, back_min, back_max = image_obj
	preds = save_particle_image_pol(
		env, ensemble, states, R, dones, dir_name + '/grids', ep_obs, t, pol, H, back_min=back_min, back_max=back_max
	)
	# save_particle_var_image_pol(
	# 	env, ensemble, states, R, dones, dir_name + '/vars', ep_obs, t, pol, H, back_min=back_min, back_max=back_max, preds=preds
	# )

def load_images_obj_pol(dir_name, t_min, t_max, t_step):
	t_i, step, num_cpu = t_min, 0, 8
	if num_cpu > 1:
		pool, jobs = mp.Pool(), []
	while t_i <= t_max:
		if num_cpu == 1:
			gen_image(dir_name, t_i)
		if num_cpu > 1:
			p = mp.Process(target=gen_image, args=(dir_name, t_i))
			jobs.append(p)
		t_i += t_step
		step += 1
		if num_cpu > 1 and step % num_cpu == 0:
			for p in jobs:
				p.start()
			for p in jobs:
				p.join()
			jobs = []
	if num_cpu > 1:
		for p in jobs:
			p.start()
		for p in jobs:
			p.join()

if __name__ == '__main__':
	load_images_obj_pol(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
