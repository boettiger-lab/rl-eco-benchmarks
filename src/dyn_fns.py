import numpy as np

def twoSp_1(X, Y, params):
	""" a two species model with competition
	
	params = 
		K_x
		K_y
		LV_xy
		r_x
		r_y
	"""
	p = params
	return np.float32([
		X + X * p['r_x'] * (1 - X / p['K_x']) 
			- p['LV_xy'] * X * Y,
		Y + Y * p['r_y'] * (1 - Y / p['K_y']) 
			- p['LV_xy'] * X * Y,
		])

def twoSp_2(X, Y, params):
	""" a two species model with predation
	
	params = 
		c
		d_y
		K_x
		r_x
		r_y

		sigma_x
		sigma_y
	"""
	p = params
	return np.float32([
		X + (
				p['r_x'] * X * (1 - X / p['K_x'])
			- p['r_y'] * Y * (X**2) / (p['c']**2 + X**2)
			+ p['sigma_x'] * X * np.random.normal()
			),
		Y + (
				p['r_y'] * Y * (
					(X**2) / (p['c']**2 + X**2)
				)
			- p['d_y'] * Y
			+ p['sigma_y'] * Y * np.random.normal()
			)
		])

def threeSp_1(X, Y, Z, params):
	""" a three species model with a trophic triangle dynamics. 
	
	params = 
		c
		D
		d_z
		K_x
		LV_xy -> Lotka-Volterra
		r_x
		r_y
		r_z

		sigma_x
		sigma_y
		sigma_z
	"""
	p = params
	return np.float32(
		[ 
			X + (
						p['r_x'] * X * (1 - X / p['K_x'])
						- p['LV_xy'] * X * Y
						- (1 - p['D']) * p['r_z'] * Z * (X**2) / (p['c']**2 + X**2)
						+ p['sigma_x'] * X * np.random.normal()
					),
			Y + (
						p['r_y'] * Y * (1 - Y / p['K_x'])
						- p['LV_xy'] * X * Y
						- (1+p['D']) * p['r_z'] * Z * (Y**2) / (p['c']**2 + Y**2)
						+ p['sigma_y'] * Y * np.random.normal()
					),
			Z + (
						p['r_z'] * Z * (
								(1-p['D']) * (X**2) / (p['c']**2 + X**2)
							+ (1+p['D']) * (Y**2) / (p['c']**2 + Y**2)
						) 
						- p['d_z'] * Z +  p['sigma_z'] * Z  * np.random.normal()
					)
		], 
	)

def threeSp_2(X, Y, Z, params):
	""" a three species model with a linear trophic dynamics X -> Y -> Z 
	
	params = 
		c_x
		c_y
		d_z
		K_x
		r_x
		r_y
		r_z

		sigma_x
		sigma_y
		sigma_z
	"""
	p = params
	return np.float32(
		[ 
			X + (
						p['r_x'] * X * (1 - X / p['K_x'])
						- p['r_y'] * Y * (X**2) / (p['c_x']**2 + X**2)
						+ p['sigma_x'] * X * np.random.normal()
					),
			Y + (
						p['r_y'] * Y * (X**2) / (p['c_x']**2 + X**2)
						- p['r_z'] * Z * (Y**2) / (p['c_y']**2 + Y**2)
						+ p['sigma_y'] * Y * np.random.normal()
				),
			Z + (
						p['r_z'] * Z * (Y**2) / (p['c_y']**2 + Y**2)
						- p['d_z'] * Z 
						+ p['sigma_z'] * Z  * np.random.normal()
					),
		], 
	)

def fourSp_1(X, Y, Z, W, params):
	""" four species model with the following trophic diagram
	Z     W
	|  \\ |
	X~~~~~Y

	where predation happens from top to bottom, and ~~ denotes Lotka-Volterra competition:
	- Z preys on X and Y
	- W preys on Y
	- X and Y compete

	params = 
		beta_z
		beta_w
		LV_xy -> Lotka-Volterra
		c_xz
		c_yz
		c_yw
		d_z
		d_w
		K_x
		K_y
		K_z
		r_x
		r_y
		r_z

		sigma_x
		sigma_y
		sigma_z
		sigma_w
	"""
	p = params
	return np.float32([
			X + (
						p['r_x'] * X * (1 - X / p['K']) # X growth
						- p['LV_xy'] * X * Y # X-Y competition
						- p['beta_z'] * Z * (X**2) / (p['c_xz']**2 + X**2) # Z preying
						+ p['sigma_x'] * X * np.random.normal()
					),
			Y + (
						p['r_y'] * Y * (1 - Y / p['K']) # Y growth
						- p['LV_xy'] * X * Y # X-Y comptetition
						- p['beta_z'] * Z * (Y**2) / (p['c_yz']**2 + Y**2) # Z preying
						- p['beta_w'] * W * (Y**2) / (p['c_yw']**2 + Y**2) # W preying
						+ p['sigma_y'] * Y * np.random.normal()
					),
			Z + (
						p['r_z'] * p['beta_z'] * Z * (
								(X**2) / (p['c_xz']**2 + X**2)
							+ (Y**2) / (p['c_yz']**2 + Y**2)
						) 
						- p['d_z'] * Z +  p['sigma_z'] * Z  * np.random.normal()
					),
			W + (
						p['r_w'] * p['beta_w'] * Z * (
							(Y**2) / (p['c_yw']**2 + Y**2)
						) 
						- p['d_w'] * Z + p['sigma_w'] * Z  * np.random.normal()
					),
		])
