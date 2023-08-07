import numpy as np
from metadata import ParamObj_3s

def dynamics_s3(X, Y, Z, parameters: ParamObj_3s):
	""" the step function for the discrete natural dynamics of the system. """
	p = parameters
	return np.array(
		[ 
			X + (
						p.r_x * X * (1 - (X + p.tau_xy * Y) / p.K)
						- (1 - p.D) * p.beta * Z * (X**2) / (p.v0**2 + X**2)
						+ p.sigma_x * X * np.random.normal()
					),
			Y + (
						p.r_y * Y * (1 - (Y + p.tau_yx * X ) / p.K)
						- p.D * p.beta * Z * (Y**2) / (p.v0**2 + Y**2)
						+ p.sigma_y * Y * np.random.normal()
					),
			Z + (
						p.f * p.beta * Z * (
							(1-p.D) * (X**2) / (p.v0**2 + X**2)
							+ p.D * (Y**2) / (p.v0**2 + Y**2)
						) 
						- p.dH * Z +  p.sigma_z * Z  * np.random.normal()
					)
		], 
		dtype = np.float32
	)