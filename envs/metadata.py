from dataclasses import dataclass, field
import numpy as np
from typing import Optional, Callable, List, Union
from warnings import warn

@dataclass(frozen=True)
class base_metadata:
	def param_vals(self):
		""" returns copy of the object properties """
		return self.__dict__.copy()

@dataclass(frozen = True)
class param_3s(base_metadata):
	#
	# XY:
	r_x: Optional[float] =  np.float32(0.13)
	r_y: Optional[float] = np.float32(0.2)
	K: Optional[float] =  np.float32(1)
	beta: Optional[float] = np.float32(.1)
	v0: Optional[float] = np.float32(0.1)
	D: Optional[float] = np.float32(0.2)
	#
	tau_yx: Optional[float] = np.float32(0)
	tau_xy: Optional[float] = np.float32(0)
	#
	# Z:
	f: Optional[float] =  np.float32(1)
	dH: Optional[float] = np.float32(0.006)
	#
	# stoch
	sigma_x: Optional[float] = np.float32(0.05)
	sigma_y: Optional[float] = np.float32(0.08)
	siga_z: Optional[float] = np.float32(0.05)

	def new_param_vals(self, **paramval_kwargs):
		""" returns dict of possibly modified parameter values """
		if paramval_kwargs == {}:
			warn(f"'ParamObj_3s.new_param_vals()' with no args returns initialization param vals.")
			return self.__dict__.copy()
		else:
			return {**self.__dict__.copy(), **paramval_kwargs} # kwargs overwrite the original dict

	def perturbed_param_vals(self, scale=0.1, loc=0, **paramval_kwargs):
		""" returns dict of perturbed parameters values """
		params = self.get_modified_params_dict(**paramval_kwargs)
		return {
			key: value * (1+np.random.normal(loc=loc, scale=scale)) 
			for key, value in params.items()
		}


@dataclass(frozen=True)
class envMetadata(base_metadata):
	#
	# which env class
	name: Optional[str] = "test_env"
	n_sp: Optional[int] = 3
	n_act: Optional[int] = 3
	_harvested_sp: Optional[Union[List[int], None]] = None 
	#
	# about episodes
	init_pop: np.ndarray = np.float32([0.5, 0.5, 0.5])
	reset_sigma: Optional[float] = 0.01
	tmax: Optional[int] = 1000
	penalty_fn: Optional[Callable] = lambda t: -self.tmax/t
	extinct_thresh: Optional[float] = 0.05
	#
	# about dynamics / control
	var_bound: Optional[float] = 2.
	_costs: Optional[Union[np.ndarray, None]] = None 
	_prices: Optional[Union[np.ndarray, None]] = None

	def __post_init__(self):
		# harvested default:
		if self._harvested_sp is None:
			object.__setattr__(self, 'harvested_sp', [i for i in range(self.n_act)])
		else:
			object.__setattr__(self, 'harvested_sp', self._harvested_sp.copy())

		# economical defaults:
		if self._costs is None:
			object.__setattr__(self, 'costs', np.zeros(self.n_act, dtype=np.float32))
		else:
			object.__setattr__(self, 'costs', self._costs.copy())
		#
		if self._prices is None:
			object.__setattr__(self, 'prices', np.ones(self.n_act, dtype=np.float32))
		else:
			object.__setattr__(self, 'prices', self._prices.copy())

		self.run_checks()


	def run_checks(self):
		""" runs sanity checks on metadata. """
		#
		assert self.n_act <= self.n_sp, "n_act must be at most n_sp"
		#
		assert self.n_act == len(self.harvested_sp), "harvested_sp does not match n_act"
		assert self.n_act == len(self.costs), "costs list does not match n_act"
		assert self.n_act == len(self.prices), "relative_prices list does not match n_act"
		#
		assert self.n_sp == len(self.init_pop), "init pop does not match n_sp"


# @dataclass(frozen=True)
# class envclass_metadata(base_metadata):
# 	#
# 	# which env class
# 	n_sp: Optional[int] = 3
# 	n_act: Optional[int] = 3



