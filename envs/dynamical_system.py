import numpy as np
import inspect

from typing import Dict, Union, Callable
from dataclasses import make_dataclass

# in-house imports
from metadata import base_metadata
from checks import fn_purity

# Warning messages:

DYN_FN_STRUC_WARN = """dyn_fn error:

dyn_fn must have the following structure:

args =
 n positional arguments [floats, or similar] (system variables)
 optionally, a dict or base_params_obj object (system parameters, if any)

output = np.ndarray of length n.
"""


class base_params_obj(base_metadata):

	def new_param_vals(self, **paramval_kwargs):
		""" returns dict of possibly modified parameter values """
		if paramval_kwargs == {}:
			warn(f"'ParamObj_3s.new_param_vals()' with no args returns initialization param vals.")
			return self.__dict__.copy()
		else:
			return {**self.__dict__.copy(), **paramval_kwargs} # kwargs overwrite the original dict

	def perturbed_param_vals(self, loc=0, scale=0.1, **paramval_kwargs):
		""" 
		returns dict of perturbed parameters values (with Gaussian noise). 
		- loc, scale = noise mean and standard deviation
		- paramval_kwargs = parameters whose 'pre-noise' value should be different than their initialization value, 
		                    together with the new value.
		"""
		params = self.get_modified_params_dict(**paramval_kwargs)
		return {
			key: value * (1+np.random.normal(loc=loc, scale=scale)) 
			for key, value in params.items()
		}

def make_params_obj(
	params_dict: Dict[str, Union[int, float, np.float32, np.float64]],
	cls_name = 'params_obj',
):
	""" 
	returns a frozen dataclass <cls_name> object with params_dict as attribs 
	and inheriting methods from base_params_obj 
	"""
	return make_dataclass(
		cls_name,
		[
			(key, type(value), value) for key, value in params_dict.items()
		],
		bases = (base_params_obj, )
		)()


class dynamical_system:
	"""
	Encapsulates a dynamical system. This includes the parameters used and the dynamical function.
	"""
	def __init__(
		self, 
		n: int,
		dyn_fn: Callable, 
		dyn_params: Dict[str, Union[int, float, np.float32, np.float64]]= {}, 
		non_stationary: bool = False, 
		non_stationarities: Dict[str, Callable] = {},
		):
		#
		# warnings
		self.dyn_fn_struc_warning = DYN_FN_STRUC_WARN
		#
		self.n = n
		#
		self.dyn_params = make_params_obj(dyn_params)
		self.parametrized = (len(dyn_params) > 0)
		#
		self.standard_state = np.float32(self.n * [0.5]) # for checks
		#
		self._raw_dyn_fn = dyn_fn
		self.dyn_fn = self._make_dyn_fn(dyn_fn)
		#
		self.non_stationary = non_stationary
		self.non_stationarities = non_stationarities
		#
		self.run_checks()

	def dyn_param_vals(self):
		return self.dyn_params.get_vals()

	def _make_dyn_fn(self, fn: Callable):
		""" makes the dynamic function callable in a standardizedx way """
		self.check_n_set_parametrization_format()
		if self.parametrized:
			if self.parametrization_type == 'base_params_obj':
				def d_fn(*args, params): return fn(*args, params)
				return d_fn
			elif self.parametrization_type == 'dict':
				def d_fn(*args, params): return fn(*args, params.param_vals())
				return d_fn
			else:
				raise Warning("Something went wrong loading the dynamical_system.parametrization_type")
		else:
			def d_fn(*args, params): return fn(*args)
			return d_fn

	def check_n_set_parametrization_format(self):
		""" checks whether dyn_fn accepts parameters in one of the allowed formats (if it does at all) """
		self.parametrization_type = ''
		if self.parametrized:
			try_dict = False
			try: 
				dyn_fn_output = self._raw_dyn_fn(
					* self.standard_state, # standardized starting state
					params=self.dyn_params, # try using base_params_obj format
				)
				self.parametrization_type = 'base_params_obj'
			except:
				try_dict = True # fall back to P = dict
			if try_dict:
				try:
					dyn_fn_output = self._raw_dyn_fn(
						* self.standard_state, # standardized starting state
						params=self.dyn_params.param_vals(), # -> dict
					)
					self.parametrization_type = 'dict'
				except:
					raise Warning(self.dyn_fn_struc_warning)
		else:
			try:
				dyn_fn_output = self._raw_dyn_fn(
					* self.standard_state, # standardized starting state
				)
			except:
				raise Warning(self.dyn_fn_struc_warning)

	def run_checks(self):
		""" checks user input to object is mutually compatible. """

		assert isinstance(self.n, int), "n must be an integer."

		if self.non_stationary & (len(self.non_stationarities) == 0):
			raise Warning(
				"Non-stationary dynamical system initialized but no non-stationarities provided.\n"
				"Provide non_stationarities parameter to dynamical_system() with the form: \n"
				"  {param_name: fn = t -> param_value }"
				)

		dyn_fn_output = self.dyn_fn(*self.standard_state, params=self.dyn_params)

		assert isinstance(dyn_fn_output, np.ndarray), self.dyn_fn_struc_warning
		assert len(dyn_fn_output) == self.n, self.dyn_fn_struc_warning


		fn_purity(self.dyn_fn, *self.standard_state, params = self.dyn_params)
		

