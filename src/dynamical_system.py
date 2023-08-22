import numpy as np
import inspect
import warnings

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

NOT_SUPP_PARAM_WARN_FN = lambda params: f""" params problem:

A non-supported params argument was passed: {params.__repr__}.

Use a base_params_obj or a dict instead.
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
		bases = (base_params_obj, ),
		frozen=True,
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
		self.not_supported_params_wrn = NOT_SUPP_PARAM_WARN_FN
		#
		self.n = n
		#
		self.standard_state = np.float32(self.n * [0.5]) # for checks
		#
		self.non_stationary = non_stationary
		self.non_stationarities = non_stationarities
		#
		self.dyn_params = make_params_obj(dyn_params)
		self.parametrized = (len(dyn_params) > 0)
		#
		self._raw_dyn_fn = dyn_fn
		self.dyn_fn = self._make_dyn_fn(dyn_fn)
		#
		self.run_checks()

	def dyn_param_vals(self, t=0):
		if not self.non_stationary:
			return self.dyn_params.get_vals()
		else: # if it IS non-stationary
			return self.new_param_vals(
				{
				param_name: non_stat_val(t) for param_name, non_stat_val in self.non_stationarities
				}
			)

	def _test(self, fn: Callable):
		if self.parametrized and self.non_stationary:
			def d_fn(*args, t, params): 
				""" non-stationarities contained in parameters """
				return fn(
					*args, 
					params=params.new_param_vals(
						**{
						p_name: nonstationarity(t) 
						for p_name, nonstationarity in self.non_stationarities.items()
						}
					)
				)
		elif self.parametrized and (not self.non_stationary):
			def d_fn(*args, t, params): 
				""" parametrized, stationary """
				return fn(
					*args, params.param_vals()
				)
		elif (not self.parametrized) and (self.non_stationary):
			def d_fn(*args, t, params): 
				""" explicit t dependence in fn """
				warnings.warn("dynamics function has explicit t dependence and no parameters.")
				return fn(*args, t=t, params=params)
		else:
			def d_fn(*args, t, params): 
				""" stationary and non-parametric """
				return fn(*args)
		return d_fn

	def _support_parameters(self, fn: Callable):
		# self.check_n_set_parametrization_format()
		if self.parametrized:
				def d_fn(*args, params, **kwargs): 
					assert isinstance(params, base_params_obj), self.not_supported_params_wrn_fn(params)
					return fn(*args, params=params.param_vals(), **kwargs)
				return d_fn
		else:
			def d_fn(*args, params, **kwargs): return fn(*args, **kwargs)
			return d_fn

	def _support_t_dependence(self, fn: Callable):
		if not self.non_stationary:
			def d_fn(*args, params, t): return fn(*args, params=params)
			return d_fn
		else: # it IS non-stationary
			def d_fn(*args, params, t): 
				return fn(
					*args, 
					params=params.new_param_vals(
						**{
						p_name: nonstationarity(t) 
						for p_name, nonstationarity in self.non_stationarities.items()
						})
					)
			return d_fn

	def _make_dyn_fn(self, fn: Callable):
		""" makes the dynamic function callable in a standardizedx way """
		return self._test(fn)
		# self._support_t_dependence(
		# 	self._support_parameters(fn)
		# )

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
				try_dict = True # fall back to params = dict
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

		# dyn_fn_output = self.dyn_fn(*self.standard_state, t=0, params=self.dyn_params)

		# assert isinstance(dyn_fn_output, np.ndarray), self.dyn_fn_struc_warning
		# assert len(dyn_fn_output) == self.n, self.dyn_fn_struc_warning


		# fn_purity(self.dyn_fn, t=0, *self.standard_state, params = self.dyn_params)
		

