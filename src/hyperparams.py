import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class hyperparam:
	""" used for hyperparameter tuning """
	name: str
	val_type_str: str
	low_bound: 'optional, only needed if val_type_str != categorical' = None
	high_bound: 'optional, only needed if val_type_str != categorical' = None
	value_list: 'optional, only needed if val_type_str == categorical' = field(default_factory=list)

	def __post_init__(self):
		if val_type_str not in ['categorical', 'int', 'float']:
			raise Warning("The only values val_type_str can take are 'categorical', 'int' and 'float'.")			
		if val_type_str == 'categorical':
			assert len(self.value_list > 0), "a categorical hyperparam needs a value_list property"
		if val_type_str != 'categorical':
			assert (low_bound != None) and (high_bound != None), "int and float hyperparams need low_bound and high_bound properties"

	def sample_fn(self):
		""" returns function which samples the hyperparameter.  
		
		note: *args, **kwargs added because the backend sometimes likes to
		pass some arguments (e.g., 'spec') which have no bearing in the sampling...
		"""
		if val_type_str == 'categorical':
			return lambda *args, **kwargs: np.random.choice(value_list)
		if val_type_str == 'int':
			return lambda *args, **kwargs: np.random.randint(self.low_bound, self.high_bound)
		if val_type_str == 'float':
			return lambda *args, **kwargs: np.random.rand(self.low_bound, self.high_bound)

def make_hyperparams(hp_dicts_list):
	"""
	structure of hp_dicts_list ('list of hyperparam dicts'):

	[
		{	
			'name': hyperparam name (str),
			'val_type_str': type of hyperparam (str: 'int', 'float', 'categorical'),
			'low_bound': [optional: if val_type_str == 'int' or 'float'] lower bound for hyperparameter,
			'high_bound': [optional: if val_type_str == 'int' or 'float']  higher bound for hyperparameter,
			'value_list': [optional: if val_type_str == 'categorical'] list of possible values 
		}
	]
	"""
	return [
		hyperparam(**hp_single_dict) for hp_single_dict in hp_dicts_list
	]