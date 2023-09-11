import numpy as np
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class hyperparam:
	""" used for hyperparameter tuning """
	name: str
	val_type_str: str
	low_bound: 'optional, only needed if val_type_str == int or float' = None
	high_bound: 'optional, only needed if val_type_str == int or float' = None
	value_list: 'optional, only needed if val_type_str == categorical' = field(default_factory=list)

	def __post_init__(self):
		if self.val_type_str not in ['categorical', 'int', 'float', 'bool', 'step_schedule']:
			raise Warning("The only values val_type_str can take are 'categorical', 'int', 'float', 'bool'.")			
		if self.val_type_str == 'categorical':
			assert len(self.value_list) > 0, "a categorical hyperparam needs a value_list property"
		if self.val_type_str in ['int', 'float']:
			assert (self.low_bound != None) and (self.high_bound != None), "int and float hyperparams need low_bound and high_bound properties"

	def sample_fn(self):
		""" returns function which samples the hyperparameter.  
		
		note: *args, **kwargs added because the backend sometimes likes to
		pass some arguments (e.g., 'spec') which have no bearing in the sampling...
		"""
		if self.val_type_str == 'categorical':
			return lambda *args, **kwargs: np.random.choice(self.value_list)
		if self.val_type_str == 'int':
			return lambda *args, **kwargs:  np.random.randint(low=self.low_bound, high=self.high_bound+1) # make high_bound *inclusive*
		if self.val_type_str == 'float':
			return lambda *args, **kwargs: self.low_bound + self.high_bound * np.random.rand()
		if self.val_type_str == 'bool':
			return lambda *args, **kwargs: np.random.choice([True, False])
		if self.val_type_str == 'step_schedule':
			pass
			# itertools.pairwise only exists for python 3.11 :(
			#
			# from itertools import pairwise
			# return lambda *args, **kwargs: [
			# 	(0, abs(next_lr-current_lr) * np.random.rand() + min(current_lr,next_lr) ) 
			# 	for current_lr, next_lr in pairwise([0.001, 0.0001, 0.00001])			
			# 	]

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