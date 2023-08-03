""""
A collection of sanity checks, especially to ensure user-defined parts are 
well behaved.

CHECKS:
	params_dyn_fn_compatible: env, fn, params -> (raise exception if params are incompatible with fn)
	fn_purity: fn -> (raise exception if fn changes arg values)
"""

from typing import Callable
from inspect import signature


def fn_purity(fn: Callable, *args, **kwargs):
	""" except if (*args, **kwargs) -> fn(*args, **kwargs) modifies argument values. """
	args_before = args
	kwargs_before = kwargs
	_ = fn(*args, **kwargs)
	args_after = args
	kwargs_after = kwargs

	if (args_before != args_after) | (kwargs_before != kwargs_after):
		raise Warning(
			f"{fn.__name__} should not modify the values of its arguments! \n"
			"Possible solutions: \n"
			"  - [safest] make it a pure function (no internal state) \n"
			"  - [still fine] copy arguments using '{arg}_copy = {arg}.copy()' at the start of fn, "
			"and modify the copied values only."
			)

