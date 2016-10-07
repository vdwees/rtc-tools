import matplotlib
matplotlib.use('Agg')
import pylab

import unittest
import numpy as np

class TestCase(unittest.TestCase):
	def assertAlmostEqual(self, a, b, tol):
		failure = (np.abs(a - b) >= tol)
		if hasattr(failure, '__iter__'):
			failure = failure.any()
		if failure:
			raise AssertionError('\n'.join(['abs(a - b) >= tol.', repr(a), repr(b)]))

	def assertAlmostLessThan(self, a, b, tol):
		failure = (a >= b + tol)
		if hasattr(failure, '__iter__'):
			failure = failure.any()
		if failure:
			raise AssertionError('\n'.join(['a >= b + tol', repr(a), repr(b)]))

	def assertAlmostGreaterThan(self, a, b, tol):
		failure = (a <= b - tol)
		if hasattr(failure, '__iter__'):
			failure = failure.any()
		if failure:
			raise AssertionError('\n'.join(['a <= b - tol', repr(a), repr(b)]))