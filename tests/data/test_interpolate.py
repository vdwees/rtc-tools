from unittest import TestCase, expectedFailure

from casadi import SX, Function, linspace

import numpy as np
import rtctools.data.interpolation.bspline1d
import rtctools.data.interpolation.bspline2d

import os
import sys


class TestBSpline1DFit(TestCase):
    # Only tests for defaut case, k = 3

    def setUp(self):
        start = 0
        end = np.pi * 4
        self.x = np.linspace(start, end, 30)
        self.y = np.sin(self.x)
        self.num_test_points = 100
        self.testpoints = linspace(start, end, self.num_test_points)

    def _y_list(self, monotonicity=0, curvature=0):
        y_list = np.empty(self.num_test_points - 1)
        tck = rtctools.data.interpolation.bspline1d.BSpline1D.fit(
            self.x, self.y, monotonicity=monotonicity, curvature=curvature)
        x = SX.sym('x')
        f = Function('f', [x], [rtctools.data.interpolation.bspline1d.BSpline1D(
                *tck)(x)])
        for xi in range(self.num_test_points - 1):
            y_list[xi] = f(self.testpoints[xi])[0]
        return y_list

    def test_monotonicity(self):
        y_list = self._y_list(monotonicity=1)
        self.assertTrue(np.all((y_list[1:] - y_list[:-1]) > 0))
        y_list = self._y_list(monotonicity=-1)
        self.assertTrue(np.all((y_list[1:] - y_list[:-1]) < 0))

    def test_curvature(self):
        y_list = self._y_list(curvature=1)
        y_slope_list = y_list[1:] - y_list[:-1]
        self.assertTrue(np.all((y_slope_list[1:] - y_slope_list[:-1]) > 0))

        y_list = self._y_list(curvature=-1)
        y_slope_list = y_list[1:] - y_list[:-1]
        self.assertTrue(np.all((y_slope_list[1:] - y_slope_list[:-1]) < 0))


# class TestBSpline2D(TestCase):
#     def setUp(self):
#         x = np.linspace(-1.0, 1.0, 10)
#         y = np.linspace(-1.0, 1.0, 10)
#         xx, yy = np.meshgrid(x, y)
#         x = xx.flatten()
#         y = yy.flatten()
#
#         z = (x + y) * np.exp(-(x**2 + y**2))
#         self.tck = scipy.interpolate.bisplrep(x, y, z)
#         self.bspline = rtctools.data.interpolation.bspline2d.BSpline2D(*self.tck)
#
#         points1 = [-1.0, -0.9999, 0.0, 0.5, 0.75, 0.99999]
#         # Test Cartesian product points1 x points1
#         self.times = [(x, y) for x in points1 for y in points1]
#         self.tolerance = 1e-10
#
#     def test_value(self):
#         x = SX.sym('x')
#         y = SX.sym('y')
#         f = SXFunction('f', [x, y], [self.bspline(x, y)])
#
#         for (x, y) in self.times:
#             [z] = f.call([x, y])
#             self.assertAlmostEqual(scipy.interpolate.bisplev(x, y, self.tck), z)
#
#     def test_derivative(self):
#         x = SX.sym('x')
#         y = SX.sym('y')
#         f = SXFunction('f', [x, y], [self.bspline(x, y)])
#
#         derx = SXFunction('derx', [x, y], [f.jac(0, 0)])
#
#         dery = SXFunction('dery', [x, y], [f.jac(1, 0)])
#
#         for (x, y) in self.times:
#             [dzdx] = derx.call([x, y])
#             self.assertAlmostEqual(scipy.interpolate.bisplev(x, y, self.tck, dx=1), dzdx)
#
#             [dzdy] = dery.call([x, y])
#             self.assertAlmostEqual(scipy.interpolate.bisplev(x, y, self.tck, dy=1), dzdy)
