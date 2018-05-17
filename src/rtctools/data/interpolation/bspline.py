from abc import ABCMeta

from casadi import if_else, logic_and


class BSpline(metaclass=ABCMeta):
    """
    B-Spline base class.
    """

    def basis(self, t, x, k, i):
        """
        Evaluate the B-Spline basis function using Cox-de Boor recursion.

        :param x: Point at which to evaluate.
        :param k: Order of the basis function.
        :param i: Knot number.

        :returns: The B-Spline basis function of the given order, at the given knot, evaluated at the given point.
        """
        if k == 0:
            return if_else(logic_and(t[i] <= x, x < t[i + 1]), 1.0, 0.0)
        else:
            if t[i] < t[i + k]:
                a = (x - t[i]) / (t[i + k] - t[i]) * self.basis(t, x, k - 1, i)
            else:
                a = 0.0
            if t[i + 1] < t[i + k + 1]:
                b = (t[i + k + 1] - x) / (t[i + k + 1] - t[i + 1]) * \
                    self.basis(t, x, k - 1, i + 1)
            else:
                b = 0.0
            return a + b
