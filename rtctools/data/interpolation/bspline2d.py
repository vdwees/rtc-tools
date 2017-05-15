# cython: embedsignature=True

from casadi import if_else, logic_and
from bspline import BSpline


class BSpline2D(BSpline):
    """
    Arbitrary order, two-dimensional, non-uniform B-Spline.
    """

    def __init__(self, tx, ty, w, kx=3, ky=3):
        """
        Create a new 2D B-Spline object.

        :param tx: Knot vector in X direction.
        :param ty: Knot vector in Y direction.
        :param w:  Weight vector.
        :param kx: Spline order in X direction.
        :param ky: Spline order in Y direction.
        """

        # Store arguments
        self._tx = tx
        self._ty = ty
        self._w = w
        self._kx = kx
        self._ky = ky

    def __call__(self, x, y):
        """
        Evaluate the B-Spline at point (x, y).

        The support of this function is the half-open interval [tx[0], tx[-1]) x [ty[0], ty[-1]).

        :param x: The coordinate of the point at which to evaluate.
        :param y: The ordinate of the point at which to evaluate.

        :returns: The spline evaluated at the given point.
        """
        z = 0.0
        for i in range(len(self._tx) - self._kx - 1):
            bx = if_else(logic_and(x >= self._tx[i], x <= self._tx[
                         i + self._kx + 1]), self.basis(self._tx, x, self._kx, i), 0.0)
            for j in range(len(self._ty) - self._ky - 1):
                by = if_else(logic_and(y >= self._ty[j], y <= self._ty[
                             j + self._ky + 1]), self.basis(self._ty, y, self._ky, j), 0.0)
                z += self._w[i * (len(self._ty) - self._ky - 1) + j] * bx * by
        return z
