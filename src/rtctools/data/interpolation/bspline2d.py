from casadi import if_else, logic_and

from .bspline import BSpline


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
        self.__tx = tx
        self.__ty = ty
        self.__w = w
        self.__kx = kx
        self.__ky = ky

    def __call__(self, x, y):
        """
        Evaluate the B-Spline at point (x, y).

        The support of this function is the half-open interval [tx[0], tx[-1]) x [ty[0], ty[-1]).

        :param x: The coordinate of the point at which to evaluate.
        :param y: The ordinate of the point at which to evaluate.

        :returns: The spline evaluated at the given point.
        """
        z = 0.0
        for i in range(len(self.__tx) - self.__kx - 1):
            bx = if_else(logic_and(x >= self.__tx[i], x <= self.__tx[
                         i + self.__kx + 1]), self.basis(self.__tx, x, self.__kx, i), 0.0)
            for j in range(len(self.__ty) - self.__ky - 1):
                by = if_else(logic_and(y >= self.__ty[j], y <= self.__ty[
                             j + self.__ky + 1]), self.basis(self.__ty, y, self.__ky, j), 0.0)
                z += self.__w[i * (len(self.__ty) - self.__ky - 1) + j] * bx * by
        return z
