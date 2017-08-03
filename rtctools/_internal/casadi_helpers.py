from casadi import MX, Function, jacobian, vertcat, reshape, mtimes
import numpy as np
import logging

logger = logging.getLogger("rtctools")


def array_from_mx(e):
    return np.array([[float(e[i, j]) for j in range(e.size2())] for i in range(e.size1())])


def is_affine(e, v):
    Af = Function('f', [v], [jacobian(e, v)])
    return (Af.sparsity_jac(0, 0).nnz() == 0)


def nullvertcat(*L):
    """
    Like vertcat, but creates an MX with consistent dimensions even if L is empty.
    """
    if len(L) == 0:
        return MX(0, 1)
    else:
        return vertcat(*L)


def reduce_matvec(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product.

    This reduces the number of nodes required to represent the linear operations.
    """
    Af = Function('Af', [MX()], [jacobian(e, v)])
    A = Af(MX())
    return reshape(mtimes(A, v), e.shape)


def substitute_in_external(expr, symbols, values):
    f = Function('f', symbols, expr)
    return f.call(values, True, False)
