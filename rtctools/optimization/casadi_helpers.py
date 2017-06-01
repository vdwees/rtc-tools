from casadi import MX, Function, jacobian, vertcat, reshape, mtimes, substitute
import numpy as np
import logging

logger = logging.getLogger("rtctools")


def is_affine(e, v):
    f = Function("f", [v], [jacobian(e, v)])
    return (f.sparsity_jac(0, 0).nnz() == 0)


def nullvertcat(L):
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
    Af = Function("Af", [], [jacobian(e, v)])
    A = Af()[0]
    return reshape(mtimes(A, v), e.shape)


def reduce_matvec_plus_b(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product plus a constant term.

    This reduces the number of nodes required to represent the affine operations.
    """
    bf = Function("bf", [v], [e])
    b = bf(0)[0]
    return reduce_matvec(e, v) + b


def is_equal(a, b):
    return np.all([((a_ - b_) == 0) for a_, b_ in zip(a, b)])
