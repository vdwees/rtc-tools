from casadi import MX, MXFunction, jacobian, vertcat, reshape, mul, substitute, iszero
import numpy as np
import logging

logger = logging.getLogger("rtctools")


def is_affine(e, v):
    f = MXFunction("f", [v], [jacobian(e, v)])
    return (f.jacSparsity(0, 0).nnz() == 0)


def nullvertcat(L):
    """
    Like vertcat, but creates an MX with consistent dimensions even if L is empty.
    """
    if len(L) == 0:
        return MX(0, 1)
    else:
        return vertcat(L)


def reduce_matvec(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product.

    This reduces the number of nodes required to represent the linear operations.
    """
    Af = MXFunction("Af", [], [jacobian(e, v)])
    A = Af([])[0]
    return reshape(mul(A, v), e.shape)


def reduce_matvec_plus_b(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product plus a constant term.

    This reduces the number of nodes required to represent the affine operations.
    """
    bf = MXFunction("bf", [v], [e])
    b = bf([0])[0]
    return reduce_matvec(e, v) + b


def resolve_interdependencies(e, v, max_recursion_depth=10):
    """
    Replaces occurences of the symbols in v with the expressions of e,
    until all symbols have been resolved or a maximum recursion depth is reached.
    """
    recursion_depth = 0
    while True:
        e_ = substitute(e, v, e)
        if iszero(vertcat(e) - vertcat(e_)):
            return e_
        e = e_
        recursion_depth += 1
        if recursion_depth > max_recursion_depth:
            raise Exception(
                "Interdependency resolution:  Maximum recursion depth exceeded")