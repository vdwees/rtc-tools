from casadi import MX, Function, jacobian, vertcat, reshape, mtimes, substitute, interpolant
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


def interpolate(ts, xs, t, equidistant, mode=0):
    if mode == 0:
        return interpolant(ts, xs, t, equidistant)
    else:
        if mode == 1:
            xs = xs[:-1] # block-forward
        else:
            xs = xs[1:] # block-backward
        t = MX(t)
        if t.size1() > 1:
            t_ = MX.sym('t')
            xs_ = MX.sym('xs', xs.size1())
            f = Function('interpolant', [t_, xs_], [mul(transpose((t_ >= ts[:-1]) * (t_ < ts[1:])), xs_)])
            f = f.map('interpolant_map', t.size1())
            return transpose(f([transpose(t), repmat(xs, 1, t.size1())])[0])
        else:
            return mul(transpose((t >= ts[:-1]) * (t < ts[1:])), xs)