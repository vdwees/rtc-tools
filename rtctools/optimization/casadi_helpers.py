from casadi import MX, Function, jacobian, vertcat, reshape, mtimes, substitute, interpolant, transpose, repmat
import numpy as np
import logging

logger = logging.getLogger("rtctools")


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


def reduce_matvec_plus_b(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product plus a constant term.

    This reduces the number of nodes required to represent the affine operations.
    """
    bf = Function('bf', [v], [e])
    b = bf(0)[0]
    return reduce_matvec(e, v) + b


def substitute_in_external(expr, symbols, values):
    f = Function('f', symbols, expr)
    return f.call(values, True, False)


def interpolate(ts, xs, t, equidistant, mode=0):
    if False: # TODO mode == 0:
        print(ts)
        print(xs)
        return interpolant('interpolant', 'linear', [ts], xs, {'lookup_mode': 'exact'})(t)
    else:
        if mode == 1:
            xs = xs[:-1] # block-forward
        else:
            xs = xs[1:] # block-backward
        t = MX(t)
        if t.size1() > 1:
            t_ = MX.sym('t')
            xs_ = MX.sym('xs', xs.size1())
            f = Function('interpolant', [t_, xs_], [mtimes(transpose((t_ >= ts[:-1]) * (t_ < ts[1:])), xs_)])
            f = f.map(t.size1(), 'serial')
            return transpose(f(transpose(t), repmat(xs, 1, t.size1())))
        else:
            return mtimes(transpose((t >= ts[:-1]) * (t < ts[1:])), xs)