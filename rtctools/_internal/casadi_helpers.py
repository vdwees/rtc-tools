import casadi as ca
import numpy as np
import logging

logger = logging.getLogger("rtctools")


def is_affine(e, v):
    Af = ca.Function('f', [v], [ca.jacobian(e, v)])
    return (Af.sparsity_jac(0, 0).nnz() == 0)


def nullvertcat(*L):
    """
    Like vertcat, but creates an MX with consistent dimensions even if L is empty.
    """
    if len(L) == 0:
        return ca.MX(0, 1)
    else:
        return ca.vertcat(*L)


def reduce_matvec(e, v):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product.

    This reduces the number of nodes required to represent the linear operations.
    """
    Af = ca.Function('Af', [ca.MX()], [ca.jacobian(e, v)])
    A = Af(ca.MX())
    return ca.reshape(ca.mtimes(A, v), e.shape)


def substitute_in_external(expr, symbols, values):
    f = ca.Function('f', symbols, expr)
    return f.call(values, True, False)


def interpolate(ts, xs, t, equidistant, mode=0):
    if False: # TODO mode == 0:
        print(ts)
        print(xs)
        return ca.interpolant('interpolant', 'linear', [ts], xs, {'lookup_mode': 'exact'})(t)
    else:
        if mode == 1:
            xs = xs[:-1] # block-forward
        else:
            xs = xs[1:] # block-backward
        t = ca.MX(t)
        if t.size1() > 1:
            t_ = ca.MX.sym('t')
            xs_ = ca.MX.sym('xs', xs.size1())
            f = ca.Function('interpolant', [t_, xs_], [ca.mtimes(ca.transpose((t_ >= ts[:-1]) * (t_ < ts[1:])), xs_)])
            f = f.map(t.size1(), 'serial')
            return ca.transpose(f(ca.transpose(t), ca.repmat(xs, 1, t.size1())))
        else:
            return ca.mtimes(ca.transpose((t >= ts[:-1]) * (t < ts[1:])), xs)
