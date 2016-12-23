from casadi import MX, MXFunction, jacobian, vertcat, reshape, mul, IMatrix, sumCols
import logging

logger = logging.getLogger("rtctools")



# Borrowed from
# https://gist.github.com/jgillis/5aebf6b09ada29355418783e8f60e8ef
def classify_linear(e, v):
    """
    Takes vector expression e, and symbolic primitives v
    Returns classification vector
    For each element in e, determines if:
      - element is nonlinear in v               (2)
      - element is    linear in v               (1)
      - element does not depend on v at all     (0)

    This method can be sped up a lot with JacSparsityTraits::sp
    """

    f = MXFunction("f", [v], [jacobian(e, v)])
    ret = ((sumCols(IMatrix(f.outputSparsity(0), 1))
            == 0) == 0).nonzeros()
    pattern = IMatrix(f.jacSparsity(
        0, 0), 1).reshape((e.shape[0], -1))
    for k in sumCols(pattern).row():
        ret[k] = 2
    return ret


def depends_on(e, v):
    """
    Return True if e depends on v.
    """
    classification = classify_linear(e, v)
    if sum(classification) > 0:
        return True
    else:
        return False


def nullvertcat(L):
    """
    Like vertcat, but creates an MX with consistent dimensions even if L is empty.
    """
    if len(L) == 0:
        return MX(0, 1)
    else:
        return vertcat(L)


def reduce_matvec(e, p):
    """
    Reduces the MX graph e of linear operations on p into a matrix-vector product.

    This reduces the number of nodes required to represent the linear operations.
    """
    temp = MXFunction("temp", [], [jacobian(e, p)])
    J = temp([])[0]
    return reshape(mul(J, p), e.shape)