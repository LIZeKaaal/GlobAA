# Reused from lsqr.py in yangliu-op/AndersonAcceleration:
# https://github.com/yangliu-op/AndersonAcceleration
# Upstream revision: 2eb2947a172d5fd28e1452447cfac30c10acf30b
# Upstream project license: Apache License, Version 2.0.
# License text: https://www.apache.org/licenses/LICENSE-2.0
# Original author attribution and references are retained below.
#
# Integrated into GlobAA by Lizekai, 2026.
# Only attribution comments were added; the numerical implementation is
# unchanged from the upstream revision identified above.
#
# As documented by the upstream author, this implementation is derived from
# scipy.sparse.linalg.lsqr. SciPy-derived portions retain the applicable
# SciPy BSD-3-Clause copyright and license terms:
# https://github.com/scipy/scipy/blob/main/LICENSE.txt
#
"""
Author: Yang Liu @ CUHK(SZ) & UQ



This is a pytorch version & modification of scipy.sparse.linalg.lsqr
To obtain a Moore-Penrose pseudo-inverse solution of x = A^{\dagger} b
where A is a non-squared real matrix in real^{n x d}, b real^{n}

Reference:

C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
equations and sparse least squares, TOMS 8(1), 43--71 (1982).

C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
equations and least-squares problems, TOMS 8(2), 195--209 (1982).


"""

import torch


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    """
    if b == 0:
        return torch.sign(a), 0, torch.abs(a)
    elif a == 0:
        return 0, torch.sign(b), torch.abs(b)
    elif torch.abs(b) > torch.abs(a):
        tau = a / b
        s = torch.sign(b) / torch.sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = torch.sign(a) / torch.sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsqr(A, b, iter_lim=None, eps=1E-16,
         reorth=True):
    """
    Pytorch version & modification of scipy.sparse.linalg.lsqr
    to obtain x = A^{\dagger} b in min || A x - b ||
    
    where A is a non-squared real matrix in real^{n x d}, b real^{n}
    """
    dtype = b.dtype
    device = b.device
    n, d = A.shape
    if iter_lim is None:
        iter_lim = min(n ,d)

    itn = 0
    istop = 0
    anorm = 0
    acond = 0
    rho = 1

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b - A*x \in \real^n,  alfa*v = A'*u in \real^d.
    u = b
    bnorm = b.norm()
    x = torch.zeros(d, dtype=dtype, device=device)
    beta = bnorm.clone()

    if beta > 0:
        u = (1/beta) * u
        v = Ax(A.T, u)
        alfa = v.norm()
    else:
        v = x.clone()
        alfa = 0
    # Abnorm = alfa*beta

    if alfa > 0:
        v = (1/alfa) * v
    w = v.clone()

    if reorth:
        U = u.reshape(-1,1)
        V = v.reshape(-1,1)
    rhobar = alfa
    phibar = beta
    
    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        return x, istop, itn, anorm, acond, arnorm

    # Main iteration loop.
    while itn <= iter_lim + 1:
        itn = itn + 1
        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alfa, v. These satisfy the relations
        #     beta*u  =  a*v   -  alfa*u,
        #     alfa*v  =  A'*u  -  beta*v.
        ul = u
        u = Ax(A, v) - alfa * ul
        beta = u.norm()
        if beta > 0:
            u = (1/beta) * u
            if reorth:
                if itn == 1:
                    u = u - (ul @ u)*ul
                else:
                    u = u - U @ (U.T @ u)
                U = torch.cat((U, u.reshape(-1, 1)), axis=1)
            anorm = torch.sqrt(anorm**2 + alfa**2 + beta**2)
            vl = v
            v = Ax(A.T, u) - beta * vl
            alfa = v.norm()
            if alfa > 0:
                v = (1 / alfa) * v
            if reorth:
                if itn == 1:
                    v = v - (vl @ v)*vl
                else:
                    v = v - V @ (V.T @ v)
                V = torch.cat((V, v.reshape(-1, 1)), axis=1)
        
        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        # cs1 = 1 and sn1 = 0
        rhobar1 = rhobar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        rhol = rho
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        
        xl = x
        x = x + t1 * w
        w = v + t2 * w
        arnorm = alfa * torch.abs(tau)

        if arnorm < eps: ## terminate if || A^T r || = 0
            istop = 1
        if rho < 1E-5 or (rho/rhol < 1E-4 and itn != 1): 
            ## problem is very ill-conditioned, return the previous iteration
            istop = 2    
            x = xl

        if istop != 0:
            break
        
    return x, istop, itn, anorm, acond, arnorm


def Ax(A, x):
    if callable(A):
        Ax = A(x)
    else:
        Ax = torch.mv(A, x)
    return Ax
