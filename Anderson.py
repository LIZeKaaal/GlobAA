# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Based on Anderson.py from yangliu-op/AndersonAcceleration:
# https://github.com/yangliu-op/AndersonAcceleration
# Upstream revision: 2eb2947a172d5fd28e1452447cfac30c10acf30b
# Upstream license: Apache License, Version 2.0.
# License text: https://www.apache.org/licenses/LICENSE-2.0
# Original author attribution is retained below.
#
# Copyright (c) 2026 Lizekai (modifications and additions only).
# Modifications and additions are licensed under Apache-2.0.
# Modified for GlobAA by Lizekai:
# - Added FAA coefficient filtering and retained-history management.
# - Added effective-memory diagnostics and related parameter validation.
# - Extended reset behavior to clear FAA history.
# The original AA computation remains the basis of the pure-AA branch.
#
"""
Anderson mixing with limited history and optional FAA coefficient filtering.

The standard coefficient method supports unregularized and regularized solves.
Residual-difference columns are scaled before the coefficient solve. The
effective-memory-one branch uses the unregularized scalar formula; eta is
applied when the effective memory is greater than one. FAA instead filters
the retained history using the angle and conditioning parameters.

Original implementation:
Created on Thu Nov  4 12:47:45 2021

@author: Liu Yang
"""

import torch
from time import time
from lsqr import lsqr
import numpy as np
import math
# from scipy.sparse.linalg import lsqr
from scipy.sparse import csc_matrix
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


VALID_COEFFICIENT_METHODS = ("pure", "faa")


def safe_power(base, power):
    if power == 0:
        return 1.0
    if base == 0.0:
        return 0.0
    log_value = power * math.log(abs(base))
    if log_value > 700.0:
        return math.inf
    return math.exp(log_value)


def safe_power_ratio(numerator_base, denominator_base, power):
    if power == 0:
        return 1.0
    return safe_power(numerator_base / denominator_base, power)


def inverse_r_column_bounds(norm2, cs):
    ct = math.sqrt(max(0.0, 1.0 - cs * cs))
    bounds = [0.0] * len(norm2)
    bounds[0] = 1.0 / norm2[0]
    if len(norm2) == 1:
        return bounds
    cs2 = cs * cs
    for j in range(1, len(norm2)):
        total = (
            safe_power_ratio(ct + cs, cs, 2 * (j - 1))
            * ct * ct / norm2[0]
        )
        for i in range(1, j):
            total += (
                ct * ct
                * safe_power(ct + cs, 2 * (j - i - 1))
                / (norm2[i] * safe_power(cs, 2 * (j - i)))
            )
        total += 1.0 / norm2[j]
        bounds[j] = total / cs2
    return bounds


def faa_coefficients(f_cols, residual, cs, kappa_bar):
    width = f_cols.shape[1]
    coefficients = torch.zeros(
        width, dtype=f_cols.dtype, device=f_cols.device)
    active = torch.zeros(width, dtype=torch.bool, device=f_cols.device)
    if width == 0:
        return coefficients, active

    norm2 = torch.sum(f_cols * f_cols, dim=0)
    tiny = torch.finfo(f_cols.dtype).tiny
    norm2_values = [
        max(float(value.detach().cpu()), tiny) for value in norm2
    ]
    bounds = inverse_r_column_bounds(norm2_values, cs)
    limit = kappa_bar * kappa_bar
    prefix_norm2 = 0.0
    prefix_bounds = 0.0
    admissible_prefix = []
    for value, bound in zip(norm2_values, bounds):
        prefix_norm2 += value
        prefix_bounds += bound
        admissible_prefix.append(prefix_norm2 * prefix_bounds <= limit)

    keep_count = 1
    for candidate in range(width, 0, -1):
        if admissible_prefix[candidate - 1]:
            keep_count = candidate
            break

    prefix_indices = torch.arange(
        keep_count, dtype=torch.long, device=f_cols.device)
    prefix = f_cols[:, prefix_indices]
    q_mat, r_mat = torch.linalg.qr(prefix, mode="reduced")
    keep_mask = torch.ones(
        keep_count, dtype=torch.bool, device=f_cols.device)
    if keep_count > 1:
        norms = torch.linalg.norm(prefix, dim=0)
        diagonal = torch.abs(torch.diagonal(r_mat))
        keep_mask[1:] = (
            (norms[1:] > 0.0)
            & (diagonal[1:] / torch.clamp(norms[1:], min=tiny) >= cs)
        )

    selected_indices = prefix_indices[keep_mask]
    if selected_indices.numel() == 0:
        return coefficients, active

    selected = f_cols[:, selected_indices]
    q_mat, r_mat = torch.linalg.qr(selected, mode="reduced")
    rhs = -(q_mat.T @ residual)
    try:
        selected_coefficients = torch.linalg.solve(r_mat, rhs)
    except RuntimeError:
        selected_coefficients = torch.linalg.lstsq(
            r_mat, rhs.unsqueeze(1)).solution[:, 0]

    coefficients[selected_indices] = selected_coefficients
    active[selected_indices] = True
    return coefficients, active


class Anderson:
    def __init__(self, x0, num, coefficient_method="pure", cs=None,
                 kappa_bar=None):
        if coefficient_method not in VALID_COEFFICIENT_METHODS:
            raise ValueError(
                f"Unknown coefficient method: {coefficient_method}.")
        if coefficient_method == "faa":
            if cs is None or not 0.0 < cs < 1.0:
                raise ValueError("FAA requires cs in (0, 1).")
            if kappa_bar is None or kappa_bar < 1.0:
                raise ValueError("FAA requires kappa_bar >= 1.")
        self.dType = x0.dtype
        self.device = x0.device
        self.mk = num
        self.dim = len(x0)
        self.coefficient_method = coefficient_method
        self.cs = cs
        self.kappa_bar = kappa_bar
        self.current_F_ = x0.clone()
        self.prev_dG_ = torch.zeros(self.dim, num, dtype=self.dType, device=self.device)
        self.prev_dF_ = torch.zeros(self.dim, num, dtype=self.dType, device=self.device)
        self.theta_ = torch.zeros(num, dtype=self.dType, device=self.device)
        self.dF_scale_ = torch.zeros(num, dtype=self.dType, device=self.device)
        self.dG_scale_ = torch.zeros(num, dtype=self.dType, device=self.device)
        self.M_ = torch.zeros(num, num, dtype=self.dType, device=self.device)  # num*num array
        self.current_u_ = x0.clone()
        self.iter_ = 0
        self.col_idx_ = -1
        self.timec = 0
        self.cond = 1
        self.condnum = 1
        self.effective_m = 0
        self.faa_dF_history = torch.empty(
            self.dim, 0, dtype=self.dType, device=self.device)
        self.faa_dG_history = torch.empty(
            self.dim, 0, dtype=self.dType, device=self.device)
        # self.prtgamma = False

    def compute(self, g, eta=0,cond=False):
        if self.coefficient_method != "pure" and eta != 0:
            raise ValueError("eta is supported only by the pure AA solver.")
        G = g.clone()
        self.current_F_ = g - self.current_u_
        self.effective_m = 0
#        print(self.current_u_)
        if self.iter_ == 0:
            # self.prev_dF_[0, :] = -self.current_F_
            # self.prev_dG_[0, :] = - G
            self.current_u_ = G.clone()

        else:
            self.prev_dF_[:, self.col_idx_] += self.current_F_
            self.prev_dG_[:, self.col_idx_] += G
            m_k = min(self.iter_, self.mk)

            if self.coefficient_method == "faa":
                new_dF = self.prev_dF_[:, self.col_idx_].clone()
                new_dG = self.prev_dG_[:, self.col_idx_].clone()
                self._compute_faa_update(G, new_dF, new_dG)
            else:
                eps = 1e-10
                norm = self.prev_dF_[:, self.col_idx_].norm()
#            print(self.prev_dF_)
                scale = max(norm, eps)
                self.dF_scale_[self.col_idx_] = scale
                self.prev_dF_[:, self.col_idx_] /= scale

                if eta == 0:
                    if m_k == 1:
                        self.theta_[0] = 0
                        dF_norm = torch.linalg.norm(self.prev_dF_[:,self.col_idx_])
                        self.theta_[0] = - torch.dot(self.prev_dF_[:,self.col_idx_], self.current_F_[:])/(dF_norm**2)
                        self.cond = 1
                    else:
                        result = lsqr(self.prev_dF_[:,0:m_k],  -self.current_F_)
                    # A = csc_matrix(self.prev_dF_[:,0:m_k].cpu())
                    # result = lsqr(A,  -self.current_F_.cpu())
                    # self.prtgamma = False
                        t_theta = result[0]
                    # error = (self.prev_dF_[:, 0:m_k].T@(self.prev_dF_[:,0:m_k]@ t_theta+self.current_F_)).norm()/(self.current_F_).norm()
                    # Q,R=torch.linalg.qr(self.prev_dF_[:,0:m_k])
                    # print(torch.pinverse(self.prev_dF_[:,0:m_k]))
                    # Qb = -(Q.T @ self.current_F_)
                    # t_theta1 = torch.linalg.solve(R,Qb)
                    
                    # t_theta = torch.linalg.lstsq(self.prev_dF_[:, 0:m_k], -self.current_F_)[0]
                    # error = torch.linalg.norm(self.prev_dF_[:, 0:m_k].T @ (self.prev_dF_[:, 0:m_k] @ t_theta + self.current_F_)) / torch.linalg.norm(self.current_F_)
    
                    # b = torch.mv(self.prev_dF_[:, 0:m_k].T, -self.current_F_)
                    # t_theta = -torch.pinverse(self.prev_dF_[:,0:m_k]) @ self.current_F_
                    # error = torch.linalg.norm(self.prev_dF_[:, 0:m_k].T @ (self.prev_dF_[:, 0:m_k] @ t_theta + self.current_F_)) / torch.linalg.norm(self.current_F_)
                    # if (error1<error2) & (error1<error3):
                    #      self.theta_[0:m_k]=t_theta1
                    #      self.error = error1
                    # elif error2<error3:
                    #      self.theta_[0:m_k]=t_theta2
                    #      self.error = error2
                    # else:
                    #      self.theta_[0:m_k]=t_theta3
                    #      self.error = error3
                        self.theta_[0:m_k] = t_theta
                        if cond:
                            t1 = time()
                            self.cond = torch.linalg.cond(self.prev_dF_[:,0:m_k])
                        # MM = self.prev_dF_[:,0:m_k].T @ self.prev_dF_[:,0:m_k]
                        # eigenvalue = torch.linalg.eigh(MM.T @ MM)[0]
                        # self.cond = torch.sqrt(max(abs(eigenvalue))/min(abs(eigenvalue)))
                        # self.cond = np.linalg.cond(self.prev_dF_[:,0:m_k].cpu().numpy())
                            self.timec = time() - t1
                else:
                    if m_k == 1:
                        dF_norm = torch.linalg.norm(self.prev_dF_[:, self.col_idx_])
                        self.M_[0, 0] = dF_norm**2
                    # coef = self.M_[0, 0] + eta * (dF_norm ** 2 + (self.prev_dF_[self.col_idx_,:]-self.prev_dG_[self.col_idx_,:]/scale).norm()**2)
                        self.cond_num = 1
                        self.theta_[0] = -torch.dot(self.prev_dF_[:, self.col_idx_], self.current_F_[:])/dF_norm**2
                    else:
                        new_inner_prod = torch.mv(self.prev_dF_[:,0:m_k].T, self.prev_dF_[:,self.col_idx_])
                        self.M_[self.col_idx_, 0:m_k] = new_inner_prod
                        self.M_[0:m_k, self.col_idx_] = new_inner_prod
                    # tt = self.prev_dF_[:,0:m_k]
                    # debug = self.M_[0:m_k,0:m_k] -tt.T @ tt
                        b = -torch.mv(self.prev_dF_[:, 0:m_k].T, self.current_F_)
                        self.theta_[0:m_k] = torch.pinverse(self.M_[0:m_k, 0:m_k] + eta * (
                                torch.norm(self.prev_dF_[:, 0:m_k],'fro')**2+torch.norm(
                                        self.prev_dF_[:, 0:m_k]-self.prev_dG_[:, 0:m_k]/self.dF_scale_[None, 
                                                0:m_k],'fro')**2 )*torch.eye(m_k, device=g.device, dtype=torch.float64)) @ b
                    # self.error = torch.linalg.norm(self.prev_dF_[:, 0:m_k].T @ (
                    #             self.prev_dF_[:, 0:m_k] @ self.theta_[0:m_k] + self.current_F_)) / torch.linalg.norm(
                        # self.current_F_)
                # self.condnum = result[5]
                # self.error = error
                # if self.error>1e-5:
                #     print('huge error=',self.error,'iter=',self.iter_)
                #     self.prtgamma = True
                # self.theta_[0:m_k] = torch.linalg.lstsq(self.prev_dF_[:,0:m_k],-self.current_F_)[0]
                #self.theta_[0:m_k] = torch.linalg.lstsq(self.M_[0:m_k, 0:m_k], b)[0]
                #self.theta_[0:m_k] = torch.pinverse(self.M_[0:m_k, 0:m_k]) @ b
                v = self.theta_[0:m_k] / self.dF_scale_[0:m_k]
#            print(self.dF_scale_[0:m_k], v)
                self.current_u_ = G + torch.mv(self.prev_dG_[:, 0:m_k], v)
                self.effective_m = m_k
        self.col_idx_ = (self.col_idx_ + 1) % self.mk
        self.prev_dF_[:, self.col_idx_] = -self.current_F_.clone()
        self.prev_dG_[:, self.col_idx_] = -G.clone()
        self.iter_ += 1

        return self.current_u_.clone()

    def _chronological_indices(self, m_k):
        return [
            (self.col_idx_ - offset) % self.mk
            for offset in range(m_k)
        ]

    def _compute_faa_update(self, G, new_dF, new_dG):
        zero_column_tol = (
            100.0 * torch.finfo(self.dType).eps
            * max(1.0, float(torch.linalg.norm(self.current_F_))))
        if float(torch.linalg.norm(new_dF)) <= zero_column_tol:
            self.current_u_ = G.clone()
            self._clear_faa_history()
            return

        f_cols = torch.cat((
            new_dF.reshape(-1, 1), self.faa_dF_history), dim=1)[:, :self.mk]
        dG_cols = torch.cat((
            new_dG.reshape(-1, 1), self.faa_dG_history), dim=1)[:, :self.mk]
        coefficients, active = faa_coefficients(
            f_cols, self.current_F_, self.cs, self.kappa_bar)
        self.current_u_ = G + torch.mv(dG_cols, coefficients)
        self.faa_dF_history = f_cols[:, active].clone()
        self.faa_dG_history = dG_cols[:, active].clone()
        self.effective_m = self.faa_dF_history.shape[1]

    def _clear_faa_history(self):
        self.faa_dF_history = torch.empty(
            self.dim, 0, dtype=self.dType, device=self.device)
        self.faa_dG_history = torch.empty(
            self.dim, 0, dtype=self.dType, device=self.device)
        self.effective_m = 0

    def replace(self, x):
        self.current_u_ = x.clone()

    def reset(self, x):
        self.current_u_ = x.clone()
        self.iter_ = 0
        self.col_idx_ = -1
        self._clear_faa_history()
        
        
def main():
    torch.manual_seed(2)
    d = 100
    W = torch.randn(d, d, dtype=torch.float64)

    A = W.T @ W
    b = torch.randn(d, dtype=torch.float64)
    f = lambda x: 0.5*torch.norm(W@x-b)**2
    g = lambda x: W.T@ (W @ x - b)
    # print(f(torch.inverse(A)@b))
    x = torch.zeros(d, dtype=torch.float64)
    L = torch.linalg.norm(A,2)
    m = 40
    maxIter = 1E5
    ng = 1
#    iters = 0
#    while iters <= maxIter:
#        gx = x - g(x)/L
#        xn = acc.compute(gx)
#        iters += 1
#        if f(xn) < f(x):
#            x = xn
#        else:
#            x = gx
#        print('f', f(x))
    iters2 = 0
    acc=Anderson(x,m)
    while iters2 <= maxIter and ng > 1E-10:
        grad = g(x)
        ng = torch.linalg.norm(grad,2)
        gx = x - grad/L
        xn = acc.compute(gx,eta=1e-11,cond=True)
        diff = 0.5*torch.sum(torch.mul(W @ (xn-gx),W @ (gx+xn)-2*b))
        if diff <= 0:
            x = xn
            print('    norm grad=', ng, iters2, acc.cond, acc.condnum)
        else:
            x = gx
            print('    norm grad=', ng)
            print('rej, cond=', acc.cond, 'iter=', acc.iter_)
            acc.reset(x)
            # print('error=',acc.error)
            print('diff=', diff)
            print('val=', f(xn))
        if iters2 % (m) == 0:
            acc.reset(x)

        iters2 = iters2 + 1
        
if __name__ == '__main__':
    main()
