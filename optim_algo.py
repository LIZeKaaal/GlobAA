# SPDX-License-Identifier: Apache-2.0
# Adapted from optim_algo.py in yangliu-op/AndersonAcceleration:
# https://github.com/yangliu-op/AndersonAcceleration
# Upstream revision: 2eb2947a172d5fd28e1452447cfac30c10acf30b
# Upstream license: Apache License, Version 2.0.
# License text: https://www.apache.org/licenses/LICENSE-2.0
# The original optimization-loop and recording framework is acknowledged.
#
# Copyright (c) 2026 Lizekai (modifications and additions only).
# Modifications and additions are licensed under Apache-2.0.
# Modified for GlobAA by Lizekai:
# - Adapted the framework to fixed-point maps and relative-residual stopping.
# - Changed restart defaults and residual-safeguard behavior.
# - Extended regularization controls, FAA support, and iteration recording.
# - Removed the upstream L-BFGS routine and original global branch.
# - Added GlobalAndersonSolver for the GlobAA algorithm.
#
"""
Fixed-point solvers for the GlobAA numerical experiments.

Pass a PyTorch fixed-point map obj(u)=G(u) and an initial vector x0.
AndersonAcc provides pure AA, residual-safeguarded AA, and filtered AA;
GlobalAndersonSolver implements the nonmonotone GlobAA framework.

Both solvers use the relative residual ||G(u)-u||/||G(x0)-x0|| for stopping,
with a numerical floor on the initial residual norm, and enforce iteration
and map-evaluation budgets. Each solver returns the final iterate and a
record matrix containing one row per recorded iterate, including x0.
""" 

from collections import deque

import torch
from time import time
from Anderson import Anderson
	 
global num_every_print, orcl_every_record
num_every_print = 1
orcl_every_record = 1E6
        
def myPrint(fk, gk_norm, orcl, iters, tmk, alphak=0, iterLS=0, dType=0):
    """
    Print the iteration diagnostics using the inherited column layout.
    """
    if iters%(num_every_print*10) == 0:
        prt1 = '   iters    Time     f          ||g||         Orcl     Direction '
        prt2 = '   alphak    iterLS'
        print(prt1 + prt2)
    
    prt1 = '%8g   %8.2f' % (iters, tmk)
    prt2 = ' %8.2e     %8.2e ' % (fk, gk_norm)
    prt3 = '%8g   %8s' % (orcl, dType)
    prt4 = '%8.2f   %8g' % (alphak, iterLS)
    print(prt1, prt2, prt3, prt4)  
    
def termination(objVal, gradNorm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):  
    """
    Return True when the residual tolerance or either work limit is reached.
    """
    termination = False
    if gradNorm < gradTol or iters >= mainLoopMaxItrs or orcl >= funcEvalMax:
        termination = True
        return termination
    
def recording(matrix, v1, v2, v3, v4, v5, v6=None, v7=None, v8=None, v9=None, dType=None):    
    """
    Append one iteration row, with optional direction and diagnostic fields.

    AndersonAcc supplies relative residual, residual norm, map evaluations,
    elapsed time, and a diagnostic value as the first five entries.
    """
    v = torch.tensor([v1, v2, v3, v4, v5], device = matrix.device)
    if dType is not None: 
        if dType in ('GD', 'Picard', 'KM'):
            v = torch.cat((v, torch.ones_like(v1).reshape(1)))
        else:
            v = torch.cat((v, torch.zeros_like(v1).reshape(1)))
    if v9 is not None:
        vv = torch.tensor([v6, v7, v8, v9], device = matrix.device)
        v = torch.cat((v, vv))
    matrix = torch.cat((matrix, v.reshape(1,-1)), axis=0)  
    return matrix


def orc_call(iterSolver, HProp, iterLS=None):
    if iterLS == None:
        iterLS = 0
    return 2 + 2*iterSolver*HProp + iterLS


def AndersonAcc(obj, x0, m, lamda, L, mainLoopMaxItrs, funcEvalMax, gamma,
                c1, c2, c3, nu, gradTol=1e-10, 
                show=True, arg='residual', record_txt=None, restart=False,
                regularization_eta=None, print_final=True,
                iterate_callback=None, coefficient_method='pure',
                cs=None, kappa_bar=None):
    """Run AA or a Picard iteration for the fixed-point map obj(u)=G(u).

    arg='pure' accepts every AA candidate; arg='residual' applies the residual
    safeguard; arg='GD' applies G directly. FAA requires arg='pure' and
    coefficient_method='faa', with cs and kappa_bar supplied.

    gradTol is a relative-residual tolerance. regularization_eta=None selects
    eta=1e-8 for the standard AA coefficient method; eta=0 disables its
    regularization. The effective-memory-one formula in Anderson.compute
    does not include the regularization term. The inherited positional
    parameters lamda, L, gamma, c1, c2, c3, and nu are retained for call
    compatibility and do not set the fixed-point map or its step size.

    Return (x, record), with record columns:
    [relative residual, residual norm, map evaluations, elapsed time,
     diagnostic, direction]. The diagnostic contains effective memory for
    FAA and the inherited diagnostic slot otherwise. The direction is 0
    for an accepted AA candidate and 1 for a Picard fallback; the legacy
    arg='GD' path also records 0. The initial row uses direction 1.
    iterate_callback(k, x) receives each iterate.
    """
    iters = 0
    orcl = 0
    fixed_point_args = ('residual', 'pure', 'GD')
    if coefficient_method != 'pure' and arg != 'pure':
        raise ValueError(
            "FAA coefficient_method is supported only with arg='pure'.")
    if arg == 'global':
        raise ValueError(
            "AndersonAcc arg='global' has been removed; use "
            "GlobalAndersonSolver for the globalized fixed-point solver.")
    if arg not in fixed_point_args:
        raise ValueError(
            "AndersonAcc arg must be one of 'residual', 'pure', or 'GD'.")

    def evaluate(x):
        Sx = obj(x).detach()
        Fx = Sx - x
        fk = 0.5 * torch.dot(Fx, Fx)
        return fk, -Fx

    # x = copy.deepcopy(x0)  
    x = x0.clone() 
    fk, gk = evaluate(x)
    gk_norm = gk.norm()
    residual_norm = gk_norm
    residual0_norm = torch.clamp(
        residual_norm.detach().clone(), min=torch.finfo(x.dtype).tiny)
    orcl = 1
    tmk = 0
    D = 10**6
    epsilon = 1/D
    if coefficient_method == 'pure':
        eta = epsilon/100 if regularization_eta is None else regularization_eta
    else:
        if regularization_eta is not None and regularization_eta != 0:
            raise ValueError("FAA does not use regularization_eta.")
        eta = 0
    acc=Anderson(
        x, m, coefficient_method=coefficient_method, cs=cs,
        kappa_bar=kappa_bar)
    iters = 0
    flag = 0
    gk_m_norm = gk_norm
    xType = 'Picard' # Initial direction marker; later used for fallback steps.
    
    R = 10
    nAA = 0
    RAA = 0
    acc_cond = 0
    
    g0_norm = gk_norm
    safeg = True
    
    initial_diagnostic = gk_m_norm if coefficient_method == 'pure' else 0
    record = torch.tensor([residual_norm / residual0_norm, residual_norm,
                           orcl, 0, initial_diagnostic, 1],
                          device=x.device).reshape(1,-1)
    if iterate_callback is not None:
        iterate_callback(0, x)
    orcl_sh = orcl_every_record
    while True:
        residual_norm = gk_norm
        relative_residual = residual_norm / residual0_norm
        stop_norm = relative_residual
        print_f = relative_residual
        print_g = residual_norm

        if (show and iters % num_every_print == 0) \
                or (print_final and (orcl >= funcEvalMax or stop_norm < gradTol)):
            myPrint(print_f, print_g, orcl, iters, tmk, dType=xType)
        if termination(print_f, stop_norm, gradTol, iters, mainLoopMaxItrs, orcl, funcEvalMax):
            break
        t0 = time()
        gx = x - gk
#        gxgrad = obj(gx, 'g')
        
        if arg == 'GD':
            x = gx # Picard update.
#            xType = 'GD'
            xType = 'Acc' # for plot use , omit the marks of GD       
        else:
            if arg == 'pure':
                xn = acc.compute(gx, eta)
                x = xn # pure AA
                xType = 'Acc'
                
            if arg == 'residual':
                FDRS = gx
                xDRS = FDRS
                gkDR = x - xDRS
                xn = acc.compute(gx, eta)
                if safeg or (RAA >= R):
                    if gkDR.norm() <= D*g0_norm*(nAA/R + 1)**(-(1+epsilon)):
                        x = xn 
                        nAA = nAA + 1
                        RAA = 1
                        safeg = False
                        xType = 'Acc'
                    else:
                        x = gx
                        xType = 'Picard'
                        acc.replace(x) # 
                        RAA = 0
                        safeg = True
                else:
                    x = xn
                    nAA = nAA + 1
                    RAA = RAA + 1
                    xType = 'Acc'
        if restart:
            if acc.col_idx_ % (m) == m-1:
                acc.reset(x)
    #            print('reset')
                flag = 2
        
        # fkl = fk
        fk, gk = evaluate(x)     
        orcl += 1
        gk_norm = gk.norm()
        t4 = time()
        if flag == 2: # for Accgeneral
            gk_m_norm = gk_norm
            flag = 0
            
        iters += 1  
        tmk += t4-t0
        residual_norm = gk_norm
        diagnostic_value = (
            acc_cond if coefficient_method == 'pure' else acc.effective_m)
        record = recording(record, residual_norm / residual0_norm,
                           residual_norm, orcl, tmk, diagnostic_value,
                           dType=xType)
        if iterate_callback is not None:
            iterate_callback(iters, x)
        if record_txt is not None and orcl >= orcl_sh:
            record_txt('%s_%s' % ('AA', orcl_sh), record)
            orcl_sh = orcl_sh*10
    
    return x, record


class GlobalAndersonSolver:
    """
    Global Anderson(m) solver for a fixed-point map G.

    INPUT:
        obj: fixed-point function G, called as obj(u)
        x0: starting point u_0
        m: Anderson memory
        lambda_: Krasnoselskii-Mann relaxation parameter lambda in (0, 1]
        mainLoopMaxItrs: maximum number of main iterations
        funcEvalMax: maximum number of calls to G
        gamma_k: scalar/list/tensor/callable admissible update parameter in
            (0, 1], or "paper_window" for
            1 - max{1/(k+2), W_k/(2 W_0)}
    residualTol: stopping tolerance for ||F(u)||/||F(u_0)||
    eta: coefficient regularization parameter (0 disables regularization)
    show: print result for every iteration

    OUTPUTS:
        x: final fixed-point iterate
        record: matrix with rows
            [||F||/||F0||, ||F||, oracle-calls, time, gamma_k,
             max-window-||F||, ||F(u_AA)||, m_k, kappa, direction-type]

    direction-type follows the existing numeric recording convention:
        0 means AA step was accepted, 1 means KM step was used.
    """
    def __init__(self, obj, x0, m, lambda_, mainLoopMaxItrs, funcEvalMax, gamma_k,
                 residualTol=1e-10, eta=0, show=True, record_txt=None,
                 print_final=True, iterate_callback=None):
        if m <= 0:
            raise ValueError('m must be a positive integer.')
        if not (0 < lambda_ <= 1):
            raise ValueError('lambda_ must be in (0, 1].')

        self.obj = obj
        self.x = x0.clone()
        self.m = int(m)
        self.lambda_ = lambda_
        self.mainLoopMaxItrs = mainLoopMaxItrs
        self.funcEvalMax = funcEvalMax
        self.gamma_k = gamma_k
        self.residualTol = residualTol
        self.eta = eta
        self.show = show
        self.record_txt = record_txt
        self.print_final = print_final
        self.iterate_callback = iterate_callback

        self.iters = 0
        self.orcl = 0
        self.tmk = 0
        self.kappa = 0
        self.xType = 'KM'
        self.acc = Anderson(self.x, self.m)
        self.F_norm_window = deque(maxlen=self.m + 1)
        self.F0_norm = None
        self.record_rows = []
        self.record = None

    def run(self):
        Gk, Fk, Fk_norm = self._evaluate(self.x)
        self.F0_norm = torch.clamp(Fk_norm.detach().clone(), min=self._tiny())
        self._append_history(Fk_norm)
        max_window_norm = self._max_window_norm(0)
        gamma_value = self._gamma_value(
            self.iters, self.x, Gk, Fk, 0, max_window_norm)
        self._recording(
            Fk_norm, gamma_value, max_window_norm, Fk_norm, 0, self.kappa,
            self.xType)
        self._record_iterate()
        orcl_sh = orcl_every_record

        while True:
            relative_residual = self._relative_residual(Fk_norm)
            if (self.show and self.iters % num_every_print == 0) \
                    or (self.print_final and (
                        self.orcl >= self.funcEvalMax
                        or relative_residual < self.residualTol)):
                myPrint(relative_residual, Fk_norm, self.orcl, self.iters,
                        self.tmk, dType=self.xType)

            if termination(relative_residual, relative_residual, self.residualTol,
                           self.iters, self.mainLoopMaxItrs, self.orcl,
                           self.funcEvalMax):
                break

            t0 = time()
            m_k = min(self.m, self.iters - self.kappa)
            max_window_norm = self._max_window_norm(m_k)
            gamma_value = self._gamma_value(
                self.iters, self.x, Gk, Fk, m_k, max_window_norm)

            u_AA = self.acc.compute(Gk, self.eta)
            G_AA, F_AA, F_AA_norm = self._evaluate(u_AA)

            if F_AA_norm <= gamma_value * max_window_norm:
                x_next = u_AA
                next_G, next_F, next_F_norm = G_AA, F_AA, F_AA_norm
                self.xType = 'AA'
            else:
                x_next = (1 - self.lambda_) * self.x + self.lambda_ * Gk
                self.kappa = self.iters + 1
                self.acc.reset(x_next)
                next_G, next_F, next_F_norm = self._evaluate(x_next)
                self.xType = 'KM'

            self.x = x_next.clone()
            Gk, Fk, Fk_norm = next_G, next_F, next_F_norm
            self.iters += 1
            self.tmk += time() - t0
            self._append_history(Fk_norm)

            self._recording(
                Fk_norm, gamma_value, max_window_norm, F_AA_norm,
                m_k, self.kappa, self.xType)
            self._record_iterate()

            if self.record_txt is not None and self.orcl >= orcl_sh:
                self.record_txt('GlobalAndersonSolver_%s' % orcl_sh,
                                self._record_tensor())
                orcl_sh = orcl_sh * 10

        self.record = self._record_tensor()
        return self.x, self.record

    def _evaluate(self, x):
        Gx = self.obj(x).detach()
        Fx = Gx - x
        self.orcl += 1
        return Gx, Fx, Fx.norm()

    def _append_history(self, Fx_norm):
        self.F_norm_window.append(float(Fx_norm.detach().item()))

    def _record_iterate(self):
        if self.iterate_callback is not None:
            self.iterate_callback(self.iters, self.x)

    def _max_window_norm(self, m_k):
        window = list(self.F_norm_window)[-(m_k + 1):]
        return self._scalar(max(window))

    def _relative_residual(self, F_norm):
        return F_norm / self.F0_norm

    def _tiny(self):
        return torch.finfo(self.x.dtype).tiny

    def _gamma_value(self, k, x, Gk, Fk, m_k, max_window_norm=None):
        if isinstance(self.gamma_k, str):
            if self.gamma_k != "paper_window":
                raise ValueError(
                    'Unknown gamma_k string mode: %s.' % self.gamma_k)
            if max_window_norm is None:
                max_window_norm = self._max_window_norm(m_k)
            inverse_decay = self._scalar(1.0 / (k + 2))
            window_decay = self._scalar(max_window_norm) / (2 * self.F0_norm)
            gamma_tensor = 1 - torch.maximum(inverse_decay, window_decay)
        elif callable(self.gamma_k):
            gamma_value = self.gamma_k(k, x, Gk, Fk, m_k, self.kappa)
            gamma_tensor = torch.as_tensor(gamma_value, dtype=x.dtype,
                                           device=x.device).reshape(())
        elif torch.is_tensor(self.gamma_k) and self.gamma_k.numel() > 1:
            gamma_value = self.gamma_k[min(k, self.gamma_k.numel() - 1)]
            gamma_tensor = torch.as_tensor(gamma_value, dtype=x.dtype,
                                           device=x.device).reshape(())
        elif isinstance(self.gamma_k, (list, tuple)):
            gamma_value = self.gamma_k[min(k, len(self.gamma_k) - 1)]
            gamma_tensor = torch.as_tensor(gamma_value, dtype=x.dtype,
                                           device=x.device).reshape(())
        else:
            gamma_value = self.gamma_k
            gamma_tensor = torch.as_tensor(gamma_value, dtype=x.dtype,
                                           device=x.device).reshape(())
        if (not torch.isfinite(gamma_tensor)) or gamma_tensor <= 0 \
                or gamma_tensor > 1:
            raise ValueError('gamma_k must be in (0, 1].')
        return gamma_tensor

    def _recording(self, F_norm, gamma_value, max_window_norm, F_AA_norm,
                   m_k, kappa, dType):
        dtype_value = 1 if dType == 'KM' else 0
        row = torch.stack((
            self._scalar(self._relative_residual(F_norm)),
            self._scalar(F_norm),
            self._scalar(self.orcl),
            self._scalar(self.tmk),
            self._scalar(gamma_value),
            self._scalar(max_window_norm),
            self._scalar(F_AA_norm),
            self._scalar(m_k),
            self._scalar(kappa),
            self._scalar(dtype_value),
        ))

        self.record_rows.append(row)
        return row.reshape(1, -1)

    def _record_tensor(self):
        if not self.record_rows:
            return torch.empty(0, 10, dtype=self.x.dtype, device=self.x.device)
        return torch.stack(self.record_rows, dim=0)

    def _scalar(self, value):
        return torch.as_tensor(value, dtype=self.x.dtype,
                               device=self.x.device).reshape(())
