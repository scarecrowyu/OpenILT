import os
import sys
sys.path.append(".")

import math
import glob
import time
import random

import numpy as np
import optuna
from scipy.stats import multivariate_normal, norm
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from pyilt.mbopc import mbopcfunc_select
from bench import *

DTYPE = torch.float32
DEVICE = "cpu"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POSTFIX = {"dtype": DTYPE, "device": DEVICE}

class NES(BlackSolver):
    def __init__(self, ranges, nobjs=1,
                 mu=None, amat=None,
                 npop=None, # Note: population = npop * batch
                 eta_mu=1.0, eta_sigma=None, eta_bmat=None,
                 fshape=True, adasam=False, njobs=1):
        super().__init__(ranges, nobjs, "NES")

        self._ndims = len(ranges)
        self._lowers = torch.tensor([x[0] for x in ranges], **POSTFIX)
        self._uppers = torch.tensor([x[1] for x in ranges], **POSTFIX)
        self._mu = 0.5 * (self._lowers + self._uppers) if mu is None else mu
        self._amat = 0.25 * (self._uppers - self._lowers) * torch.eye(self._ndims, **POSTFIX) if amat is None else amat
        self._sigma = torch.abs(torch.det(self._amat))**(1.0/self._ndims)
        self._sigma_old = None
        self._bmat = self._amat*(1.0/self._sigma)
    
        self._npop = int(4 + 3*math.log(self._ndims)) if npop is None else npop
        # self._npop = int(3)
        self._eta_mu = eta_mu
        self._eta_sigma = self._eta_sigma_init = 3*(3+math.log(self._ndims))*(1.0/(5*self._ndims*math.sqrt(self._ndims))) if eta_sigma is None else eta_sigma
        self._eta_bmat = 3*(3+math.log(self._ndims))*(1.0/(5*self._ndims*math.sqrt(self._ndims))) if eta_bmat is None else eta_bmat
        self._fshape = fshape
        self._adasam = adasam
        self._njobs = njobs

        self._randnQ = []
        self._paramQ = []
        self._valueQ = []

    def ask(self, batch=1):
        if len(self._paramQ) > 0:
            assert len(self._paramQ) - len(self._valueQ) >= batch
            start = len(self._valueQ)
            return self._paramQ[start:start+batch]
        population = self._npop * batch
        s = np.random.randn(population, self._ndims)
        z = self._mu + self._sigma * np.dot(s, self._bmat)
        z = torch.round(torch.clip(z, self._lowers, self._uppers)).long()
        # z = torch.clip(z, self._lowers, self._uppers)
        self._randnQ = s.tolist()
        self._paramQ = z.tolist()
        return self._paramQ[:batch]

    def tell(self, params, values):
        super().tell(params, values)
        self._valueQ.extend(values)
        if len(self._valueQ) < len(self._paramQ):
            return
        assert len(self._valueQ) == len(self._paramQ)

        randn = torch.tensor(self._randnQ, **POSTFIX)
        params = torch.tensor(self._paramQ, **POSTFIX)
        values = torch.tensor(self._valueQ, **POSTFIX)
        if len(values.shape) > 1:
            values = torch.mean(values, dim=-1)
        utilities = values
        if self._fshape:
            weights = []
            population = len(self._paramQ)
            utilities = [max(0, math.log(1 + 0.5*population) - math.log(k))
                         for k in range(1, population+1)]
            utilities = torch.tensor(utilities, **POSTFIX)
            utilities /= torch.sum(utilities)
            utilities -= 1.0 / population
            utilities = utilities.flip(0)
        isort = torch.argsort(values)

        s = randn[isort]
        u = utilities[isort]
        z = params[isort]
        f = values[isort]
        if self._adasam and not self._sigma_old is None:
            self._eta_sigma = self.adasam(self._eta_sigma, self._mu, self._sigma, self._bmat, self._sigma_old, z)

        eyemat = torch.eye(self._ndims)
        dj_delta = u.view(1, -1) @ s
        dj_mmat = s.T @ (s*u.view(-1, 1)) - torch.sum(u) * eyemat
        dj_sigma = torch.trace(dj_mmat) * (1.0/self._ndims)
        dj_bmat = dj_mmat - dj_sigma * eyemat
        self._sigma_old = self._sigma

        self._mu += (self._eta_mu * self._sigma * (dj_delta @ self._bmat)).view(-1)
        self._mu = torch.clip(self._mu, self._lowers, self._uppers)
        self._sigma *= torch.exp(0.5 * self._eta_sigma * dj_sigma)
        self._bmat = self._bmat @ torch.matrix_exp(0.5 * self._eta_bmat * dj_bmat)

        self._randnQ = []
        self._paramQ = []
        self._valueQ = []

    def adasam(self, eta_sigma, mu, sigma, bmat, sigma_old, z):
        if isinstance(z, torch.Tensor):
            z = z.detach().cpu().numpy()

        eta_sigma_init = self._eta_sigma_init
        dim = self._ndims
        c = .1
        rho = 0.5 - 1./(3*(dim+1))  # empirical

        bbmat = bmat.T @ bmat
        cov = sigma**2 * bbmat
        sigma_ = sigma * math.sqrt(sigma*(1./sigma_old))  # increase by 1.5
        cov_ = sigma_**2 * bbmat

        p0 = multivariate_normal.logpdf(z, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(z, mean=mu, cov=cov_)
        w = np.exp(p1-p0)

        # Mann-Whitney. It is assumed z was in ascending order.
        n = z.shape[0]
        n_ = np.sum(w)
        u_ = np.sum(w * (np.arange(0, n) + 0.5))
        u_mu = n*n_*0.5
        u_sigma = math.sqrt(n*n_*(n+n_+1)/12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1-c)*eta_sigma + c*eta_sigma_init
        else:
            return min(1, (1+c)*eta_sigma)


class SNES(BlackSolver):
    def __init__(self, ranges, nobjs=1,
                 mu=None, sigma=None,
                 npop=None, # Note: population = npop * batch
                 eta_mu=1.0, eta_sigma=None, fshape=True, njobs=1):
        super().__init__(ranges, nobjs, "NES")

        self._ndims = len(ranges)
        self._lowers = torch.tensor([x[0] for x in ranges], **POSTFIX)
        self._uppers = torch.tensor([x[1] for x in ranges], **POSTFIX)
        self._mu = 0.5 * (self._lowers + self._uppers) if mu is None else mu
        self._sigma = 0.25 * (self._uppers - self._lowers) if sigma is None else sigma

        self._npop = int(4 + 3*math.log(self._ndims)) if npop is None else npop
        self._eta_mu = eta_mu
        self._eta_sigma = self._eta_sigma_init = (3+math.log(self._ndims))/(5*math.sqrt(self._ndims)) if eta_sigma is None else eta_sigma
        self._fshape = fshape
        self._njobs = njobs

        self._randnQ = []
        self._paramQ = []
        self._valueQ = []

    def ask(self, batch=1):
        if len(self._paramQ) > 0:
            assert len(self._paramQ) - len(self._valueQ) >= batch
            start = len(self._valueQ)
            return self._paramQ[start:start+batch]
        population = self._npop * batch
        s = np.random.randn(population, self._ndims)
        z = self._mu + self._sigma.view(1, -1) * s
        z = torch.clip(z, self._lowers, self._uppers)
        self._randnQ = s.tolist()
        self._paramQ = z.tolist()
        return self._paramQ[:batch]

    def tell(self, params, values):
        super().tell(params, values)
        self._valueQ.extend(values)
        if len(self._valueQ) < len(self._paramQ):
            return
        assert len(self._valueQ) == len(self._paramQ)

        randn = torch.tensor(self._randnQ, **POSTFIX)
        params = torch.tensor(self._paramQ, **POSTFIX)
        values = torch.tensor(self._valueQ, **POSTFIX)
        if len(values.shape) > 1:
            values = torch.mean(values, dim=-1)
        utilities = values
        if self._fshape:
            weights = []
            population = len(self._paramQ)
            utilities = [max(0, math.log(1 + 0.5*population) - math.log(k))
                         for k in range(1, population+1)]
            utilities = torch.tensor(utilities, **POSTFIX)
            utilities /= torch.sum(utilities)
            utilities -= 1.0 / population
            utilities = utilities.flip(0)
        isort = torch.argsort(values)

        s = randn[isort]
        u = utilities[isort]
        z = params[isort]
        f = values[isort]

        eyemat = torch.eye(self._ndims)
        dj_delta = (u.view(1, -1) @ s).view(-1)
        dj_sigma = (u.view(1, -1) @ (s * s - 1)).view(-1)

        self._mu += self._eta_mu * self._sigma * dj_delta
        self._mu = torch.clip(self._mu, self._lowers, self._uppers)
        self._sigma *= torch.exp(0.5 * self._eta_sigma * dj_sigma)

        self._randnQ = []
        self._paramQ = []
        self._valueQ = []


if __name__ == "__main__":
    NITERS = 50 #200
    BATCH = 1
    NOBJS = 1
    NVARS = 10
    REFPOINT = [2e3 for _ in range(NOBJS)]
    # ranges = [(0, 1) for _ in range(NVARS)]
    ranges = [(50,100),(10,50),(60,120),(120,150),(10,30),(10,30), (10,20),(8,10),(8,10),(5,8),(30,100),(4,15),(5,20),(40,60),(1,5),(1,5),(1,5)]
    solver = NES(ranges, nobjs=1, mu=None, amat=None,
                 eta_mu=1, eta_sigma=None, eta_bmat=None, npop=3,
                 fshape=True, adasam=True, njobs=1)
    # solver = SNES(ranges, nobjs=1, mu=None, sigma=None,
    #               eta_mu=1, eta_sigma=None, npop=None,
    #               fshape=True, njobs=1)
    problem = BatchWrapper(mbopcfunc_select, ndims = len(ranges))
    # problem = BatchWrapper(Branin, ndims=len(ranges))
    runner = BenchRunner(solver, problem, nobjs=NOBJS, refpoint=REFPOINT)
    result = runner.run(niters=NITERS, batch=BATCH)
    print(f"Score: {result} <- {runner._params}")
    print(f" -> {solver._mu}")
    print(f" -> {solver._sigma}")