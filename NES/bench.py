import os
import sys
import glob
import time
import random

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import botorch
from botorch.utils.multi_objective.hypervolume import Hypervolume
from tqdm import tqdm

from utils import calcHV, newParetoSet
from testfunc import *

class BenchRunner:
    def __init__(self, solver, problem, nobjs=1, refpoint=None):
        self._solver = solver
        self._problem = problem
        self._nobjs = nobjs
        self._refpoint = refpoint
        self._params = []
        self._values = []
        self._history = []
        self._solvetimes = []
        self._evaltimes = []
        assert self._nobjs > 0, f"[BenchRunner]: Invalid number of objectives {self._nobjs}"

    @property
    def params(self):
        return self._params
    @property
    def values(self):
        return self._values
    @property
    def history(self):
        return self._history
    @property
    def solvetime(self):
        return np.mean(self._solvetimes)
    @property
    def evaltime(self):
        return np.mean(self._evaltimes)

    def run(self, niters=100, batch=1):
        progress = tqdm(range(niters))
        for idx in progress:
            solvetime = time.time()
            params = self._solver.ask(batch)  # Suggest batch of samples 
            solvetime = time.time() - solvetime
            evaltime = time.time()
            values = self._problem(params)  # Evaluate values of parameters
            evaltime = time.time() - evaltime
            self._solver.tell(params, values)
            self._history.append((params, values))
            self._solvetimes.append(solvetime)
            self._evaltimes.append(evaltime)
            if self._nobjs == 1:
                index = np.argmin(values)
                if len(self._values) == 0 or values[index][0] < self._values[0][0]:
                    self._params = [params[index], ]
                    self._values = [values[index], ]
            else:
                for var, objs in zip(params, values):
                    self._params, self._values = newParetoSet(self._params, self._values, var, objs)
            quality = self._values[0][0] if self._nobjs == 1 else calcHV(self._refpoint, self._values)
            progress.set_postfix(quality=f"{quality:.4}")
        return quality


class OptunaRunner:
    def __init__(self, solver, problem, ranges, nobjs=1, refpoint=None):
        self._problem = problem
        self._ranges = ranges
        self._nobjs = nobjs
        self._refpoint = refpoint
        self._params = []
        self._values = []

        sampler = None
        if solver == "TPE":
            sampler = optuna.samplers.TPESampler()
        elif solver == "NSGAII":
            sampler = optuna.samplers.NSGAIISampler()
        elif solver == "CMAES":
            sampler = optuna.samplers.CmaEsSampler()
        else:
            sampler = optuna.samplers.QMCSampler()
        self._solver = optuna.create_study(directions=["minimize" for _ in range(self._nobjs)], \
                                           sampler=sampler)

    def run(self, niters=100, batch=1):
        if batch > 1:
            print(f"[OptunaRunner]: Warning, optuna only runs sequentially, the number of iterations is niters x batch = {niters * batch}")
        def objective(trial):
            param = []
            for idx in range(len(self._ranges)):
                name = f"x{idx}"
                # value = trial.suggest_float(name, self._ranges[idx][0], self._ranges[idx][1])
                value = trial.suggest_int(name, self._ranges[idx][0], self._ranges[idx][1])
                param.append(value)
            score = self._problem([param, ])[0]
            return score
        self._solver.optimize(objective, n_trials=niters*batch)
        if self._nobjs == 1:
            trial = self._solver.best_trial
            self._params = [trial.params, ]
            self._values = [[trial.value], ]
        else:
            trials = self._solver.best_trials
            results = []
            for trial in trials:
                param = trial.params
                value = trial.values
                self._params.append(param)
                self._values.append(value)
        quality = self._values[0][0] if self._nobjs == 1 else calcHV(self._refpoint, self._values)
        return quality


class BlackSolver:
    def __init__(self, ranges, nobjs=1, name="BlackSolver"):
        self._name = name
        self._ranges = ranges
        self._nparams = len(ranges)
        self._nobjs = nobjs
        self._params = []
        self._values = []

    def ask(self, batch=1):
        pass

    def tell(self, params, values):
        self._params.extend(params)
        self._values.extend(values)


class RandomSearch(BlackSolver):
    def __init__(self, ranges, nobjs=1):
        super().__init__(ranges, nobjs, "RandomSearch")

    def ask(self, batch=1):
        params = []
        for idx in range(batch):
            param = []
            for jdx in range(self._nparams):
                param.append(round(self._ranges[jdx][0] + random.random() * (self._ranges[jdx][1] - self._ranges[jdx][0])))
            params.append(param)
        return params

    def tell(self, params, values):
        super().tell(params, values)


def testBench0():
    NITERS = 10
    BATCH = 1
    NOBJS = 1
    NVARS = 10
    REFPOINT = [2e3 for _ in range(NOBJS)]
    ranges = [(0, 1) for _ in range(NVARS)]
    solver = RandomSearch(ranges, nobjs=NOBJS)
    problem = BatchWrapper(Branin, ndims=len(ranges))
    runner = BenchRunner(solver, problem, nobjs=NOBJS, refpoint=REFPOINT)
    result = runner.run(niters=NITERS, batch=BATCH)
    print(f"Score: {result}")

def testBench1():
    NITERS = 10
    BATCH = 1
    NOBJS = 2
    NVARS = 10
    REFPOINT = [2e3 for _ in range(NOBJS)]
    ranges = [(0, 1) for _ in range(NVARS)]
    solver = RandomSearch(ranges, nobjs=NOBJS)
    problem = BatchWrapper([Branin, Powell], ndims=len(ranges))
    runner = BenchRunner(solver, problem, nobjs=NOBJS, refpoint=REFPOINT)
    result = runner.run(niters=NITERS, batch=BATCH)
    print(f"Score: {result}")

def testOptuna0():
    NITERS = 10
    BATCH = 1
    NOBJS = 1
    NVARS = 10
    REFPOINT = [2e3 for _ in range(NOBJS)]
    ranges = [(0, 1) for _ in range(NVARS)]
    solver = "NSGAII"
    problem = BatchWrapper(Branin, ndims=len(ranges))
    runner = OptunaRunner(solver, problem, ranges, nobjs=NOBJS, refpoint=REFPOINT)
    result = runner.run(niters=NITERS, batch=BATCH)
    print(f"Score: {result}")

def testOptuna1():
    NITERS = 200
    BATCH = 1
    NOBJS = 2
    NVARS = 10
    REFPOINT = [2e3 for _ in range(NOBJS)]
    ranges = [(0, 1) for _ in range(NVARS)]
    solver = "NSGAII"
    problem = BatchWrapper([Branin, Powell], ndims=len(ranges))
    runner = OptunaRunner(solver, problem, ranges, nobjs=NOBJS, refpoint=REFPOINT)
    result = runner.run(niters=NITERS, batch=BATCH)
    print(f"Score: {result}")

if __name__ == "__main__":
    # testBench0()
    # testBench1()
    testOptuna0()
    # testOptuna1()