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

def dominate(a, b):
    assert len(a) == len(b)
    domin1 = True
    domin2 = False
    for idx in range(len(a)):
        if a[idx] > b[idx]:
            domin1 = False
        elif a[idx] < b[idx]:
            domin2 = True
    return domin1 and domin2

def newParetoSet(paretoParams, paretoValues, newParams, newValue):
    assert len(paretoParams) == len(paretoValues)
    dupli = False
    removed = set()
    indices = []
    for idx, elem in enumerate(paretoValues):
        if str(paretoParams[idx]) == str(newParams):
            dupli = True
            break
        if dominate(newValue, elem):
            removed.add(idx)
    if dupli:
        return paretoParams, paretoValues
    for idx, elem in enumerate(paretoValues):
        if not idx in removed:
            indices.append(idx)
    newParetoParams = []
    newParetoValues = []
    for index in indices:
        newParetoParams.append(paretoParams[index])
        newParetoValues.append(paretoValues[index])
    bedominated = False
    for idx, elem in enumerate(newParetoValues):
        if dominate(elem, newValue):
            bedominated = True
    if len(removed) > 0:
        assert not bedominated
    if len(removed) > 0 or len(paretoParams) == 0 or not bedominated:
        newParetoParams.append(newParams)
        newParetoValues.append(newValue)
    return newParetoParams, newParetoValues

def pareto(params, values):
    paretoParams = []
    paretoValues = []

    for var, objs in zip(params, values):
        paretoParams, paretoValues = newParetoSet(paretoParams, paretoValues, var, objs)

    return paretoParams, paretoValues

def calcHV(refpoint, pareto):
    model = Hypervolume(-torch.tensor(refpoint))
    return model.compute(-torch.tensor(pareto))
    