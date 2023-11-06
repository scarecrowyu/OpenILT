import sys
sys.path.append(".")
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func
import seaborn as sns
import matplotlib.pyplot as plt

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
import pylitho.simple as lithosim
from pycommon.polygon import FragmentsOneEdge, Mask # For model based opc
# import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

class MbOPCCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[SimpleILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]

class MbOPC:
    def __init__(self, config=MbOPCCfg("./config/mbopc2048.txt"), lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False): 
        super(MbOPC, self).__init__()
        self._config = config
        self._device = device
        # LevelSet
        self._lithosim = lithosim.to(DEVICE)
        if multigpu: 
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1


    def solve(self, target, params, curv = None, verbose = 0):
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor): 
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
            backup = params
            params = params.clone().detach().requires_grad_(True)

            #Optimize Model-based  

    
# def parallel():
SCALE = 1
l2s = []
pvbs = []
epes = []
shots = []
runtimes = []
cfg   = MbOPCCfg("./config/mbopc2048.txt")
litho = lithosim.LithoSim("./config/lithosimple.txt")
solver = MbOPC(cfg, litho)
# for i in range(1):
design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-66560.glp", down=10)
design.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
target, params = initializer.PixelInit().run(design, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
pattern = Mask(target)
pattern.find_contour()
pattern.fragment_edge(projection=30, lengthCorner=10, lengthNormal=30, lengthMin=50)
pattern.add_sraf(range1=30, range2=30, distance2=100, distance3=150, width1=20, width2=20, width3=10, lratio1=0.8, lratio2=0.6, lratio3=0.4)
mask = pattern._mask
for idx in range(cfg._config["Iterations"]): 
    printedNom, printedMax, printedMin = solver._lithosim(mask)
    l2, pvb, epe, shot = evaluation.evaluate(mask, target, litho, scale=SCALE, shots=False)
    cv2.imwrite(f"./tmp/mbopc_{idx}.png", (mask * 255).detach().cpu().numpy())
    print(f"[Testcase {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; ")
    pattern.update_fragments(projection_step=3, corner_step=3, normal_step=6, nominalImage=printedNom)
    mask = pattern.updateMask()
