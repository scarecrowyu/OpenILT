import sys
sys.path.append(".")
import os
import random
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


    def solve(self, target, pattern:Mask, param_dic, curv = None, verbose = 1):
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)

        #Optimize Model-based  
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestMask = None
        bestIteration=None
        pattern.find_contour()
        pattern.fragment_edge(projection=param_dic['projection_range'], lengthCorner=param_dic['lengthCorner'], lengthNormal=param_dic['lengthNormal'], lengthMin=param_dic['lengthMin'])
        pattern.add_sraf(range1=param_dic['range1'], range2=param_dic['range2'], distance2=param_dic['distance2'], distance3=param_dic['distance3'],\
                         width1=param_dic['width1'], width2=param_dic['width2'], width3=param_dic['width3'], lratio1=param_dic['lratio1'], lratio2=param_dic['lratio2'], lratio3=param_dic['lratio3'])
        mask = pattern._mask
        for idx in range(self._config["Iterations"]): 
            printedNom, printedMax, printedMin = self._lithosim(mask)
            # l2, pvb, epe, shot = evaluation.evaluate(mask, target, litho, scale=SCALE, shots=False)
            # l2loss = func.mse_loss(printedNom, target, reduction="sum")
            # pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            # pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            # pvband = torch.sum((printedMax >= 0.5) != (printedMin >= 0.5))
            l2, pvb, epe, shot = evaluation.evaluate(mask, target, self._lithosim, scale=1, shots=False)
            loss = l2 + self._config["WeightPVBand"] * pvb + 4000 * epe
            if idx ==0:
                loss_init = loss
            # cv2.imwrite(f"./tmp/mbopc_{idx}_mask.png", (mask * 255).detach().cpu().numpy())
            # cv2.imwrite(f"./tmp/mbopc_{idx}_wafer.png", (printedNom* 255).detach().cpu().numpy())
            # print(f"[Iteration {idx}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; ")  
            if idx <10:
                pattern.update_fragments(projection_step=param_dic['projection_step'], corner_step=param_dic['corner_step'], normal_step=param_dic['normal_step'], nominalImage=printedNom, epeprobe = 8)
            else:
                pattern.update_fragments(projection_step=1, corner_step=1, normal_step=1, nominalImage=printedNom,epeprobe=14)
            mask = pattern.updateMask()
            if verbose == 1: 
                print(f"[Testcase ]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")
            if bestMask is None or loss < lossMin: 
                lossMin, l2Min, pvbMin = loss, l2, pvb
                bestIteration = idx
                bestMask = mask.detach().clone()  
        cv2.imwrite(f"./tmp/mbopc_best_mask.png", (bestMask * 255).detach().cpu().numpy())
        bestPrintNom, _, _ = self._lithosim(bestMask)
        binaryNom = torch.zeros_like(bestPrintNom)
        binaryNom[bestPrintNom>=0.5] = 1
        cv2.imwrite(f"./tmp/mbopc_best_wafer.png", (binaryNom* 255).detach().cpu().numpy()) 
        
        return l2Min, pvbMin, bestMask,bestIteration, loss_init            
    
# def parallel():

def mbopcfunc(parameter, dim):
    parameter = [100,50,120,150,20,20,10,9,8,5,100,5,10,50,2,2,3]
    parameter_dic = {}
    parameter_dic['range1'] = parameter[0]
    parameter_dic['range2'] = parameter[1]
    parameter_dic['distance2'] = parameter[2]
    parameter_dic['distance3'] = parameter[3]
    parameter_dic['width1'] = parameter[4]
    parameter_dic['width2'] = parameter[5]
    parameter_dic['width3'] = parameter[6]
    parameter_dic['lratio1'] = float(parameter[7])/10
    parameter_dic['lratio2'] = float(parameter[8])/10
    parameter_dic['lratio3'] = float(parameter[9])/10

    parameter_dic['projection_range'] = parameter[10]
    parameter_dic['lengthCorner'] = parameter[11]
    parameter_dic['lengthNormal'] = parameter[12]
    parameter_dic['lengthMin'] = parameter[13]
    parameter_dic['projection_step'] = parameter[14]
    parameter_dic['corner_step'] = parameter[15]
    parameter_dic['normal_step'] = parameter[16]

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
    # design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-66560.glp", down=10)
    design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-148480.glp", down=10)
    # design = glp.Design(f"./benchmark/ICCAD2013/M1_test3.glp", down=SCALE)
    design.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
    target, params = initializer.PixelInit().run(design, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
    pattern = Mask(target)
    l2, pvb, bestMask = solver.solve(target, pattern, param_dic = parameter_dic, curv=None)
    l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=False)
    print(f"[Testcase ]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f};")
    cost = l2+pvb
    return cost
    # l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=False)
    # print(f"[Testcase ]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")

def mbopcfunc_select(parameter, dim):
    sample_size = 1
    # parameter = [100,50,120,150,20,20,10,9,8,5,100,5,10,50,2,2,3]
    parameter_dic = {}
    parameter_dic['range1'] = parameter[0]
    parameter_dic['range2'] = parameter[1]
    parameter_dic['distance2'] = parameter[2]
    parameter_dic['distance3'] = parameter[3]
    parameter_dic['width1'] = parameter[4]
    parameter_dic['width2'] = parameter[5]
    parameter_dic['width3'] = parameter[6]
    parameter_dic['lratio1'] = float(parameter[7])/10
    parameter_dic['lratio2'] = float(parameter[8])/10
    parameter_dic['lratio3'] = float(parameter[9])/10

    parameter_dic['projection_range'] = parameter[10]
    parameter_dic['lengthCorner'] = parameter[11]
    parameter_dic['lengthNormal'] = parameter[12]
    parameter_dic['lengthMin'] = parameter[13]
    parameter_dic['projection_step'] = parameter[14]
    parameter_dic['corner_step'] = parameter[15]
    parameter_dic['normal_step'] = parameter[16]

    SCALE = 1
    l2s = []
    pvbs = []
    epes = []
    shots = []
    runtimes = []
    costs = []
    cfg   = MbOPCCfg("./config/mbopc2048.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solver = MbOPC(cfg, litho)
    folder_path = "/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/"
    file_list = [file for file in os.listdir(folder_path) if file.startswith("cropped") and file.endswith(".glp")]
    random_files = random.sample(file_list, sample_size)
    for idx in range(sample_size):
        # design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-66560.glp", down=10)
        # design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-148480.glp", down=10)
        design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-97280.glp", down=10)
        # design = glp.Design(folder_path+file_list[idx], down=10)
        # design = glp.Design(f"./benchmark/ICCAD2013/M1_test3.glp", down=SCALE)
        design.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        target, params = initializer.PixelInit().run(design, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
        pattern = Mask(target)
        l2, pvb, bestMask,bestIter,loss_init = solver.solve(target, pattern, param_dic = parameter_dic, curv=None, verbose=0)
        l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=False)
        print(f"[Testcase {random_files[idx]}]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}; Iteration:{bestIter:.0f}")
        cost = (l2+pvb+4000*epe)/loss_init
        l2s.append(l2)
        pvbs.append(pvb)
        epes.append(epe)
        costs.append(cost)
    return np.mean(costs)
    # l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=SCALE, shots=False)
    # print(f"[Testcase ]: L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shot: {shot:.0f}")
parameter = [100,50,100,130,25,25,15,9,8,7,50,5,10,50,2,2,3]
parameter_NES = [78, 52, 60, 120, 14, 15, 13, 9, 10, 5, 76, 11, 12, 40, 4, 1, 3]
# fit = mbopcfunc(parameter, len(parameter))
fit = mbopcfunc_select(parameter, len(parameter))
