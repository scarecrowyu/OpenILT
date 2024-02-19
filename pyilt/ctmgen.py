import sys
sys.path.append(".")
import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as func

from pycommon.settings import *
import pycommon.utils as common
import pycommon.glp as glp
# import pylitho.simple as lithosim
import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation

# DEVICE = "cpu"
class CurvILTCfg: 
    def __init__(self, config): 
        # Read the config from file or a given dict
        if isinstance(config, dict): 
            self._config = config
        elif isinstance(config, str): 
            self._config = common.parseConfig(config)
        required = ["Iterations", "TargetDensity", "SigmoidSteepness", "SigmoidOffset", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize", 
                    "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in required: 
            assert key in self._config, f"[CurvILT]: Cannot find the config {key}."
        intfields = ["Iterations", "TileSizeX", "TileSizeY", "OffsetX", "OffsetY", "ILTSizeX", "ILTSizeY"]
        for key in intfields: 
            self._config[key] = int(self._config[key])
        floatfields = ["TargetDensity", "SigmoidSteepness", "SigmoidOffset", "WeightEPE", "WeightPVBand", "WeightPVBL2", "StepSize"]
        for key in floatfields: 
            self._config[key] = float(self._config[key])
    
    def __getitem__(self, key): 
        return self._config[key]

class CurvILT: 
    def __init__(self, config, lithosim=lithosim.LithoSim("./config/lithosimple.txt"), device=DEVICE, multigpu=False): 
        super(CurvILT, self).__init__()
        self._config = config
        self._device = device
        # Lithosim
        self._lithosim = lithosim.to(DEVICE)
        if multigpu: 
            self._lithosim = nn.DataParallel(self._lithosim)
        # Filter
        self._filter = torch.zeros([self._config["TileSizeX"], self._config["TileSizeY"]], dtype=REALTYPE, device=self._device)
        self._filter[self._config["OffsetX"]:self._config["OffsetX"]+self._config["ILTSizeX"], \
                     self._config["OffsetY"]:self._config["OffsetY"]+self._config["ILTSizeY"]] = 1
    
    def solve(self, target, params, verbose=0): 
        # Initialize
        if not isinstance(target, torch.Tensor): 
            target = torch.tensor(target, dtype=REALTYPE, device=self._device)
        if not isinstance(params, torch.Tensor): 
            params = torch.tensor(params, dtype=REALTYPE, device=self._device)
        backup = params
        params = params.clone().detach().requires_grad_(True)

        # Optimizer 
        opt = optim.SGD([params], lr=self._config["StepSize"])
        # opt = optim.Adam([params], lr=0.2*self._config["StepSize"])

        # Optimization process
        lossMin, l2Min, pvbMin = 1e12, 1e12, 1e12
        bestParams = None
        bestMask = None
        for idx in range(self._config["Iterations"]): 
            if len(params.shape) == 2: 
                pooled = func.avg_pool2d(params[None, None, :, :], 7, stride=1, padding=3)[0, 0]
            else: 
                pooled = func.avg_pool2d(params.unsqueeze(1), 7, stride=1, padding=3)[:, 0]
            mask = torch.sigmoid(self._config["SigmoidSteepness"] * (pooled - self._config["SigmoidOffset"])) * self._filter
            printedNom, printedMax, printedMin = self._lithosim(mask)
            l2loss = func.mse_loss(printedMax, target, reduction="sum")
            pvbl2 = func.mse_loss(printedMax, target, reduction="sum") + func.mse_loss(printedMin, target, reduction="sum")
            pvbloss = func.mse_loss(printedMax, printedMin, reduction="sum")
            pvband = torch.sum((printedMax >= self._config["TargetDensity"]) != (printedMin >= self._config["TargetDensity"]))

            kernelCurv = torch.tensor([[-1.0/16, 5.0/16, -1.0/16], [5.0/16, -1.0, 5.0/16], [-1.0/16, 5.0/16, -1.0/16]], dtype=REALTYPE, device=DEVICE)
            curvature = func.conv2d(printedNom[None, None, :, :], kernelCurv[None, None, :, :])[0, 0]
            losscurv = func.mse_loss(curvature, torch.zeros_like(curvature), reduction="sum")

            loss = l2loss + self._config["WeightPVBL2"] * pvbl2 + self._config["WeightPVBand"] * pvbloss # + 2e2 * losscurv
            if verbose == 1: 
                print(f"[Iteration {idx}]: L2 = {l2loss.item():.0f}; PVBand: {pvband.item():.0f}")

            if bestParams is None or bestMask is None or loss.item() < lossMin: 
                lossMin, l2Min, pvbMin = loss.item(), l2loss.item(), pvband.item()
                bestParams = params.detach().clone()
                if len(params.shape) == 2: 
                    pooled = func.avg_pool2d(bestParams[None, None, :, :], 7, stride=1, padding=3)[0, 0]
                else: 
                    pooled = func.avg_pool2d(bestParams.unsqueeze(1), 7, stride=1, padding=3)[:, 0]
                bestMask = torch.sigmoid(self._config["SigmoidSteepness"] * (pooled - self._config["SigmoidOffset"])) * self._filter
                bestMask[bestMask > 0.5] = 1.0
                bestMask[bestMask <= 0.5] = 0.0
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        return l2Min, pvbMin, bestParams, bestMask


if __name__ == "__main__": 
    infile = f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_271360-286720.glp"
    if len(sys.argv) > 1: 
        infile = sys.argv[1]
    prefix = f"tmp/CTMGen"
    if len(sys.argv) > 2: 
        prefix = sys.argv[2]

    ScaleLow = 8
    ScaleMid = 4
    l2s = []
    pvbs = []
    epes = []
    runtimes = []
    cfgLow = CurvILTCfg("./config/ctmgen256.txt")
    cfgMid = CurvILTCfg("./config/ctmgen512.txt")
    litho = lithosim.LithoSim("./config/lithosimple.txt")
    solverLow = CurvILT(cfgLow, litho)
    solverMid = CurvILT(cfgMid, litho)
    test = evaluation.Basic(litho, 0.5)
    epeCheck = evaluation.EPEChecker(litho, 0.5)

    runtime = 0
    # Reference 
    ref = glp.Design(infile, down=10)
    ref.center(cfgMid["TileSizeX"]*ScaleMid, cfgMid["TileSizeY"]*ScaleMid, cfgMid["OffsetX"]*ScaleMid, cfgMid["OffsetY"]*ScaleMid)
    
    # Low resolution
    design = glp.Design(infile, down=ScaleLow*10)
    design.center(cfgLow["TileSizeX"], cfgLow["TileSizeY"], cfgLow["OffsetX"], cfgLow["OffsetY"])
    target, params = initializer.PixelInit().run(design, cfgLow["TileSizeX"], cfgLow["TileSizeY"], cfgLow["OffsetX"], cfgLow["OffsetY"])
    begin = time.time()
    l2, pvb, bestParams, bestMask = solverLow.solve(target, target)
    runtime += time.time() - begin

    # -> Evaluation
    target, params = initializer.PixelInit().run(ref, cfgLow["TileSizeX"]*ScaleLow, cfgLow["TileSizeY"]*ScaleLow, cfgLow["OffsetX"]*ScaleLow, cfgLow["OffsetY"]*ScaleLow)
    l2, pvb = test.run(bestMask, target, scale=ScaleLow)
    epeIn, epeOut = epeCheck.run(bestMask, target, scale=ScaleLow)
    epe = epeIn + epeOut
    logLow = f"L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}"

    # Mid resolution
    design = glp.Design(infile, down=ScaleMid*10)
    design.center(cfgMid["TileSizeX"], cfgMid["TileSizeY"], cfgMid["OffsetX"], cfgMid["OffsetY"])
    target, params = initializer.PixelInit().run(design, cfgMid["TileSizeX"], cfgMid["TileSizeY"], cfgMid["OffsetX"], cfgMid["OffsetY"])
    params = func.interpolate(bestParams[None, None, :, :], scale_factor=2, mode="nearest")[0, 0]
    mask = func.interpolate(bestMask[None, None, :, :], scale_factor=2, mode="nearest")[0, 0]
    begin = time.time()
    l2, pvb, bestParams, bestMask = solverMid.solve(target, params)
    runtime += time.time() - begin

    # -> Evaluation
    target, params = initializer.PixelInit().run(ref, cfgMid["TileSizeX"]*ScaleMid, cfgMid["TileSizeY"]*ScaleMid, cfgMid["OffsetX"]*ScaleMid, cfgMid["OffsetY"]*ScaleMid)
    l2, pvb = test.run(bestMask, target, scale=ScaleMid)
    epeIn, epeOut = epeCheck.run(bestMask, target, scale=ScaleMid)
    epe = epeIn + epeOut
    logMid = f"L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}"

    # -> Evaluation
    target, params = initializer.PixelInit().run(ref, cfgMid["TileSizeX"]*ScaleMid, cfgMid["TileSizeY"]*ScaleMid, cfgMid["OffsetX"]*ScaleMid, cfgMid["OffsetY"]*ScaleMid)
    l2, pvb, epe, shot = evaluation.evaluate(bestMask, target, litho, scale=ScaleMid, shots=False)
    logMid = f"L2 {l2:.0f}; PVBand {pvb:.0f}; EPE {epe:.0f}; Shots: {shot:.0f}"

    # Print Information
    def colormap():
        return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF','#B39EB1'], 20)

    print(f"[{infile}]: Low: {logLow} -> Mid: {logMid}; Runtime: {runtime:.2f}s")
    mask, resist = test.sim(bestMask, target, scale=ScaleMid)
    maskCTM = torch.sigmoid(cfgMid["SigmoidSteepness"] * (bestParams - cfgMid["SigmoidOffset"]))
    sns.set(font_scale = 2.5)
    fig = plt.figure(figsize=(512, 512), dpi=1)
    plt.subplots_adjust(top = 0.95, bottom = 0.05, right = 0.95, left = 0.05, hspace = 0, wspace = 0)
    sns_plot = sns.heatmap(maskCTM.detach().cpu().numpy(), vmin = 0.0, vmax = 1.0,cmap=colormap(),cbar= False, center = 0.5, xticklabels=False, yticklabels=False )
    plt.savefig('./tmp/ctm_mask.png')
    color_mask = torch.tensor([213,186,225,255])
    color_wafer = torch.tensor([83,126,238,255])
    # rgba_mask = (maskCTM.detach().cpu().unsqueeze(-1) * color_mask.unsqueeze(0).unsqueeze(0)).numpy()
    rgba_wafer = (resist.detach().cpu().unsqueeze(-1) * color_wafer.unsqueeze(0).unsqueeze(0)).numpy()
        #Plot Target

    color_target = torch.tensor([66,75,105,255])
    rgba_target = (target.detach().cpu().unsqueeze(-1) * color_target.unsqueeze(0).unsqueeze(0)).numpy()
    cv2.imwrite(f"./tmp/ctm_target.png", rgba_target)
    # cv2.imwrite(f"./tmp/ctm_mask.png", rgba_mask)
    cv2.imwrite(f"./tmp/ctm_wafer.png", rgba_wafer)
    # cv2.imwrite(f"{prefix}_target.png", cv2.resize((target * 255).detach().cpu().numpy(), (2048, 2048)))
    # cv2.imwrite(f"{prefix}_ctm.png",  cv2.resize((maskCTM * 255).detach().cpu().numpy(), (2048, 2048)))
    # cv2.imwrite(f"{prefix}_mask.png",  cv2.resize((mask * 255).detach().cpu().numpy(), (2048, 2048)))
    # cv2.imwrite(f"{prefix}_resist.png",  cv2.resize((resist * 255).detach().cpu().numpy(), (2048, 2048)))