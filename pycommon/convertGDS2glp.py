import os
import sys
sys.path.append(".")
import glob
import argparse

from readGDS import *
from glp import *

def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Input GDS file. ")
    parser.add_argument("-t", "--type", required=False, default="gds2glp", help="Convertion type: gds2glp, glp2img. ")
    parser.add_argument("-s", "--scale", required=False, type=float, default=1.0)
    parser.add_argument("-l", "--layer", required=False, type=int, default=11)
    parser.add_argument("-o", "--output", default=".", required=False, type=str, help="Output folder. ")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parseArgs()
    fmt = "gds" if args.type == "gds2glp" else "glp"
    filenames = [args.input, ]
    if os.path.exists(args.input) and os.path.isdir(args.input): 
        filenames = glob.glob(f"{args.input}/*.{fmt}")
    
    for filename in filenames: 
        if args.type == "gds2glp": 
            reader = ReaderGDS(filename)
            structs = reader.structs
            assert len(structs) == 1
            name, struct = list(structs.items())[0]
            struct.exportGLP(f"{args.output}/{os.path.basename(filename)[:-4]}.glp", 
                             scale=args.scale, layers=(args.layer, ))
        elif args.type == "glp2img": 
            design = Design(filename)
            img = design.image(sizeX=2048, sizeY=2048, offsetX=512, offsetY=512)
            cv2.imwrite(f"{args.output}/{os.path.basename(filename)[:-4]}.png", img)
        else: 
            assert args.type in ("gds2glp", "glp2img")