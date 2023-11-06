import os
import sys
import math
import argparse

import klayout.db as pya

def parseArgs(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str, help="Input GDS file. ")
    parser.add_argument("-x", "--sizeX", default=10240, required=False, type=int, help="X size of a crop. ") #1024 / logical_unit
    parser.add_argument("-y", "--sizeY", default=10240, required=False, type=int, help="Y size of a crop. ")
    parser.add_argument("-s", "--stride", default=5120, required=False, type=int, help="Stride of cropping. ")#256 / logical_unit
    parser.add_argument("-l", "--layer", required=False, type=int, default=11)
    parser.add_argument("-o", "--output", default="cropped", required=False, type=str, help="Output GDS prefix. ")
    return parser.parse_args()

if __name__ == "__main__": 
    args = parseArgs()
    infile = args.input
    prefix = args.output
    sizeX = args.sizeX
    sizeY = args.sizeY
    areaXY = sizeX * sizeY
    stride = args.stride

    ly = pya.Layout()
    ly.read(infile)

    bbox = ly.top_cell().bbox()
    topcell = ly.top_cell().cell_index()
    layer = ly.layer(args.layer, 0)
    left = bbox.left
    bottom = bbox.bottom
    right = bbox.right
    top = bbox.top
    print(f"Bounding box: {bbox}, selected layer: {layer}")

    countValid = 0
    countTotal = 0
    for idx in range(left, right + (stride-1), stride): 
        print(f"X: {idx}")
        for jdx in range(bottom, top + (stride - 1), stride): 
            countTotal += 1
            cbox = pya.Box.new(idx, jdx, idx + sizeX, jdx + sizeY)
            cropped = ly.clip(topcell, cbox)
            cell = ly.cell(cropped)
            cell.flatten(-1)
            cell.name = "Cropped"

            shapes = cell.shapes(layer)
            if len(shapes) <= 3: 
                continue
            
            area = 0
            todel = []
            for shape in shapes.each(): 
                ratio = shape.area() / areaXY
                if ratio < 0.01: 
                    todel.append(shape)
                else: 
                    area += shape.area()
            for shape in todel: 
                shapes.erase(shape)
            for shape in shapes.each(): 
                ratio = shape.area() / areaXY
                assert ratio >= 0.01
            ratio = area / areaXY
            if ratio < 0.3: 
                continue
            else: 
                countValid += 1

            filename = f"{prefix}_{idx}-{jdx}.gds"
            print(f"To write: {filename}")
            cell.write(f"{prefix}_{idx}-{jdx}.gds")

    print(f"Valid count: {countValid} / {countTotal}")
