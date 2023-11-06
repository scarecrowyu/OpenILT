import numpy as np
from torch import nn
import torch

import gdspy
from gdspy import Cell, CellReference
from typing import Dict, List

device = "cpu"
class Layout:
    nano_unit = 1e-9
    def __init__(self, name : str, layer : int, filePath : str = None, unit : float = 1e-6) -> None:
        if filePath != None:
            self.filePath = filePath
        else:
            self.filePath = f"/home/omen/Documents/code/python/DAC24-ziyang/benchmark/gds"
        self.fullFilePath = self.filePath + "/" + name + ".gds"
        self.unit = 1e-6
        self.layer = layer
        self.gdslib = gdspy.GdsLibrary(unit=unit)
        self.gdslib.read_gds(self.fullFilePath)
        self.top_cell : Cell = self.gdslib.top_level()[0]
        self.scales = self.unit // self.nano_unit
        
        self.cell_ref : Dict[str, List] = dict()
    
        
        self.__clean()
        self.__getReferences()
        
    def getTopCell(self):
        return self.top_cell
        
    def __clean(self):
        remove = []
        for cell in self.gdslib.cells.values():
            assert isinstance(cell, Cell)
            
            cell.remove_paths(lambda x : x)
            cell.remove_labels(lambda x : x)
            cell.remove_polygons(lambda pts, layer, datatype: layer != self.layer)
            
            if len(cell.references) == 0 and len(cell.get_polygons()) == 0:
                remove.append(cell.name)
            else:
                self.cell_ref[cell.name] = []
        for name in remove:
            self.gdslib.cells.pop(name)
        
    def writeGds(self, path : str):
        self.gdslib.write_gds(path)
        
    def __getReferences(self):
        for ref in self.top_cell.references:
            assert isinstance(ref, CellReference)
            if isinstance(ref.ref_cell, str):
                ref_name = ref.ref_cell
            else:
                assert isinstance(ref.ref_cell, Cell)
                ref_name = ref.ref_cell.name
            if ref_name not in self.cell_ref.keys():
                continue
            self.cell_ref[ref_name].append(ref)

    def getNumCells(self):
        s = 0
        for _, v in self.cell_ref.items():
            s += len(v)
        return s
    

if __name__ == "__main__":
    layout = Layout("gcd_45nm", layer=11)
    print(f"[NUM CELLS] : {layout.getNumCells()}")
    


            
            

            
            
            
        
            
                
            
            
        
        

        
    
        
        
        
        
        
        
        