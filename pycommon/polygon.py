import sys
sys.path.append(".")
import math

import numpy as np 
import cv2
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
# import pylitho.exact as lithosim

import pyilt.initializer as initializer
import pyilt.evaluation as evaluation
# from pyilt.mbopc import MbOPCCfg

class FragmentsOneEdge:
    def __init__(self, direction = None, ref_edge = None): 
        self._corner_edge_pairs = None
        self._projection_edge_pairs = None
        self._normal_edge_pairs = None
        self._corner_offsets = None
        self._projection_offsets = None
        self._normal_offsets = None
        self._direction = direction
        self._ref_edge = ref_edge

        
    def add_corner_edge_pairs(self, edge_pairs):
        self._corner_edge_pairs = edge_pairs
        self._corner_offsets = torch.zeros(edge_pairs.shape[0]).long()
    
    def add_projection_edge_pairs(self, edge_pairs):
        self._projection_edge_pairs = edge_pairs
        self._projection_offsets = torch.zeros(edge_pairs.shape[0]).long()

    def add_normal_edge_pairs(self, edge_pairs):
        self._normal_edge_pairs = edge_pairs
        self._normal_offsets = torch.zeros(edge_pairs.shape[0]).long()

    def corner_edge_pairs(self):
        return self._corner_edge_pairs
    
    def projection_edge_pairs(self):
        return self._projection_edge_pairs
    
    def normal_edge_pairs(self):
        return self._normal_edge_pairs
    
    def corner_offsets(self):
        return self._corner_offsets
    
    def pojection_offsets(self):
        return self._projection_offsets 

    def normal_offsets(self):
        return self._normal_offsets
    
    def direction(self):
        return self._direction
    
    def ref_edge(self):
        return self._ref_edge
    
    def smooth(self, edge_gap):
        if self._corner_offsets == None:
            cat_offsets = self._normal_offsets
        else:
            cat_offsets = torch.cat([self._corner_offsets[0].unsqueeze(0), self._normal_offsets, self._corner_offsets[1].unsqueeze(0)])
        for i in range(1, len(cat_offsets-1)):
            if cat_offsets[i] - cat_offsets[i-1] > edge_gap:
                cat_offsets[i] = cat_offsets[i-1] + edge_gap
            elif cat_offsets[i] - cat_offsets[i-1] < - edge_gap:
                cat_offsets[i] = cat_offsets[i-1] - edge_gap
        if self._corner_offsets == None:
            self._normal_offsets = cat_offsets
        else:
            self._corner_offsets[0] = cat_offsets[0]
            self._corner_offsets[1] = cat_offsets[-1]
            self._normal_offsets = cat_offsets[1:-1]

    
    def movesegment(self, move):
        assert move.shape ==  self._offsets.shape, "The shape of movement should be equal to fragments"
        


class Mask:
    def __init__(self, ts_target):
        self._target = ts_target.clone()
        self._polygons = []
        self._fragments_h = []
        self._fragments_v = []
        # self._fragments_projection = []
        # self._fragments_corner = []
        # self._fragments_normal = []
        self._mask = ts_target.clone()

    def find_contour(self):
        np_mask = self._target.detach().cpu().numpy()
        contours, _  = cv2.findContours(np_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours: 
            contour = contour.reshape([-1, 2])
            contour_switch_xy = contour[:,[1,0]]
            self._polygons.append(torch.tensor(contour_switch_xy))

    def num_polygons(self):
        return len(self._polygons)

    def fragments_h(self):
        return self._fragments_h
    
    def fragments_v(self):
        return self._fragments_v

    def add_sraf(self, range1, range2, distance2, distance3, width1, width2,width3, lratio1, lratio2, lratio3):
        '''
        If there exist other targets in ( ,'range1'] then no SRAF, range1-> forbidden range
        if there exist other targets in ('range1', 'range2'], add one in center line with 'width1'
        if there does not have targets in (range2, ), add one 'width2' in 'distance2', add another one 'width3' in 'distance3' 
        '''
        target = self._target
        # self._mask = torch.zeros((target.shape[0], target.shape[1]))
        for polygon in self._polygons:
            polygon = torch.cat((polygon, polygon[0].unsqueeze(0)), dim=0)
            edge = polygon[1:] - polygon[:-1]  # Distance-1
            p1 = polygon[0]
            for i in range(1, len(polygon)):
                p2 = polygon[i]
                if edge[i-1][1]==0: # Vertical
                    if p2[0] > p1[0]: # means the right side of edge is inside the target, downward edge
                        forbidden_region = target[p1[0]-range1: p2[0]+range1, p1[1]-range1 : p1[1]]  
                        if not(forbidden_region.any().item()):
                            sraf1_region = target[p1[0]-range1-range2 : p2[0]+range1+range2, p1[1]-range1-range2 : p1[1]] 
                            if sraf1_region.any().item():
                                proj_edge_col_index = torch.nonzero(sraf1_region.any(dim=0))[-1]
                                srafy,srafx1 = (p1[0]+p2[0])//2, (p1[1] + proj_edge_col_index)//2
                                lly1, llx1, ury1, urx1 = (srafy - lratio1*(p2[0]-p1[0])//2).int(), srafx1-width1//2, (srafy+lratio1*(p2[0]-p1[0])//2).int(), srafx1+width1//2
                                # print(lly1,llx1, ury1, urx1)
                                self.drawSRAF(lly1, llx1, ury1, urx1)
                            else:
                                srafy, srafx2, srafx3 = (p1[0]+p2[0])//2, p1[1]-distance2, p1[1]-distance3
                                lly2, llx2, ury2, urx2 = (srafy-lratio2*(p2[0]-p1[0])//2).int(), srafx2-width2//2, (srafy+lratio2*(p2[0]-p1[0])//2).int(), srafx2+width2//2
                                lly3, llx3, ury3, urx3 = (srafy-lratio3*(p2[0]-p1[0])//2).int(), srafx3-width3//2, (srafy+lratio3*(p2[0]-p1[0])//2).int(), srafx3+width3//2
                                self.drawSRAF(lly2, llx2, ury2, urx2)
                                self.drawSRAF(lly3, llx3, ury3, urx3)
                    elif p1[0] > p2[0]: # means the left side of the edge is inside the target, upward edge
                        forbidden_region = target[p2[0]-range1 : p1[0]+range1 , p1[1]+1 : p1[1]+range1+1]
                        if not(forbidden_region.any().item()):
                            sraf1_region = target[p2[0]-range1-range2 : p1[0]+range1+range2, p1[1]+1 : p1[1]+range1+range2+1] 
                            if sraf1_region.any().item():
                                proj_edge_col_index = torch.nonzero(sraf1_region.any(dim=0))[0]
                                srafy,srafx1 = (p1[0]+p2[0])//2, (p1[1] + proj_edge_col_index)//2
                                lly1, llx1, ury1, urx1 = (srafy - lratio1*(p1[0]-p2[0])//2).int(), srafx1-width1//2, (srafy+lratio1*(p1[0]-p2[0])//2).int(), srafx1+width1//2
                                self.drawSRAF(lly1, llx1, ury1, urx1)
                            else:
                                srafy, srafx2, srafx3 = (p1[0]+p2[0])//2, p1[1]+distance2, p1[1]+distance3
                                lly2, llx2, ury2, urx2 = (srafy-lratio2*(p1[0]-p2[0])//2).int(), srafx2-width2//2, (srafy+lratio2*(p1[0]-p2[0])//2).int(), srafx2+width2//2
                                lly3, llx3, ury3, urx3 = (srafy-lratio3*(p1[0]-p2[0])//2).int(), srafx3-width3//2, (srafy+lratio3*(p1[0]-p2[0])//2).int(), srafx3+width3//2
                                self.drawSRAF(lly2, llx2, ury2, urx2)
                                self.drawSRAF(lly3, llx3, ury3, urx3)
                elif edge[i-1][0] == 0: # horizontal
                    if p2[1]>p1[1]: # means the upper side of edge is inside the target, rightward edge
                        forbidden_region = target[p1[0]+1: p1[0]+range1+1, p1[1]-range1: p2[1]+range1] 
                        if not(forbidden_region.any().item()): 
                            sraf1_region = target[p1[0]+1: p1[0]+range1+range2+1, p1[1]-range1-range2 : p2[1]+range1+range2] 
                            if sraf1_region.any().item():        
                                proj_edge_row_index = torch.nonzero(sraf1_region.any(dim=1))[0]     
                                srafy1, srafx = (p1[0]+proj_edge_row_index)//2,  (p1[1]+p2[1])//2
                                lly1, llx1, ury1, urx1 = srafy1 - width1//2, (srafx-lratio1*(p2[1]-p1[1])//2).int(), srafy1 + width1//2, (srafx+lratio1*(p2[1]-p1[1])//2).int()
                                self.drawSRAF(lly1, llx1, ury1, urx1)
                            else:
                                srafy2, srafy3, srafx = p1[0] + distance2, p1[0] + distance3, (p1[1]+p2[1])//2
                                lly2, llx2, ury2, urx2 = srafy2 - width2//2, (srafx-lratio2*(p2[1]-p1[1])//2).int(), srafy2 + width2//2, (srafx+lratio2*(p2[1]-p1[1])//2).int()
                                lly3, llx3, ury3, urx3 = srafy3 - width3//2, (srafx-lratio3*(p2[1]-p1[1])//2).int(), srafy3 + width3//2, (srafx+lratio3*(p2[1]-p1[1])//2).int()
                                self.drawSRAF(lly2, llx2, ury2, urx2)
                                self.drawSRAF(lly3, llx3, ury3, urx3)
                    elif p1[1]>p2[1]: # means the lower side of edge is inside the target, leftward edge
                        print('has this leftwards segments')
                        forbidden_region = target[p1[0]-range1: p1[0], p2[1]-range1: p1[1]+range1]
                        if not(forbidden_region.any().item()): 
                            sraf1_region = target[p1[0]-range1-range2: p1[0], p2[1]-range1-range2 : p1[1]+range1+range2]
                            if sraf1_region.any().item():
                                proj_edge_row_index = torch.nonzero(sraf1_region.any(dim=1))[-1]
                                srafy1, srafx = (p1[0]+proj_edge_row_index)//2,  (p1[1]+p2[1])//2
                                lly1, llx1, ury1, urx1 = srafy1-width1//2, (srafx-lratio1*(p1[1]-p2[1])//2).int(), srafy1 + width1//2, (srafx+lratio1*(p1[1]-p2[1])//2).int()
                                self.drawSRAF(lly1, llx1, ury1, urx1)
                            else:
                                srafy2, srafy3, srafx = p1[0] - distance2, p1[0] - distance3, (p1[1]+p2[1])//2
                                lly2, llx2, ury2, urx2 = srafy2 - width2//2, (srafx-lratio2*(p1[1]-p2[1])//2).int(), srafy2 + width2//2, (srafx+lratio2*(p1[1]-p2[1])//2).int()
                                lly3, llx3, ury3, urx3 = srafy3 - width3//2, (srafx-lratio3*(p1[1]-p2[1])//2).int(), srafy3 + width3//2, (srafx+lratio3*(p1[1]-p2[1])//2).int()
                                self.drawSRAF(lly2, llx2, ury2, urx2)
                                self.drawSRAF(lly3, llx3, ury3, urx3)
                        else:
                            print('all forbidden')
                p1 = p2                 


    def drawSRAF(self, lowlefty, lowleftx, upperrighty, upperrightx):
        self._mask[lowlefty: upperrighty, lowleftx: upperrightx] = 1                      

    def fragment_edge(self, projection, lengthCorner, lengthNormal, lengthMin):
        target = self._target
        for polygon in self._polygons:
            polygon = torch.cat((polygon, polygon[0].unsqueeze(0)), dim=0)
            edge = polygon[1:] - polygon[:-1]  # Distane-1
            legal_mask = (edge == 0).any(axis=1) #If mask value is true, the corresponding distances is h/v edge distance
            p1 = polygon[0]
            for i in range(1, len(polygon)):   
                # print('This is the {i}th edge of the polygon'.format(i=i))     
                p2 = polygon[i]
                # Find Projection part 
                if edge[i-1][1] == 0:  # vertical 
                    fragment = FragmentsOneEdge(direction ='down' if p2[0] > p1[0] else 'up', ref_edge =torch.cat((p1,p2)))
                    # Projection segment
                    if p2[0] > p1[0]: #means the right side of edge is inside the target, downward edge
                        target_slice = target[min(p1[0], p2[0]) : max(p1[0],p2[0]), p1[1]-projection : p1[1]]                     
                    elif p1[0] > p2[0]: # means the left side of the edge is inside the target, upward edge
                        target_slice = target[min(p1[0], p2[0]) : max(p1[0],p2[0]) , p1[1]+1 : p1[1]+projection+1]
                    if target_slice.any().item():
                        if p2[0] > p1[0]:
                            proj_edge_col_index = torch.nonzero(target_slice.any(dim=0))[-1]
                        elif p1[0] > p2[0]:  
                            proj_edge_col_index = torch.nonzero(target_slice.any(dim=0))[0]
                        proj_edge_col = target_slice[:, proj_edge_col_index]
                        # padded_col = np.insert(np.append(proj_edge_col, 0), 0, 0)
                        padded_col = torch.flatten(torch.cat((torch.zeros((1, 1)), proj_edge_col, torch.zeros((1, 1))),dim=0))
                        start_indices = torch.where(torch.diff(padded_col)==1)[0]
                        end_indices = torch.where(torch.diff(padded_col)==-1)[0]-1
                        edge_pairs = torch.stack((min(p1[0], p2[0])+start_indices, torch.full_like(start_indices, p1[1]), min(p1[0], p2[0])+end_indices, torch.full_like(end_indices, p1[1])),dim=1) #[y1, x, y2, x]
                        fragment.add_projection_edge_pairs(edge_pairs)   # Add projection fragments corresponding to one target edge
                    # Corner segments and uniform segments
                    if(abs(edge[i-1][0])<lengthMin):
                        unifrag_count = (abs(edge[i-1][0])) // lengthNormal
                        start_points = torch.stack([p1+ i*torch.tensor([lengthNormal, 0]) for i in range(unifrag_count+1)]) if p1[0] < p2[0] else torch.stack([p1 - i*torch.tensor([lengthNormal, 0]) for i in range(unifrag_count+1)])
                        end_points = start_points[1:, :].clone()
                        end_points[:,0] = end_points[:,0] - 1 if p1[0] < p2[0] else end_points[:,0] + 1
                        end_points = torch.cat((end_points, p2.unsqueeze(0)), dim=0) #(n,2)
                        edge_pairs = torch.cat((start_points, end_points), dim=1)  #(n,4)
                        if edge_pairs[-1][0] == edge_pairs[-1][2]:
                            edge_pairs = edge_pairs.narrow(0, 0, edge_pairs.shape[0] - 1)
                        fragment.add_normal_edge_pairs(edge_pairs)# Add Normal fragments corresponding to one target edge
                        # normal_fragments.append(edge_pairs)
                    elif (lengthMin <= abs(edge[i-1][0])):
                        #First consider corner small fragments 
                        corner_start1 = torch.cat((p1, torch.tensor([p1[0]+lengthCorner-1, p1[1]]))) if p1[0] < p2[0] else torch.cat((p1, torch.tensor([p1[0]-lengthCorner+1, p1[1]]))) #torch.size([4])
                        corner_end1 = torch.cat((torch.tensor([p2[0] - lengthCorner + 1, p2[1]]), p2)) if p1[0] < p2[0] else torch.cat((torch.tensor([p2[0]+lengthCorner -1, p2[1]]), p2)) 
                        edge_pairs = torch.stack((corner_start1, corner_end1)) # (2,4)
                        fragment.add_corner_edge_pairs(edge_pairs)# Add Corner fragments corresponding to one target edge
                        #Then consider uniform fragments between corner fragments
                        unifrag_count = (abs(edge[i-1][0]) - 2*lengthCorner) // lengthNormal
                        start_points = torch.stack([torch.tensor([p1[0]+ lengthCorner, p1[1]]) + i*torch.tensor([lengthNormal, 0]) for i in range(unifrag_count + 1)]) if p1[0] < p2[0] else torch.stack([torch.tensor([p1[0] - lengthCorner, p1[1]]) - i*torch.tensor([lengthNormal, 0]) for i in range(unifrag_count+1)])
                        end_points = start_points[1:,:].clone()
                        end_points[:,0] = end_points[:,0] - 1 if p1[0] < p2[0] else end_points[:,0] + 1
                        end_points = torch.cat((end_points, torch.tensor([p2[0] - lengthCorner, p2[1]]).unsqueeze(0)), dim=0) if p1[0]<p2[0] else torch.cat((end_points, torch.tensor([p2[0] + lengthCorner, p2[1]]).unsqueeze(0)), dim=0)
                        edge_pairs = torch.cat((start_points, end_points), dim=1)
                        if edge_pairs[-1][0] == edge_pairs[-1][2]:
                            edge_pairs = edge_pairs.narrow(0, 0, edge_pairs.shape[0] - 1)
                        fragment.add_normal_edge_pairs(edge_pairs) # Add Normal fragments corresponding to one target edge
                    self._fragments_v.append(fragment)
                elif edge[i-1][0] == 0: # horizontal
                    fragment = FragmentsOneEdge(direction ='right' if p2[1] > p1[1] else 'left', ref_edge =torch.cat((p1,p2)))
                    # Projection segment
                    if p2[1]>p1[1]: # means the upper side of edge is inside the target, rightward edge
                        target_slice = target[p1[0]+1: p1[0]+projection+1, p1[1]: p2[1]]                       
                    elif p1[1]>p2[1]: # means the lower side of edge is inside the target, leftward edge
                        target_slice = target[p1[0]-projection: p1[0], p2[1]: p1[1]]
                    if target_slice.any().item():
                        if p2[1]>p1[1]:
                            projection_edge_row_index = torch.nonzero(target_slice.any(dim=1))[0]
                        elif p1[1]>p2[1]:
                            projection_edge_row_index = torch.nonzero(target_slice.any(dim=1))[-1]
                        proj_edge_row = target_slice[projection_edge_row_index, :]
                        padded_row = torch.flatten(torch.cat((torch.zeros((1, 1)), proj_edge_row, torch.zeros((1, 1))),dim=0))
                        start_indices = torch.where(torch.diff(padded_row)==1)[0]
                        end_indices = torch.where(torch.diff(padded_row)==-1)[0]-1
                        edge_pairs = torch.stack((torch.full_like(start_indices, p1[0]), min(p1[1], p2[1])+start_indices, torch.full_like(start_indices, p1[0]), min(p1[1], p2[1])+end_indices)) #[y, x1, y, x2]
                        fragment.add_projection_edge_pairs(edge_pairs)# Add projection fragments corresponding to one target edge
                    # Corner segments and uniform segments
                    if(abs(edge[i-1][1])<lengthMin):
                        unifrag_count = (abs(edge[i-1][1])) // lengthNormal
                        start_points = torch.stack([p1+ i*torch.tensor([0, lengthNormal]) for i in range(unifrag_count+1)]) if p1[1] < p2[1] else torch.stack([p1 - i*torch.tensor([0, lengthNormal]) for i in range(unifrag_count+1)])
                        end_points = start_points[1:, :].clone()
                        end_points[:,1] = end_points[:,1] - 1 if p1[1] < p2[1] else end_points[:,1] + 1
                        end_points = torch.cat((end_points, p2.unsqueeze(0)), dim=0)
                        edge_pairs = torch.cat((start_points, end_points), dim=1)
                        if edge_pairs[-1][1] == edge_pairs[-1][3]:
                            edge_pairs = edge_pairs.narrow(0, 0, edge_pairs.shape[0] - 1)
                        fragment.add_normal_edge_pairs(edge_pairs) # Add projection fragments corresponding to one target edge
                    elif (lengthMin <= abs(edge[i-1][1])):
                        # First consider corner small fragments 
                        corner_start1 = torch.cat((p1, torch.tensor([p1[0], p1[1]+lengthCorner-1]))) if p1[1] < p2[1] else torch.cat((p1, torch.tensor([p1[0], p1[1]-lengthCorner+1])))
                        corner_end1 = torch.cat((torch.tensor([p2[0], p2[1]-lengthCorner + 1]), p2)) if p1[1] < p2[1] else torch.cat((torch.tensor([p2[0], p2[1]+lengthCorner-1]), p2))
                        edge_pairs = torch.stack((corner_start1, corner_end1))
                        fragment.add_corner_edge_pairs(edge_pairs)# Add projection fragments corresponding to one target edge
                        # Then consider uniform fragments between corner fragments
                        unifrag_count = (abs(edge[i-1][1]) - 2*lengthCorner) // lengthNormal
                        start_points = torch.stack([torch.tensor([p1[0], p1[1]+ lengthCorner])+ i* torch.tensor([0, lengthNormal]) for i in range(unifrag_count+1)]) if p1[1] < p2[1] else torch.stack([torch.tensor([p1[0], p1[1]+ lengthCorner]) - i*torch.tensor([0, lengthNormal]) for i in range(unifrag_count+1)])
                        end_points = start_points[1:,:].clone()
                        end_points[:,1] = end_points[:,1] - 1 if p1[1] < p2[1] else end_points[:,1] + 1
                        end_points = torch.cat((end_points, torch.tensor([p2[0], p2[1]- lengthCorner]).unsqueeze(0)), dim=0) if p1[1]<p2[1] else torch.cat((end_points, torch.tensor([p2[0], p2[1]+lengthCorner]).unsqueeze(0)), dim=0)
                        edge_pairs = torch.cat((start_points, end_points),dim=1)
                        if edge_pairs[-1][1] == edge_pairs[-1][3]:
                            edge_pairs = edge_pairs.narrow(0, 0, edge_pairs.shape[0] - 1)
                        fragment.add_normal_edge_pairs(edge_pairs) # Add projection fragments corresponding to one target edge)
                    self._fragments_h.append(fragment)
                    print(fragment.ref_edge())
                p1 = p2
    
    # def maskedge_smooth(self, gap_threshold):
    #     '''
    #     The perpendicular gap between neighbouring fragments should be no larger than the threshold
    #     '''
    #     pass

        

    def update_fragments(self, projection_step, corner_step, normal_step, nominalImage, outerImage=None, innerImage=None):
        """
        priority_order means the relative order of consideration, pcn: projection->corner->normal
        """
        for edge_h in self._fragments_h:
            if edge_h.direction() == 'right':
                #Normal edge 
                normal_pairs = edge_h._normal_edge_pairs
                ref_points = torch.stack((normal_pairs[:,0], (normal_pairs[:,1]+normal_pairs[:,3])//2), dim=1)
                image_outer = nominalImage[ref_points[:,0]+6, ref_points[:,1]]
                image_inner = nominalImage[ref_points[:,0]-6, ref_points[:,1]]
                edge_h._normal_offsets[image_outer >= 0.5] -= normal_step
                edge_h._normal_offsets[image_inner <= 0.5] += normal_step
                #Corner edge
                if edge_h._corner_edge_pairs != None:
                    corner_pairs = edge_h._corner_edge_pairs
                    ref_points = torch.stack((corner_pairs[:,0],(corner_pairs[:,1]+corner_pairs[:,3])//2), dim=1)
                    image_outer = nominalImage[ref_points[:,0]+6, ref_points[:,1]]
                    image_inner = nominalImage[ref_points[:,0]-6, ref_points[:,1]]
                    edge_h._corner_offsets[image_outer >= 0.5] -= corner_step
                    edge_h._corner_offsets[image_inner <= 0.5] += corner_step
                #Projection edge
                if edge_h._projection_edge_pairs != None:
                    projection_pairs = edge_h._projection_edge_pairs
                    ref_points = torch.stack((projection_pairs[:,0], (projection_pairs[:,1]+projection_pairs[:,3])//2), dim=1)
                    image_outer = nominalImage[ref_points[:,0]+6, ref_points[:,1]]
                    image_inner = nominalImage[ref_points[:,0]-6, ref_points[:,1]]
                    edge_h._projection_offsets[image_outer >= 0.5] -= projection_step
                    edge_h._projection_offsets[image_inner <= 0.5] += projection_step
                edge_h.smooth(edge_gap= 6)

            elif edge_h.direction() == 'left':
                #Normal edge 
                normal_pairs = edge_h._normal_edge_pairs
                ref_points = torch.stack((normal_pairs[:,0], (normal_pairs[:,1]+normal_pairs[:,3])//2), dim=1)
                image_outer = nominalImage[ref_points[:,0]-6, ref_points[:,1]]
                image_inner = nominalImage[ref_points[:,0]+6, ref_points[:,1]]
                edge_h._normal_offsets[image_outer >= 0.5] += normal_step # image expand
                edge_h._normal_offsets[image_inner <= 0.5] -= normal_step # image shrink
                #Corner edge
                if edge_h._corner_edge_pairs != None:
                    corner_pairs = edge_h._corner_edge_pairs
                    ref_points = torch.stack((corner_pairs[:,0],(corner_pairs[:,1]+corner_pairs[:,3])//2), dim=1)
                    image_outer = nominalImage[ref_points[:,0]-6, ref_points[:,1]]
                    image_inner = nominalImage[ref_points[:,0]+6, ref_points[:,1]]
                    edge_h._corner_offsets[image_outer >= 0.5] += corner_step # image expand 
                    edge_h._corner_offsets[image_inner <= 0.5] -= corner_step # image shrink
                #Projection edge
                if edge_h._projection_edge_pairs != None:
                    projection_pairs = edge_h._projection_edge_pairs
                    ref_points = torch.stack((projection_pairs[:,0], (projection_pairs[:,1]+projection_pairs[:,3])//2), dim=1)
                    image_outer = nominalImage[ref_points[:,0]-6, ref_points[:,1]]
                    image_inner = nominalImage[ref_points[:,0]+6, ref_points[:,1]]
                    edge_h._projection_offsets[image_outer >= 0.5] += projection_step
                    edge_h._projection_offsets[image_inner <= 0.5] -= projection_step
                edge_h.smooth(edge_gap= 6)

        for edge_v in self._fragments_v:
            if edge_v.direction() == 'down':
                #Normal edge 
                normal_pairs = edge_v._normal_edge_pairs
                ref_points = torch.stack(((normal_pairs[:,0]+normal_pairs[:,2])//2, normal_pairs[:,1]), dim=1)
                image_outer = nominalImage[ref_points[:,0], ref_points[:,1]-6]
                image_inner = nominalImage[ref_points[:,0], ref_points[:,1]+6]
                edge_v._normal_offsets[image_outer >= 0.5] += normal_step 
                edge_v._normal_offsets[image_inner <= 0.5] -= normal_step
                #Corner edge
                if edge_v._corner_edge_pairs != None:
                    corner_pairs = edge_v._corner_edge_pairs
                    ref_points = torch.stack(((corner_pairs[:,0]+corner_pairs[:,2])//2, corner_pairs[:,1]), dim=1)
                    image_outer = nominalImage[ref_points[:,0], ref_points[:,1]-6]
                    image_inner = nominalImage[ref_points[:,0], ref_points[:,1]+6]
                    edge_v._corner_offsets[image_outer >= 0.5] += corner_step 
                    edge_v._corner_offsets[image_inner <= 0.5] -= corner_step
                #Projection edge
                if edge_v._projection_edge_pairs != None:
                    projection_pairs = edge_v._projection_edge_pairs
                    ref_points = torch.stack(((projection_pairs[:,0]+projection_pairs[:,2])//2, projection_pairs[:,1]), dim=1)
                    image_outer = nominalImage[ref_points[:,0], ref_points[:,1]-6]
                    image_inner = nominalImage[ref_points[:,0], ref_points[:,1]+6]
                    edge_v._projection_offsets[image_outer >= 0.5] += projection_step 
                    edge_v._projection_offsets[image_inner <= 0.5] -= projection_step
                edge_v.smooth(edge_gap= 6)

            elif edge_v.direction() == 'up':
                #Normal edge 
                normal_pairs = edge_v._normal_edge_pairs
                ref_points = torch.stack(((normal_pairs[:,0]+normal_pairs[:,2])//2, normal_pairs[:,1]), dim=1)
                image_outer = nominalImage[ref_points[:,0], ref_points[:,1]+6]
                image_inner = nominalImage[ref_points[:,0], ref_points[:,1]-6]
                edge_v._normal_offsets[image_outer >= 0.5] -= normal_step 
                edge_v._normal_offsets[image_inner <= 0.5] += normal_step
                #Corner edge
                if edge_v._corner_edge_pairs != None:
                    corner_pairs = edge_v._corner_edge_pairs
                    ref_points = torch.stack(((corner_pairs[:,0]+corner_pairs[:,2])//2, corner_pairs[:,1]), dim=1)
                    image_outer = nominalImage[ref_points[:,0], ref_points[:,1]+6]
                    image_inner = nominalImage[ref_points[:,0], ref_points[:,1]-6]
                    edge_v._corner_offsets[image_outer >= 0.5] -= corner_step 
                    edge_v._corner_offsets[image_inner <= 0.5] += corner_step
                #Projection edge
                if edge_v._projection_edge_pairs != None:
                    projection_pairs = edge_v._projection_edge_pairs
                    ref_points = torch.stack(((projection_pairs[:,0]+projection_pairs[:,2])//2, projection_pairs[:,1]), dim=1)
                    image_outer = nominalImage[ref_points[:,0], ref_points[:,1]+6]
                    image_inner = nominalImage[ref_points[:,0], ref_points[:,1]-6]
                    edge_v._projection_offsets[image_outer >= 0.5] -= projection_step 
                    edge_v._projection_offsets[image_inner <= 0.5] += projection_step
                edge_v.smooth(edge_gap= 6)
                
    def updateMask(self):
        mask = self._mask.clone()
        for edge_h in self._fragments_h:
            if edge_h.direction() == 'right':
                normal_pairs = edge_h._normal_edge_pairs
                normal_offsets = edge_h._normal_offsets
                assert normal_pairs.shape[0] == normal_offsets.shape[0]
                for i in range(normal_offsets.shape[0]):
                    if normal_offsets[i] > 0.01: #expand
                        mask[normal_pairs[i,0]+1 : normal_pairs[i,0]+normal_offsets[i]+1, normal_pairs[i,1] : normal_pairs[i,3]+1] = 1
                    elif normal_offsets[i] <= -0.01:  # shrink
                        mask[normal_pairs[i,0]+normal_offsets[i]+1 : normal_pairs[i,0]+1, normal_pairs[i,1] : normal_pairs[i,3]+1] = 0
                if edge_h._corner_edge_pairs != None:
                    corner_pairs = edge_h._corner_edge_pairs
                    corner_offsets = edge_h._corner_offsets
                    assert corner_pairs.shape[0] == corner_offsets.shape[0]
                    for i in range(corner_offsets.shape[0]):
                        if corner_offsets[i] > 0.01:
                            mask[corner_pairs[i,0]+1 : corner_pairs[i,0]+corner_offsets[i]+1, corner_pairs[i,1] : corner_pairs[i,3]+1] = 1
                        elif corner_offsets[i]<=-0.01:
                            mask[corner_pairs[i,0]+corner_offsets[i]+1 : corner_pairs[i,0]+1, corner_pairs[i,1] : corner_pairs[i,3]+1] = 0
                #Projection edge
                if edge_h._projection_edge_pairs != None:
                    projection_pairs = edge_h._projection_edge_pairs
                    projection_offsets = edge_h._projection_offsets
                    outermost=torch.max(torch.max(corner_offsets), torch.max(normal_offsets))
                    innermost=torch.min(torch.min(corner_offsets), torch.min(normal_offsets))
                    outermost = 20
                    innermost = 15
                    assert projection_pairs.shape[0]==projection_offsets.shape[0]
                    for i in range(projection_offsets.shape[0]):
                        # recover the original patterns
                        mask[projection_pairs[i,0]+1 : projection_pairs[i,0]+1+outermost, projection_pairs[i,1] : projection_pairs[i,3]+1]=0
                        mask[projection_pairs[i,0]-innermost : projection_pairs[i,0], projection_pairs[i,1] : projection_pairs[i,3]+1] = 1
                        if projection_offsets[i] > 0.01:
                            mask[projection_pairs[i,0]+1 : projection_pairs[i,0]+projection_offsets[i]+1, projection_pairs[i,1] : projection_pairs[i,3]+1] = 1
                        elif projection_offsets[i] <= -0.01:
                            mask[projection_pairs[i,0]+projection_offsets[i]+1 : projection_pairs[i,0]+1, projection_pairs[i,1] : projection_pairs[i,3]+1] = 0                
            elif edge_h.direction() == 'left':
                normal_pairs = edge_h._normal_edge_pairs
                normal_offsets = edge_h._normal_offsets
                assert normal_pairs.shape[0] == normal_offsets.shape[0]
                for i in range(normal_offsets.shape[0]):
                    if normal_offsets[i] > 0.01:
                        mask[normal_pairs[i,0]-normal_offsets[i] : normal_pairs[i,0], normal_pairs[i,3] : normal_pairs[i, 1]+1] = 1
                    elif normal_offsets[i] < -0.01:
                        mask[normal_pairs[i,0] : normal_pairs[i,0]-normal_offsets[i], normal_pairs[i,3] : normal_pairs[i, 1]+1] = 0
                if edge_h._corner_edge_pairs != None:
                    corner_pairs = edge_h._corner_edge_pairs
                    corner_offsets = edge_h._corner_offsets
                    assert corner_pairs.shape[0] == corner_offsets.shape[0]    
                    for i in range(corner_offsets.shape[0]):
                        if corner_offsets[i] > 0.01:
                            mask[corner_pairs[i,0]-corner_offsets[i] : corner_pairs[i,0], corner_pairs[i,3] : corner_pairs[i, 1]+1] = 1
                        elif corner_offsets[i] < -0.01:
                            mask[corner_pairs[i,0] : corner_pairs[i,0]-corner_offsets[i], corner_pairs[i,3] : corner_pairs[i, 1]+1] = 0    
                if edge_h._projection_edge_pairs != None:
                    projection_pairs = edge_h._projection_edge_pairs
                    projection_offsets = edge_h._projection_offsets
                    outermost=torch.max(torch.max(corner_offsets), torch.max(normal_offsets))
                    innermost=torch.abs(torch.min(torch.min(corner_offsets), torch.min(normal_offsets)))
                    outermost = 20
                    innermost = 15
                    assert projection_pairs.shape[0]==projection_offsets.shape[0]
                    for i in range(projection_offsets.shape[0]):
                        # recover the original patterns
                        mask[projection_pairs[i,0]-outermost : projection_pairs[i,0], projection_pairs[i,3] : projection_pairs[i,1]+1]=0
                        mask[projection_pairs[i,0] : projection_pairs[i,0]+innermost, projection_pairs[i,3] : projection_pairs[i,1]+1] = 1
                    if projection_offsets[i] > 0.01:
                        mask[projection_pairs[i,0]-projection_offsets[i] : projection_pairs[i,0], projection_pairs[i,3] : projection_pairs[i,1]+1] = 1
                    elif projection_offsets[i] < -0.01:
                        mask[projection_pairs[i,0] : projection_pairs[i,0]-projection_offsets[i], projection_pairs[i,3] : projection_pairs[i,1]+1] = 0 
        for edge_v in self._fragments_v:
            if edge_v.direction() == 'down':
                normal_pairs = edge_v._normal_edge_pairs
                normal_offsets = edge_v._normal_offsets
                assert normal_pairs.shape[0] == normal_offsets.shape[0]
                if normal_offsets[i] > 0.01:
                    mask[normal_pairs[i,0] : normal_pairs[i,2]+1, normal_pairs[i,1]-normal_offsets[i] : normal_pairs[i,1]] = 1
                elif normal_offsets[i] < -0.01:
                    mask[normal_pairs[i,0] : normal_pairs[i,2]+1, normal_pairs[i,1] : normal_pairs[i,1]-normal_offsets[i]] = 0
                if edge_v._corner_edge_pairs != None:
                    corner_pairs = edge_v._corner_edge_pairs
                    corner_offsets = edge_v._corner_offsets
                    assert corner_pairs.shape[0] == corner_offsets.shape[0] 
                    if corner_offsets[i] > 0.01:
                        mask[corner_pairs[i,0] : corner_pairs[i,2]+1, corner_pairs[i,1]-corner_offsets[i] : corner_pairs[i,1]] = 1
                    elif normal_offsets[i] < -0.01:
                        mask[corner_pairs[i,0] : corner_pairs[i,2]+1, corner_pairs[i,1] : corner_pairs[i,1]-corner_offsets[i]] = 0                
                if edge_v._projection_edge_pairs != None:
                    projection_pairs = edge_v._projection_edge_pairs
                    projection_offsets = edge_v._projection_offsets
                    outermost=torch.max(torch.max(corner_offsets), torch.max(normal_offsets))
                    innermost=torch.abs(torch.min(torch.min(corner_offsets), torch.min(normal_offsets)))
                    outermost = 20
                    innermost = 15
                    assert projection_pairs.shape[0]==projection_offsets.shape[0]
                    for i in range(projection_offsets.shape[0]):
                        # recover the original patterns
                        mask[projection_pairs[i,0] : projection_pairs[i,2]+1, projection_pairs[i,1]-outermost : projection_pairs[i,1]] = 0
                        mask[projection_pairs[i,0] : projection_pairs[i,2]+1, projection_pairs[i,1] : projection_pairs[i,1]+innermost] = 1
                        if projection_offsets[i] > 0.01:
                            mask[projection_pairs[i,0] : projection_pairs[i,2]+1, projection_pairs[i,1]-projection_offsets[i] : projection_pairs[i,1]] = 1
                        elif projection_offsets[i] <= -0.01:
                            mask[projection_pairs[i,0] : projection_pairs[i,2]+1, projection_pairs[i,1] : projection_pairs[i,1]-projection_offsets[i]] = 0
            if edge_v.direction() == 'up':
                normal_pairs = edge_v._normal_edge_pairs
                normal_offsets = edge_v._normal_offsets
                assert normal_pairs.shape[0] == normal_offsets.shape[0]
                if normal_offsets[i] > 0.01:
                    mask[normal_pairs[i,2] : normal_pairs[i,0]+1, normal_pairs[i,1]+1 : normal_pairs[i,1]+1+normal_offsets[i]] = 1
                elif normal_offsets[i] < -0.01:
                    mask[normal_pairs[i,2] : normal_pairs[i,0]+1, normal_pairs[i,1]-normal_offsets[i] : normal_pairs[i,1]] = 0
                if edge_v._corner_edge_pairs != None:
                    corner_pairs = edge_v._corner_edge_pairs
                    corner_offsets = edge_v._corner_offsets
                    assert corner_pairs.shape[0] == corner_offsets.shape[0] 
                    if corner_offsets[i] > 0.01:
                        mask[corner_pairs[i,2] : corner_pairs[i,0]+1, corner_pairs[i,1]+1 : corner_pairs[i,1]+1+corner_offsets[i]] = 1
                    elif normal_offsets[i] < -0.01:
                        mask[corner_pairs[i,2] : corner_pairs[i,0]+1, corner_pairs[i,1]-corner_offsets[i] : corner_pairs[i,1]] = 0                
                if edge_v._projection_edge_pairs != None:
                    projection_pairs = edge_v._projection_edge_pairs
                    projection_offsets = edge_v._projection_offsets
                    outermost=torch.max(torch.max(corner_offsets), torch.max(normal_offsets))
                    innermost=torch.abs(torch.min(torch.min(corner_offsets), torch.min(normal_offsets)))
                    outermost = 20
                    innermost = 15
                    assert projection_pairs.shape[0]==projection_offsets.shape[0]
                    for i in range(projection_offsets.shape[0]):
                        # recover the original patterns
                        mask[projection_pairs[i,2] : projection_pairs[i,0]+1, projection_pairs[i,1]+1 : projection_pairs[i,1]+1+projection_offsets[i]] = 0
                        mask[projection_pairs[i,2] : projection_pairs[i,0]+1, projection_pairs[i,1]-innermost : projection_pairs[i,1]] = 1
                        if projection_offsets[i] > 0.01:
                            mask[projection_pairs[i,0] : projection_pairs[i,2]+1, projection_pairs[i,1]-projection_offsets[i] : projection_pairs[i,1]] = 1
                        elif projection_offsets[i] <= -0.01:
                            mask[projection_pairs[i,0] : projection_pairs[i,2]+1, projection_pairs[i,1] : projection_pairs[i,1]-projection_offsets[i]] = 0
        return mask


                


# if __name__ == "__main__":
#     SCALE = 1
#     l2s = []
#     pvbs = []
#     epes = []
#     shots = []
#     runtimes = []
#     cfg   = MbOPCCfg("./config/mbopc2048.txt")
#     litho = lithosim.LithoSim("./config/lithosimple.txt")
#     # for i in range(1):
#     design = glp.Design(f"/data/zyyu/GDSfile/gcd_45nm_clip/layer11/glp_ver/cropped_10240-66560.glp", down=10)
#     design.center(cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)
#     target, params = initializer.PixelInit().run(design, cfg["TileSizeX"]*SCALE, cfg["TileSizeY"]*SCALE, cfg["OffsetX"]*SCALE, cfg["OffsetY"]*SCALE)    
#     pos = Mask(target)
#     pos.find_contour()
#     pos.fragment_edge(20,10,30,50)
#     pos.add_sraf(30, 30, 150, 200, 20,20,10, 0.8,0.6,0.4)