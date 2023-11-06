def fragment_edge(polygon_set, np_target:np.ndarray, projection, lengthCorner, lengthNormal, lengthMin, lengthMax):
    frag_points = [] # Each element store all segments from one polygon.
    for polygon in polygon_set:
        edge = polygon[1:] - polygon[:-1]  #Distane-1
        legal_mask = (edge ==0).any(axis=1) #If mask value is true, the corresponding distances is h/v edge distance
        p1 = polygon[0]
        frag_points.append(p1)
        for i in range(1, len(polygon)):        
            p2 = polygon[i]
            projection_fragments = []
            corner_frangments = []
            normal_fragments = []
            #Find Projection part 
            if edge[i-1][1] == 0:  # vertical 
                # Projection segment
                if p2[0] > p1[0]: #means the right side of edge is inside the target, downward edge
                    target_slice = np_target[min(p1[0], p2[0]) : max(p1[0],p2[0]), p1[1]-projection : p1[1]]
                    proj_edge_col_index = np.nonzero(target_slice.any(axis=0))[-1]
                elif p1[0] > p2[0]: # means the left side of the edge is inside the target, upward edge
                    target_slice = np_target[min(p1[0], p2[0]) : max(p1[0],p2[0]) , p1[1]+1 : p1[1]+projection+1]
                    proj_edge_col_index = np.nonzero(target_slice.any(axis=0))[0]
                proj_edge_col = target_slice[:, proj_edge_col_index]
                padded_col = np.insert(np.append(proj_edge_col, 0), 0, 0)
                start_indices = np.where(np.diff(padded_col)==1)[0]
                end_indices = np.where(np.diff(padded_col)==-1)[0]-1
                edge_pairs = np.column_stack((min(p1[0], p2[0])+start_indices, np.full_like(start_indices, p1[1]), min(p1[0], p2[0])+end_indices, np.full_like(end_indices, p1[1]))) #[y1, x, y2, x]
                projection_fragments.append(edge_pairs) # Add projection fragments corresponding to one target edge
                #Corner segments and uniform segments
                if(abs(edge[i-1][0])<lengthMin):
                    unifrag_count = (abs(edge[i-1][0])) // lengthNormal
                    start_points = np.array([p1+ i*np.array([lengthNormal, 0]) for i in range(unifrag_count+1)]) if p1[0] < p2[0] else np.array([p1 - i*np.array([lengthNormal, 0]) for i in range(unifrag_count+1)])
                    end_points = start_points[1:, :].copy()
                    end_points[:,0] = end_points[:,0] - 1 if p1[0] < p2[0] else end_points[:,0] + 1
                    end_points = np.append(end_points, [p2], axis=0)
                    edge_pairs = np.column_stack((start_points, end_points))
                    if edge_pairs[-1][0] == edge_pairs[-1][2]:
                        edge_pairs = np.delete(edge_pairs, -1, axis = 0)
                    normal_fragments.append(edge_pairs)
                elif (lengthMin <= abs(edge[i-1][0]) < lengthMax):
                    #First consider corner small fragments 
                    corner_start1 = np.append(p1, [p1[0]+lengthCorner-1, p1[1]]) if p1[0] < p2[0] else np.append(p1, [p1[0]-lengthCorner+1, p1[1]])
                    corner_end1 = np.append([p2[0] - lengthCorner + 1, p2[1]], p2) if p1[0] < p2[0] else np.append([p2[0]+lengthCorner -1, p2[1]], p2)
                    corner_frangments.append(np.column_stack((corner_start1, corner_end1)))
                    #Then consider uniform fragments between corner fragments
                    unifrag_count = (abs(edge[i-1][0]) - 2*lengthCorner) // lengthNormal
                    start_points = np.array([[p1[0]+ lengthCorner, p1[1]] + i*np.array([lengthNormal, 0]) for i in range(unifrag_count + 1)]) if p1[0] < p2[0] else np.array([[p1[0] - lengthCorner, p1[1]] - i*np.array([lengthNormal, 0]) for i in range(unifrag_count+1)])
                    end_points = start_points[1:,:].copy()
                    end_points[:,0] = end_points[:,0] - 1 if p1[0] < p2[0] else end_points[:,0] + 1
                    end_points = np.append(end_points, [[p2[0] - lengthCorner, p2[1]]], axis=0) if p1[0]<p2[0] else np.append(end_points, [[p2[0] + lengthCorner, p2[1]]], axis=0)
                    edge_pairs = np.column_stack((start_points, end_points))
                    if edge_pairs[-1][0] == edge_pairs[-1][2]:
                        edge_pairs = np.delete(edge_pairs, -1, axis = 0)
                    normal_fragments.append(edge_pairs)

            elif edge[i-1][0] == 0: #horizontal
                # Projection segment
                if p2[1]>p1[1]:#means the upper side of edge is inside the target, rightward edge
                    target_slice = np_target[p1[0]+1: p1[0]+projection+1, p1[1]: p2[1]]
                    projection_edge_row_index = np.nonzero(target_slice.any(axis=1))[0]
                elif p1[1]>p2[1]: #means the lower side of edge is inside the target, leftward edge
                    target_slice = np_target[p1[0]-projection: p1[0], p2[1]: p1[1]]
                    projection_edge_row_index = np.nonzero(target_slice.any(axis=1))[-1]
                proj_edge_row = target_slice[projection_edge_row_index, :]
                padded_row = np.insert(np.append(proj_edge_row, 0), 0, 0)
                start_indices = np.where(np.diff(padded_row)==1)[0]
                end_indices = np.where(np.diff(padded_row)==-1)[0]-1
                edge_pairs = np.column_stack((np.full_like(start_indices, p1[0]), min(p1[1], p2[1])+start_indices, np.full_like(start_indices, p1[0]), min(p1[1], p2[1])+end_indices)) #[y, x1, y, x2]
                projection_fragments.append(edge_pairs) # Add projection fragments corresponding to one target edge
                #Corner segments and uniform segments
                if(abs(edge[i-1][1])<lengthMin):
                    unifrag_count = (abs(edge[i-1][1])) // lengthNormal
                    start_points = np.array([p1+ i*np.array([0, lengthNormal]) for i in range(unifrag_count+1)]) if p1[1] < p2[1] else np.array([p1 - i*np.array([0, lengthNormal]) for i in range(unifrag_count+1)])
                    end_points = start_points[1:, :].copy()
                    end_points[:,1] = end_points[:,1] - 1 if p1[1] < p2[1] else end_points[:,1] + 1
                    end_points = np.append(end_points, [p2], axis=0)
                    edge_pairs = np.column_stack((start_points, end_points))
                    if edge_pairs[-1][1] == edge_pairs[-1][3]:
                        edge_pairs = np.delete(edge_pairs, -1, axis = 0)
                    normal_fragments.append(edge_pairs)
                elif (lengthMin <= abs(edge[i-1][0]) < lengthMax):
                    #First consider corner small fragments 
                    corner_start1 = np.append(p1, [p1[0], p1[1]+lengthCorner-1]) if p1[1] < p2[1] else np.append(p1, [p1[0], p1[1]-lengthCorner+1])
                    corner_end1 = np.append([p2[0], p2[1]-lengthCorner + 1], p2) if p1[1] < p2[1] else np.append([p2[0], p2[1]+lengthCorner-1], p2)
                    corner_frangments.append(np.column_stack((corner_start1, corner_end1)))
                    #Then consider uniform fragments between corner fragments
                    unifrag_count = (abs(edge[i-1][1]) - 2*lengthCorner) // lengthNormal
                    start_points = np.array([[p1[0], p1[1]+lengthCorner] + i*np.array([0, lengthNormal]) for i in range(unifrag_count + 1)]) if p1[1] < p2[1] else np.array([[p1[0], p1[1]-lengthCorner] - i*np.array([0, lengthNormal]) for i in range(unifrag_count+1)])
                    end_points = start_points[1:,:].copy()
                    end_points[:,0] = end_points[:,1] - 1 if p1[1] < p2[1] else end_points[:,1] + 1
                    end_points = np.append(end_points, [[p2[0], p2[1]- lengthCorner]], axis=0) if p1[1]<p2[1] else np.append(end_points, [[p2[0], p2[1]+lengthCorner]], axis=0)
                    edge_pairs = np.column_stack((start_points, end_points))
                    if edge_pairs[-1][1] == edge_pairs[-1][3]:
                        edge_pairs = np.delete(edge_pairs, -1, axis = 0)
                    normal_fragments.append(edge_pairs)
            p1 = p2
    return frag_points