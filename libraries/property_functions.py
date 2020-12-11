#Defines functions that will be used to compute surface properties used as input for machine-learned models

import numpy as np

def _min_image_distance(pos1, pos2, boxl):
    rij = pos2-pos1
    rij = rij - boxl*np.rint(rij/boxl)
    return rij

def index_to_coord(i, j):
    y = j*np.sqrt(3)/2.
    if ((j%2) == 0):
        x = i
    else:
        x = i+0.5
    return x, y

def gr(pos, boxl):
    n = len(pos)
    l_over_2_max = np.min(boxl)/2.
    dr = 0.25
    bin_edges = np.arange(0, l_over_2_max, dr)
    bin_centers = bin_edges[:-1]+dr/2.
    bins = np.zeros(len(bin_edges)-1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pos1 = pos[i, :]
            pos2 = pos[j, :]
            rij = _min_image_distance(pos1, pos2, boxl)
            r = np.sqrt(np.sum(rij*rij))
            i_bin = np.digitize(r, bin_edges)
            if i_bin < len(bin_edges):
                bins[i_bin-1] += 1
    vol_bin = (bin_edges[1:]**2-bin_edges[:-1]**2)*np.pi 
    n_ig = vol_bin * n / (boxl[0]*boxl[1])
    return bin_centers, bins/n_ig/(n-1)

#Check about boxl thing... not sure if necessary?  Did I get rid of hard-coding right?
def gr_ch3(struct):
    structure_array = struct.makeCartoon(rows=6, cols=8)
    boxl = np.array([structure_array.shape[0], 0.5*np.sqrt(3)*structure_array.shape[1]]) 
    coords_all_x = np.zeros(structure_array.shape)
    coords_all_y = np.zeros(structure_array.shape)
    for i in range(structure_array.shape[0]):
        for j in range(structure_array.shape[1]):
            coords_all_x[i, j], coords_all_y[i, j] = index_to_coord(i, j)
    ch3_indices = np.nonzero(1-structure_array)
    coords_ch3_x = coords_all_x[ch3_indices]
    coords_ch3_y = coords_all_y[ch3_indices]
    coords_ch3 = np.hstack((np.reshape(coords_ch3_x, (len(coords_ch3_x), 1)), 
                            np.reshape(coords_ch3_y, (len(coords_ch3_y), 1))))
    bin_centers, gr_ch3 = gr(coords_ch3, boxl)
    return gr_ch3

def gr_oh(struct):
    structure_array = struct.makeCartoon(rows=6, cols=8)
    boxl = np.array([structure_array.shape[0], 0.5*np.sqrt(3)*structure_array.shape[1]]) 
    coords_all_x = np.zeros(structure_array.shape)
    coords_all_y = np.zeros(structure_array.shape)
    for i in range(structure_array.shape[0]):
        for j in range(structure_array.shape[1]):
            coords_all_x[i, j], coords_all_y[i, j] = index_to_coord(i, j)
    oh_indices = np.nonzero(structure_array)
    coords_oh_x = coords_all_x[oh_indices]
    coords_oh_y = coords_all_y[oh_indices]
    coords_oh = np.hstack((np.reshape(coords_oh_x, (len(coords_oh_x), 1)), 
                           np.reshape(coords_oh_y, (len(coords_oh_y), 1))))
    bin_centers_oh, gr_oh = gr(coords_oh, boxl)
    return gr_oh

def gr_int(struct, typeInt):
    structure_array = struct.makeCartoon(rows=6, cols=8)
    boxl = np.array([structure_array.shape[0], 0.5*np.sqrt(3)*structure_array.shape[1]])
    coords_all_x = np.zeros(structure_array.shape)
    coords_all_y = np.zeros(structure_array.shape)
    for i in range(structure_array.shape[0]):
        for j in range(structure_array.shape[1]):
            coords_all_x[i, j], coords_all_y[i, j] = index_to_coord(i, j)
    type_indices = np.where(structure_array == typeInt)
    coords_type_x = coords_all_x[type_indices]
    coords_type_y = coords_all_y[type_indices]
    coords_type = np.hstack((np.reshape(coords_type_x, (len(coords_type_x), 1)),
                             np.reshape(coords_type_y, (len(coords_type_y), 1))))
    bin_centers, gr_int = gr(coords_type, boxl)
    return gr_int


