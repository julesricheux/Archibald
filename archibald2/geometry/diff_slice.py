# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 14:25:12 2025

@author: jrich
"""

import os
import sys

__root_dir = os.path.dirname(os.path.abspath(__file__))
if __root_dir not in sys.path:
    sys.path.append(os.path.dirname(__root_dir))
    
# import numpy as np

from typing import Union, List
import copy

import archibald2.numpy as np
from archibald2.tools.math_utils import ReLU
from archibald2.geometry.differentiable_mesh import DifferentiableMesh, DifferentiablePolygon

import casadi as ca

np.set_printoptions(precision=2)

#%%

def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


#%% COMPUTING
if __name__=='__main__':
    
    
    
    import trimesh
    
    from archibald2.optimization import Opti
    
    def ReLU(x):
        return x * (x > 0)
    
    def waterplane_rotation(heel, trim):
        """Return 3x3 rotation matrix."""
    
        x0 = np.array([1., 0., 0.])
        y0 = np.array([0., 1., 0.])
        
        Rheel = np.rotation_matrix_3D(np.radians(-heel), x0)
        
        y1 = Rheel @ y0
        
        Rtrim = np.rotation_matrix_3D(np.radians(-trim), y0)
        
        x1 = Rtrim @ x0
        
        z1 = np.cross(x1, y1)
        
        return np.vstack((x1, y1, z1))
    
    
    def build_waterplane(P0, dz, heel, trim):
        
        mat = waterplane_rotation(heel = 40., trim = 3.)
        xn = mat[:,0]
        yn = mat[:,1]
        normal = mat[:,2]
        point = P0 + dz * normal
        
        return point, mat, xn, yn, normal
    
    archibald_root = 'C:/Users/jrich/NEOLINE DEVELOPPEMENT/NeoDev - ND-DÉVELOPPEMENT - Documents/ND-DÉVELOPPEMENT/01_Développement/08_Outils'
    # hullStl = archibald_root+'/archibald-main/Private/n136_data/n136.stl' # path to a STL mesh of the hull
    hullStl = archibald_root+'/archibald-main/Private/n136_data/n136_quad.stl' # path to a STL mesh of the hull
    hullMesh = trimesh.load(hullStl)
    
    diff_mesh = DifferentiableMesh(hullMesh.vertices, hullMesh.faces)
    
    opti = Opti()

    T0 = 5.332476916708436
    
    # T = opti.variable(init_guess=T0, lower_bound=1e-3)
    # T = opti.parameter(T0)
    T = T0
    
    # X0 = np.mean(diff_mesh.vertices, axis=0)[0]
    P0 = np.mean(diff_mesh.vertices, axis=0) * np.array([1., 0., 0.])
    
    # mat = waterplane_normal(P0, dz=T, heel=40., trim=3.)
    point, mat, xn, yn, normal = build_waterplane(P0, dz=T, heel=40., trim=3.)
    
    
    dist = diff_mesh.vertices_distances_to_plane(point, normal)
    
    # edges = diff_mesh.edges
    edges_raw = np.concatenate([
        diff_mesh.faces[:, [0, 1]],
        diff_mesh.faces[:, [1, 2]],
        diff_mesh.faces[:, [2, 0]],
    ], axis=0)
    
    face_indices = np.repeat(np.arange(len(diff_mesh.faces)), 3)  # each face repeated 3 times
    
    # edges_sorted = np.sort(edges, axis=1)  # shape (3*n_faces, 2)
    
    edges_raw = np.sort(edges_raw, axis=1)
    
    # edges, inverse_indices = np.unique(edges_raw, axis=0, return_inverse=True)
    edges = edges_raw
    
    edge_dist = dist[edges]
    edge_prod = np.prod(edge_dist, axis=1)
    edge_sign = np.sign(edge_prod)
    edge_mask = ((1-edge_sign)/2)
    
    
    selected_edges = edges[edge_mask == 1.]
    
    # face_mask = np.sum(edge_mask[inverse_indices].reshape((3,-1)).T, axis=1)//2
    face_mask = np.sum(edge_mask.reshape((3,-1)).T, axis=1)//2 # sliced faces
    
    # Get coordinates of the selected edge endpoints
    v0 = diff_mesh.vertices[selected_edges[:, 0]]  # shape: (n_crossing, 3)
    v1 = diff_mesh.vertices[selected_edges[:, 1]]
    
    v0p = diff_mesh.vertices[edges[:, 0]]  # shape: (n_crossing, 3)
    v1p = diff_mesh.vertices[edges[:, 1]]
    
    # Get distances of the endpoints to the plane
    d0 = edge_dist[edge_mask == 1.][:, 0]
    d1 = edge_dist[edge_mask == 1.][:, 1]
    
    d0p = edge_dist[:, 0]
    d1p = edge_dist[:, 1]
    
    # Compute interpolation factor t (clip to avoid numerical instability)
    t = d0 / (d0 - d1 + 1e-12)
    
    tp = d0p / (d0p - d1p + 1e-12)
    
    # inter_mask = ca.if_else(ca.logic_and(tp > 0, tp < 1), 1.0, np.nan)
    # inter_mask = ca.if_else(ca.logic_and(tp > 0, tp < 1), 1.0, np.nan)
    inter_mask = (tp>0) * (tp<1)
    
    inter_blow = ca.if_else(ca.logic_and(tp > 0, tp < 1), 1.0, np.nan)
    
    # Compute intersection points
    intersections = v0 + tall(t) * (v1 - v0)
    intersections_p = v0p + tall(tp*edge_mask) * (v1p - v0p) # WARNING, approx 100x slower
    
    vertices = diff_mesh.vertices.copy()
    faces = diff_mesh.faces.copy()
    nV = len(vertices)
    nF = len(faces)
    
    # full_result = np.full((edges.shape[0]), -1)
    # full_result[edge_mask==1.] = np.arange(len(intersections))
    
    # intersection_indices = full_result[inverse_indices].reshape((3,-1)).T
    # intersection_indices = full_result.reshape((3,-1)).T
    # intersection_indices = ((np.cumsum(inter_mask)-1)).reshape((3,-1)).T
    # intersection_indices = ((np.cumsum(inter_mask)-1)*(1/inter_mask)).reshape((3,-1)).T
    # intersection_indices = np.repeat(np.array([np.nan, 0, 1]), nF).reshape((3,-1)).T *\
    intersection_indices = (np.full(edges.shape[0], -1) +\
        np.cumsum(edge_mask)*edge_mask).reshape((3, -1)).T
    intersection_indices_p = (np.full(edges.shape[0], -1) +\
        (np.arange(len(edge_mask))+1)*edge_mask).reshape((3, -1)).T

    new_vertices = np.concatenate((vertices, intersections))
    new_vertices_p = np.concatenate((vertices, intersections_p))
    data = []
    
    new_faces_p1 = []
    new_faces_p2 = []
    
    real_mask = np.full(4*nF, np.nan)
    # real_mask1 = []
    # real_mask2 = []
    
    II = intersection_indices[np.where(face_mask, True, False)].astype(int)
    
    slice_edges = []
    
    for i in range(len(II)):
        I = II[i,:]
        
        shift = np.argwhere(I==-1)[0,0]
        i1 = (1 + shift) % 3
        i2 = (2 + shift) % 3
        
        slice_edges.append([I[i2], I[i1]])
                           
    slice_edges = np.array(slice_edges)
    
    poly = DifferentiablePolygon(intersections, slice_edges)
    
    for i in range(nF):
        V = faces[i,:]
        I = intersection_indices[i,:]
        Ip = intersection_indices_p[i,:]
        
        # initial
        a = [V]
        
        # sliced
        # shift = np.argwhere(np.isinf(I)|np.isnan(I))[0,0]
        shift = np.argwhere(I==-1)[0,0]
        # shift = 0
        
        i0 = (0 + shift) % 3
        i1 = (1 + shift) % 3
        i2 = (2 + shift) % 3
        
        b = [
            np.array([V[i2], nV+I[i2], nV+I[i1]]),
            np.array([V[i1], nV+I[i1], nV+I[i2]]),
            np.array([V[i0], V[i1], nV+I[i2]])
        ]
        
        bp = [
            np.array([V[i2], nV+Ip[i2], nV+Ip[i1]]),
            np.array([V[i1], nV+Ip[i1], nV+Ip[i2]]), 
            np.array([V[i0], V[i1], nV+Ip[i2]])    
        ]
        
        new_faces_p1.append(a)
        new_faces_p2.append(bp)
        
        # real_mask1.append(np.array([1-face_mask[i]]))
        # real_mask2.append(np.array([face_mask[i]]*3))
        
        real_mask[i] = 1-face_mask[i]
        real_mask[nF+3*i:nF+3*i+3] = face_mask[i]
        
        data.append(
            a * (1-int(face_mask[i])) +\
            b * int(face_mask[i])
            )
    
    new_faces = np.concatenate(data)
    new_faces_p = np.concatenate(new_faces_p1+new_faces_p2)
    # real_mask = np.concatenate(real_mask1+real_mask2)
    
    new_mesh = DifferentiableMesh(vertices=new_vertices, faces=new_faces.astype(int))
    # new_mesh_p = DifferentiableMesh(vertices=new_vertices, faces=new_faces_p[real_mask==1].astype(int))
    new_mesh_p = DifferentiableMesh(vertices=new_vertices_p, faces=new_faces_p.astype(int))
    
    # center = np.mean(intersections, axis=0)
    center = point
    centers = np.repeat(wide(center), new_mesh.vertices.shape[0], axis=0)
    
    new_dist = np.concatenate((dist, np.zeros(len(intersections))))
    
    wet_weight = (1 - np.tanh(1e6*(new_dist-1e-4)))/2
    dry_weight = (1 + np.tanh(1e6*(new_dist+1e-4)))/2
        
    wet_vertices = centers * tall(1-wet_weight) + new_mesh.vertices * tall(wet_weight)
    dry_vertices = centers * tall(1-dry_weight) + new_mesh.vertices * tall(dry_weight)
    
    cross = dry_weight * wet_weight
    
    wp_vertices = centers * tall(1 - cross) + new_mesh.vertices * tall(cross)
    
    wet_mesh = DifferentiableMesh(wet_vertices, faces=new_faces.astype(int))
    dry_mesh = DifferentiableMesh(dry_vertices, faces=new_faces.astype(int))
    
    wp_mesh = DifferentiableMesh(wp_vertices, faces=new_faces.astype(int))
    
    Vaa = dry_mesh.volume
    Caa = dry_mesh.volume_centroid
    
    Vw = wet_mesh.volume
    cob = wet_mesh.volume_centroid
    
    Awp = wp_mesh.area / 2
    cof = wp_mesh.area_centroid

    
    rho = 1.025
    displacement = 10e3
    volume = displacement / rho
    
    opti.minimize((Vw - volume)**2)
    
    # sol = opti.solve()
    
        
    print()
    # print('Vaa', Vaa)
    print('Vw', Vw)
    print('cob', cob)
    # print('cof', cof)
    # print('Awp', Awp)
    
    # print((Vw+Vaa)/new_mesh.volume)
    
    # TODO: OPTIMIZE EXECUTION TIME
    
    fdist = new_mesh.faces_distances_to_plane(point, normal)
    
    zfac = ReLU(-fdist)# * (1-face_mask) #+ fac1 * face_mask
    
    # p = (rho * g * zfac)
    p = zfac
    # dS = tall(diff_mesh._data['triangle_areas']) * normals
    dS = new_mesh.normals
    
    pi = - tall(p) * dS
    
    volume = np.sum(pi[:,2], axis=0)
    
    # f = np.linalg.norm(pi, axis=1)
    # f = np.linalg.norm(pi[:,:2], axis=1)
    f = pi[:,2]
    
    lcb = np.sum(f * new_mesh._data['triangle_centers'][:,0], axis=0)/ volume
    tcb = np.sum(f * new_mesh._data['triangle_centers'][:,1], axis=0)/ volume
    vcb = np.sum(f * new_mesh._data['triangle_centers'][:,2], axis=0)/ volume
    
    result = diff_mesh.hydrostatics(point, normal)
    
    print()
    print(result)
    
    # # print(lcb, tcb, vcb)
    # v0, v1, v2 = diff_mesh._v0, diff_mesh._v1, diff_mesh._v2
    # cross_prod = np.cross(v2-v0, v1-v0)
    # vols = np.sum((v0-point) * cross_prod / 6.0, axis=1)
    
    # fdist = diff_mesh.faces_distances_to_plane(point, -normal)
    
    # print(np.sum(vols * np.fmax(np.fmin(fdist+0.5,1.),0.)))
    # print(np.sum(vols * (fdist>0)))


    #%% DRAWING
    
    import pyvista as pv
    
    # Create a PyVista plotter
    plotter = pv.Plotter()
    
    color = 'cyan'
    opacity = 0.5
    show_edges = True
    draw_plane = True
    # point = np.zeros(3)
    # normal = np.array([0., 0., 1.])
    mesh_color = 'grey'
    plane_color = 'orange'
    plane_opacity = 0.01
    backend: str = 'pyvista'
    show: bool = False
    # show: bool = True
    
    
    if show:
        # Create the mesh
        
        mesh = pv.PolyData(new_mesh.vertices, np.hstack([[3, *face] for face in new_mesh.faces]))
        plotter.add_mesh(mesh, 
                          # color=color, 
                          scalars=wet_weight,       # This lets PyVista color using a colormap
                          cmap='viridis',
                          opacity=opacity,
                          show_edges=show_edges
                          )
        
        # mesh = pv.PolyData(new_mesh_p.vertices, np.hstack([[3, *face] for face in new_mesh_p.faces]))
        # plotter.add_mesh(mesh, 
        #                   # color=color, 
        #                   scalars=real_mask,       # This lets PyVista color using a colormap
        #                  cmap='viridis',
        #                  opacity=real_mask,
        #                  show_edges=show_edges
        #                  )
        
        if draw_plane:
            # Add the plane
            plane = pv.Plane(center=point, direction=normal, i_size=200, j_size=200)
            plotter.add_mesh(plane, color=plane_color, opacity=plane_opacity)
            
        # if len(selected_edges) > 0:
        #     lines = np.hstack([[2, *edge] for edge in selected_edges]).astype(np.int32)
        #     # lines = intersections[slice_edges]
        #     line_mesh = pv.PolyData()
        #     line_mesh.points = diff_mesh.vertices
        #     line_mesh.lines = lines
        #     # line_mesh.lines = slice_edges
        #     plotter.add_mesh(line_mesh, color="red", line_width=3)
            
        for u, v in slice_edges:
            # Extract endpoints
            pts = intersections[[u, v]]
            
            # Each line cell format: [number_of_points, id0, id1]
            # For 2-point line it's always [2, 0, 1]
            line = pv.PolyData(pts, lines=np.array([2, 0, 1]))
            
            # Add to plotter
            plotter.add_mesh(line, color="red", line_width=3)
            
        # if len(intersections) > 0:
        #     # Add intersection points
        #     # intersection_cloud = pv.plot(intersections)
        #     plotter.add_points(intersections,
        #                            style='points_gaussian',
        #                             render_points_as_spheres=True,
        #                            # scalars=rgba,
        #                            # rgba=True,
        #                            color='red',
        #                            point_size=0.5,
        #                            )
        
        
        # Display the plot
        plotter.show()
        
        volume = np.sum(np.sum((diff_mesh._v0 - wide(point)) * diff_mesh._data["cross_product"] / 6.0, axis=1) * np.fmax(np.fmin(-(diff_mesh._data['triangle_centers'] - wide(point)) @ normal + 0.5, 1.0), 0.0))
