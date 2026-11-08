# -*- coding: utf-8 -*-
"""
Created 04/07/2023 Last update: 03/01/2024.

Useful geometrical tools for archibald

@author: Jules Richeux
@university: ENSA Nantes, FRANCE
@contributors: -

"""

#%% DEPENDENCIES

import numpy as np
import ezdxf
import math
import scipy.interpolate as itrp


#%% FUNCTIONS

_VALID_DXF_TYPES = [
    "POLYLINE",
    "SPLINE" ,
    "LINE",
    "LWPOLYLINE",
    "ARC",
]

def build_spline(control_points, degree, n=200):
    """Points of a B-spline through the given control points."""

    control_points = np.array(control_points)
    tck, u = itrp.splprep(control_points.T, k=degree)
    
    u = np.linspace(1, 0, n)
    points = itrp.splev(u, tck)
    
    return points


def build_bezier(ctrl, n=200):
    """Points of a Bézier curve with the given control points."""
    
    m = len(ctrl) - 1
    d = len(ctrl[0])
    
    t = np.linspace(0., 1., n)
    tt = np.vstack([t.T]*d).T
    
    points = np.zeros((n, d))
    
    for i in range(m+1):
        bernstein_i_m = math.comb(m, i) * tt**i * (1-tt)**(m-i)
        
        points += ctrl[i] * bernstein_i_m
        
    return np.transpose(points)


def evaluate_curve(points, z_coordinate):
    """Point of a curve closest to the given z coordinate."""
    
    closest_point_idx = np.argmin(np.abs(points[2] - z_coordinate))
    closest_point = (points[0][closest_point_idx], points[1][closest_point_idx], points[2][closest_point_idx])
    
    return closest_point


def spline_to_le_chords(leControl, leDeg, teControl, teDeg, nSections, direction='Z'):
    """Leading edges and chords of sections between two splines."""
    
    leCurve = build_spline(leControl, leDeg)
    teCurve = build_spline(teControl, teDeg)
    
    if direction == 'Z':
        axis = 2
    elif direction == 'Y':
        axis = 1
    elif direction == 'X':
        axis = 0
    else:
        raise('Invalid axis')
        return None, None
        
    minZ = np.max([np.min(leControl[:,2]), np.min(teControl[:,2])])
    maxZ = np.min([np.max(leControl[:,2]), np.max(teControl[:,2])])
    
    Z = np.linspace(minZ, maxZ, nSections)
    
    le = []
    chords = []
    
    for z in Z:
        lePt = evaluate_curve(leCurve, z)
        tePt = evaluate_curve(teCurve, z)
        
        le.append(lePt)
        chords.append(lePt[0]-tePt[0])
        
    le, chords = np.array(le), np.array(chords)
    
    sortedIdx = np.argsort(le[:, axis])
    
    return le[sortedIdx], chords[sortedIdx]


def bezier_to_le_chords(leControl, teControl, nSections):
    """Leading edges and chords of sections between two Bézier curves."""
    
    leCurve = build_bezier(leControl)
    teCurve = build_bezier(teControl)
        
    minZ = np.max([np.min(leControl[:,2]), np.min(teControl[:,2])])
    maxZ = np.min([np.max(leControl[:,2]), np.max(teControl[:,2])])
    
    Z = np.linspace(minZ, maxZ, nSections)
    
    le = []
    chords = []
    
    for z in Z:
        lePt = evaluate_curve(leCurve, z)
        tePt = evaluate_curve(teCurve, z)
        
        le.append(lePt)
        chords.append(lePt[0]-tePt[0])
        
    return np.array(le), np.array(chords)


def read_dxf(filename, nSections, method='bezier'):
    """Reads the leading and trailing edge curves from a DXF file."""
    
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    splines = []
    
    for entity in msp:
        if entity.dxftype() in _VALID_DXF_TYPES:
            splines.append(entity)
    
    if len(splines) == 2:
        
        def get_entity_data(entity):
            dtype = entity.dxftype()
            if dtype == 'SPLINE':
                control = np.array(entity.control_points[:])
                deg = entity.dxf.degree
            elif dtype == 'POLYLINE':
                control = np.array(list(entity.points()))
                deg = 1
            elif dtype == 'LWPOLYLINE':
                control = np.array(list(entity.vertices_in_wcs()))
                deg = 1
            elif dtype == 'ARC':
                # Convert the arc to a spline to preserve exact curvature data
                spline_ent = entity.to_spline(replace=False)
                control = np.array(spline_ent.control_points[:])
                deg = spline_ent.dxf.degree
            else:  # LINE or other entities with start/end attributes
                start = [entity.dxf.start.x, entity.dxf.start.y, entity.dxf.start.z]
                end = [entity.dxf.end.x, entity.dxf.end.y, entity.dxf.end.z]
                control = np.linspace(start, end, 3)
                deg = 2
            return control, deg

        leControl, leDeg = get_entity_data(splines[0])
        teControl, teDeg = get_entity_data(splines[1])
        
        if np.max(leControl[:, 0]) <= np.max(teControl[:, 0]): # if le is behind te
            # swap the edges
            tmpControl = leControl
            leControl = teControl
            teControl = tmpControl
            tmpDeg = leDeg
            leDeg = teDeg
            teDeg = tmpDeg
        
        return leControl, leDeg, teControl, teDeg
    
    else:
        return None, None, None, None
            

def dxf_to_le_chords(filename, nSections, method='bezier', direction='Z'):
    """Leading edges and chords of sections from the curves of a DXF file."""
    
    leControl, leDeg, teControl, teDeg = read_dxf(filename, nSections, method)
    
    if leControl is None:
        print('Invalid DXF file')
        return None, None

    else:
        if method == 'spline':
            return spline_to_le_chords(leControl, leDeg, teControl, teDeg, nSections, direction)
        elif method == 'bezier':
            return bezier_to_le_chords(leControl, teControl, nSections)
        else:
            print('Invalid method')


def outline_to_le_chords(filename, nSections, margin=1e-3, flattening=1e-3):
    """
    Leading edges and chords of a wing drawn as a planform outline in a DXF.

    Unlike ``dxf_to_le_chords``, which needs exactly two curves (leading edge
    first, then trailing edge), the outline can be made of any number of
    entities (lines, splines, arcs, polylines), in any order and direction, as
    long as they lie in a plane containing the span along z. At each station
    the outline is cut by the plane z = const: the most forward crossing is the
    leading edge, the most aft one the trailing edge (x forward).

    Args:
        filename: str, DXF file.
        nSections: int, number of sections, evenly spaced in z.
        margin: float, distance [m] kept from the lowest and highest points of the
            outline, where the chord would vanish.
        flattening: float, maximum distance [m] between a curve and its polyline.

    Returns:
        le: array (nSections, 3), leading edge points, from bottom to top [m].
        chords: array (nSections,), chords [m].
    """
    msp = ezdxf.readfile(filename).modelspace()
    polylines = []
    for entity in msp:
        if entity.dxftype() not in _VALID_DXF_TYPES:
            continue
        if entity.dxftype() == 'LINE':
            points = [entity.dxf.start, entity.dxf.end]
        else:
            points = list(ezdxf.path.make_path(entity).flattening(flattening))
        polylines.append(np.array([[p[0], p[1], p[2]] for p in points], dtype=float))

    if not polylines:
        raise ValueError(f"No curve found in {filename}")

    def cut(z):
        crossings = []
        for p in polylines:
            z0, z1 = p[:-1, 2], p[1:, 2]
            for k in np.nonzero((np.minimum(z0, z1) <= z) & (np.maximum(z0, z1) >= z) & (z0 != z1))[0]:
                t = (z - z0[k]) / (z1[k] - z0[k])
                crossings.append(p[k] + t * (p[k + 1] - p[k]))
        return np.array(crossings)

    # Span where the outline has both edges, i.e. is cut at least twice
    # (an open outline, such as two lines of different lengths, is shorter)
    allZ = np.concatenate([p[:, 2] for p in polylines])
    fine = np.linspace(allZ.min(), allZ.max(), 2001)
    cut_twice = fine[[len(cut(z)) >= 2 for z in fine]]
    if len(cut_twice) == 0:
        raise ValueError(f"The outline of {filename} has no span with both edges")
    Z = np.linspace(cut_twice.min() + margin, cut_twice.max() - margin, nSections)

    le = []
    chords = []
    for z in Z:
        crossings = cut(z)
        if len(crossings) < 2:
            raise ValueError(f"The outline of {filename} is crossed less than twice at z = {z:.4f}")
        front = crossings[np.argmax(crossings[:, 0])]
        back = crossings[np.argmin(crossings[:, 0])]
        le.append(front)
        chords.append(front[0] - back[0])

    return np.array(le), np.array(chords)


# DEPRECATED
def world_to_vlm(coords3D):
    """Converts coordinates from the world frame to the VLM frame."""
    return np.transpose(np.vstack([-coords3D[:,0], coords3D[:,2], coords3D[:,1]]))