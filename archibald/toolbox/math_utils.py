# -*- coding: utf-8 -*-
"""
Created 2023-07-04

Useful math tools for Archibald

@author: Jules Richeux
@university: ENSA Nantes, FRANCE

"""


#%% DEPENDENCIES

import csv
import casadi as ca
import xarray as xr
import archibald2.numpy as np
import scipy.interpolate as itrp

#%% FUNCTIONS

def read_coefs(name, delim='\t', skipRows=0, columns=[]):
    TAB = list()
    
    with open(name, 'r') as file:
        reader = csv.reader(file, delimiter = delim)
        for row in reader:
            TAB.append(row)
    
    if columns:
        return np.array(TAB[skipRows:])[:,columns].astype('float32')
    return np.array(TAB[skipRows:])[:,:].astype('float32')


def tall(array):
    return np.reshape(array, (-1, 1))


def wide(array):
    return np.reshape(array, (1, -1))


def rotation_matrix(vector, angle):
    """
    Computes the rotation matrix for given axis and angle.

    Parameters
    ----------
    vector ((3,) array): rotation axis
    angle (float): rotation angle in deg

    Returns
    -------
    (3,3) array: corresponding rotation matrix

    """
    
    # Normalize the vector
    vector = vector / np.linalg.norm(vector)
    
    # Components of the vector
    x, y, z = vector
    
    # Compute the rotation matrix elements
    c = cosd(angle)
    s = sind(angle)
    t = 1 - c
    
    # Construct the rotation matrix
    rotation = np.array([[t * x**2 + c, t * x * y - s * z, t * x * z + s * y],
                         [t * x * y + s * z, t * y**2 + c, t * y * z - s * x],
                         [t * x * z - s * y, t * y * z + s * x, t * z**2 + c]])
    
    return rotation


def rotate_x(vec3d, angle):
    """
    Rotates a 3D vector around the x-axis.

    Parameters
    ----------
    vec3d ((3,) array): vector to rotate
    angle (float): rotation angle in deg

    Returns
    -------
    (3,) array: rotated vector

    """
    
    rot_matrix = np.array([[1, 0, 0],
                           [0, cosd(angle), -sind(angle)],
                           [0, sind(angle), cosd(angle)]])
    
    return np.dot(rot_matrix, vec3d)


def rotate(x, axis, angle, x0=None):
    """
    

    Parameters
    ----------
    x : ndarray
    x0 : ndarray
    axis : ((3,) array)
    angle : float

    Returns
    -------
    Rotation of x about the axis passing through the point x0

    """
    if x0 is None:
        x0 = np.zeros(x[0].shape)
    
    return np.dot(x - x0, rotation_matrix(axis, angle)) + x0


def set_normal(heel, trim):
    z0 = np.array([0,0,1])
    y0 = np.array([0,1,0])
    
    z1 = rotate_x(z0, heel)
    y1 = rotate_x(y0, heel)
    
    z2 = z1*cosd(trim) + np.cross(y1, z1)*sind(trim) + y1*np.dot(y1, z1)*(1 - cosd(trim))

    return -z2


def build_interpolation(coefs, method='cubic'):
    A = list()
    
    for a in range(1, coefs.shape[1]):
        A.append(itrp.interp1d(coefs[:,0], coefs[:,a], kind=method))
        
    return A

def build_2D_interpolator_from_csv(csvFile, delim=','):
    """
    
    Parameters
    ----------
    csvFile : str. Path to 2D gridded and ordered data.
    delim : str. CSV separator. The default is ','.

    Returns
    -------
    interpolator : interpax.Interpolator2D. R^2 to R function interpolating the given data.

    """
    data = np.genfromtxt(csvFile, delimiter=delim)
    
    # Assuming the first column represents x values, and the first row represents y values
    x = data[1:, 0]  # Exclude the first element which represents y values
    y = data[0, 1:]  # Exclude the first element which represents x values
    z = data[1:, 1:]  # Exclude the first row and column
    
    # Create 2D interpolator
    #interpolator = Interpolator2D(x, y, z, method='cubic')
    #interpolator = itrp.interp2d(x, y, z, kind='cubic')
    interpolator = itrp.interp2d(x, y, z)
    
    return interpolator


def compute_AW(tws, twa, V):
    """
    Computes apparent wind from true wind.

    Parameters
    ----------
    tws : float. True wind speed in m/s
    twa : float. True wind angle in deg
    V : float. Boat speed in m/s

    Returns
    -------
    Apparent wind speed in m/s
    Apparent wind angle in deg

    """
    
    # Convert inputs to numpy arrays for vectorized operations
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    V = np.asarray(V)
    
    # Calculate true wind components
    TW_x = tws * cosd(twa)
    TW_y = tws * sind(twa)
    
    # Boat speed components (assuming boat is moving along x-axis)
    SW_x = V
    SW_y = np.zeros_like(V)
    
    # Apparent wind components
    AW_x = TW_x + SW_x
    AW_y = TW_y + SW_y
    
    # Apparent wind speed and angle
    aws = np.sqrt(AW_x**2 + AW_y**2)
    awa = np.arctan2(AW_y, AW_x)
    
    return aws, awa * 180./np.pi
    # return aws, np.rad2deg(awa)


def compute_TW(aws, awa, V):
    """
    Computes true wind from apparent wind, vectorized.

    Parameters
    ----------
    aws : array-like
        Apparent wind speed [m/s]
    awa : array-like
        Apparent wind angle [deg]
    V : array-like
        Boat speed [m/s] (positive along +x, forward)

    Returns
    -------
    tws : ndarray
        True wind speed [m/s]
    twa : ndarray
        True wind angle [deg] (math convention, CCW from x-axis)
    """
    aws = np.asarray(aws)
    awa = np.asarray(awa)
    V   = np.asarray(V)

    # Apparent wind vector in boat frame (x = forward, y = starboard)
    AWx = aws * cosd(awa)
    AWy = aws * sind(awa)

    # Ship motion vector (boat speed forward, 0 sideways)
    SWx = V
    SWy = 0.0

    # True wind vector = apparent wind - ship velocity
    TWx = AWx - SWx
    TWy = AWy - SWy

    # Magnitude and angle
    tws = np.hypot(TWx, TWy)
    twa = np.degrees(np.arctan2(TWy, TWx))  # radians → deg

    return tws, twa


def kts2ms(Vkts):
    """
    Converts knots to m/s

    Parameters
    ----------
    Vkts : float. Speed in knots

    Returns
    -------
    float: speed in m/s

    """
    return Vkts * 1.852 / 3.6


def ms2kts(Vms):
    """
    Converts m/s to knots

    Parameters
    ----------
    Vkts : float. Speed in m/s

    Returns
    -------
    float: speed in knots

    """
    return Vms / 1.852 * 3.6


def build_casadi_interpolant_from_path_from_dataset(ds, column, method='linear'):
    """
    NB: dataset coordinates must be a grid.
    """
    x = [ds.coords[coord].values for coord in ds.coords]
    y = ds[column].to_numpy().ravel(order='F')

    return ca.interpolant('LUT',
                          method,
                          x,
                          y)


def build_casadi_interpolant_from_path(path, column, method='linear'):
    
    ds = xr.load_dataset(path)
    
    return build_casadi_interpolant_from_path_from_dataset(ds, column, method=method)


def ReLU(x):
    """
    Rectified Linear Unit function.

    """
    return np.fmax(x, 0.)


def GeLU(x):
    """
    Approximation of the Gaussian Error Linear Units function.

    """
    return 0.5*x*(1+np.tanh(np.sqrt(2/np.pi)*(x+0.044715*x**3)))


def smooth_ramp(x):
    """
    Tanh ramp.
    """
    return (np.tanh(2*x + 1.) + 1.) / 2.


def ramp(x, tol=1e-3):
    """
    Ramp function going from 0 at x=-tol to 1 at x=0

    """
    return np.fmin(1.,
                    np.fmax(0.,
                            (x+tol)/tol))


def rotate_points(points, matrix, center):
    """
    Rotates a set of 3D points around a given center using a rotation matrix.

    Parameters:
    - points: (n, 3) array-like, 3D points to rotate.
    - matrix: (3, 3) array-like, the rotation matrix.
    - center: (3,) array-like, the center point of rotation.

    Returns:
    - Rotated points in the same format as the input (NumPy or CasADi).

    See: https://numpy.org/doc/stable/reference/generated/numpy.dot.html
    """
    # if np.is_casadi_type(points) or np.is_casadi_type(matrix) or np.is_casadi_type(center):
    #     if len(points.shape) < 2:
    #         points = wide(points)
    
    arr = np.add(np.add(points, -wide(center)) @ matrix, wide(center))
    
    if not np.is_casadi_type(arr) and arr.shape[0] == 1:
        arr = arr.reshape(arr.shape[1])
        
    return arr


def rotate_single_vector(vector, matrix, center):
    return np.sum(np.add(np.add(wide(vector), -wide(center)) @ matrix, wide(center)), axis=0)


def sym_array(x, fac=1.):
    return np.hstack((x[::-1] * fac,
                      x[1:]))
