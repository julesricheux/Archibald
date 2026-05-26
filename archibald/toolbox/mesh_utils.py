import os
import archibald.numpy as np
from typing import Tuple
"""
Documentation of (points, faces) standard format, which is an unstructured mesh format:

Meshes are given here in the common (points, faces) format. In this format, `points` is a Nx3 array, where each row 
gives the 3D coordinates of a vertex in the mesh. Entries into this array are floating-point, generally speaking.

`faces` is a Mx3 array in the case of a triangular mesh, or a Mx4 array in the case of a quadrilateral mesh. Each row 
in this array represents a face. The entries in each row are integers that correspond to the index of `points` where 
the vertex locations of that face are found. 

"""


def stack_meshes(
        *meshes: Tuple[Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes in a series of tuples (points, faces) and merges them into a single tuple (points, faces). All (points,
    faces) tuples are meshes given in standard format.

    Args:
        *meshes: Any number of mesh tuples in standard (points, faces) format.

    Returns: Points and faces of the combined mesh. Standard unstructured mesh format: A tuple of `points` and
    `faces`, where:

        * `points` is a `n x 3` array of points, where `n` is the number of points in the mesh.

        * `faces` is a `m x 3` array of faces if `method` is "tri", or a `m x 4` array of faces if `method` is "quad".

            * Each row of `faces` is a list of indices into `points`, which specifies a face.

    """
    if len(meshes) == 1:
        return meshes[0]
    elif len(meshes) == 2:
        points1, faces1 = meshes[0]
        points2, faces2 = meshes[1]

        faces2 = faces2 + len(points1)

        points = np.concatenate((points1, points2))
        faces = np.concatenate((faces1, faces2))

        return points, faces
    else:
        points, faces = stack_meshes(
            meshes[0],
            meshes[1]
        )
        return stack_meshes(
            (points, faces),
            *meshes[2:]
        )


def convert_mesh_to_polydata_format(
        points: np.ndarray,
        faces: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    PyVista uses a slightly different convention for the standard (points, faces) format as described above. They
    give `faces` as a single 1D vector of roughly length (M*3), or (M*4) in the case of quadrilateral meshing.
    Basically, the mesh displayer goes down the `faces` array, and when it sees a number N, it interprets that as the
    number of vertices in the following face. Then, the next N entries are interpreted as integer references to the
    vertices of the face.

    This has the benefit of allowing for mixed tri/quad meshes.

    Args:
        points: `points` array of the original standard-format mesh
        faces: `faces` array of the original standard-format mesh

    Returns:

        (points, faces), except that `faces` is now in a pyvista.PolyData compatible format.

    """
    faces = [
        [len(face), *face]
        for face in faces
    ]
    faces = np.reshape(np.array(faces), -1)
    return points, faces

def read_binary_stl(file_path):
    """Reads a binary STL using NumPy's highly optimized fromfile method."""
    # Define the exact byte-structure of a binary STL triangle
    # 3 floats for normal, 9 floats for vertices, 1 uint16 for attribute count
    stl_dtype = np.dtype([
        ('normals', np.float32, (3,)),
        ('v0', np.float32, (3,)),
        ('v1', np.float32, (3,)),
        ('v2', np.float32, (3,)),
        ('attr', np.uint16)
    ])
    
    with open(file_path, 'rb') as f:
        # Skip the 80-byte header
        f.read(80)
        # Read the number of triangles
        num_triangles = np.fromfile(f, dtype=np.uint32, count=1)[0]
        # Read the rest of the file directly into a structured NumPy array
        mesh_data = np.fromfile(f, dtype=stl_dtype, count=num_triangles)

    # Pre-allocate an array for all vertices (3 vertices per triangle)
    all_vertices = np.empty((num_triangles * 3, 3), dtype=np.float32)
    
    # Interleave the vertices so v0, v1, v2 stay sequential for each triangle
    all_vertices[0::3] = mesh_data['v0']
    all_vertices[1::3] = mesh_data['v1']
    all_vertices[2::3] = mesh_data['v2']

    # Deduplicate vertices and let NumPy generate the face indices automatically
    vertices, indices = np.unique(all_vertices, axis=0, return_inverse=True)
    
    # Reshape the flat indices array back into triplets (faces)
    faces = indices.reshape(-1, 3)

    return vertices, faces

def read_ascii_stl(file_path):
    """Reads an ASCII STL and uses NumPy for deduplication."""
    vertices_list = []
    
    # Read text and extract just the vertex lines
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == 'vertex':
                vertices_list.append([float(parts[1]), float(parts[2]), float(parts[3])])
                
    all_vertices = np.array(vertices_list, dtype=np.float32)

    # ASCII files also output vertices in groups of 3 sequentially
    vertices, indices = np.unique(all_vertices, axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    
    return vertices, faces

def load_stl(file_path):
    """
    Detects whether an STL is ASCII or Binary,
    parses it using NumPy, and returns (vertices, faces).
    """
    file_size = os.path.getsize(file_path)
    
    with open(file_path, 'rb') as f:
        header = f.read(80)
        if len(header) < 80:
            return read_ascii_stl(file_path)

        num_triangles_bytes = f.read(4)
        if len(num_triangles_bytes) < 4:
            return read_ascii_stl(file_path)

        # Unpack the number of triangles (returns a np.uint32)
        num_triangles = np.frombuffer(num_triangles_bytes, dtype=np.uint32)[0]
        
        # FIX: Cast to standard Python int to prevent uint32 overflow on ASCII files
        expected_binary_size = 84 + (int(num_triangles) * 50)

        # If the file size matches exactly, it's binary. Otherwise, it's ASCII.
        if file_size == expected_binary_size:
            return read_binary_stl(file_path)
        else:
            return read_ascii_stl(file_path)
