import tifffile
from skimage.measure import marching_cubes
from stl import mesh
import numpy as np


def get_mesh(volume, pixelDims=(2., 1.015625, 1.015625)):
    verts, faces, normals, values = marching_cubes(volume, spacing=pixelDims)

    # Create mesh
    model_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model_3d.vectors[i][j] = verts[f[j], :]

    return model_3d

