import tifffile
from skimage.measure import marching_cubes_lewiner
from skimage.morphology import binary_dilation
from stl import mesh
import numpy as np
import argparse


def get_mesh(volume, pixelDims=(2., 1.015625, 1.015625)):
    verts, faces, normals, values = marching_cubes_lewiner(volume, spacing=pixelDims)

    # Create mesh
    model_3d = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            model_3d.vectors[i][j] = verts[f[j], :]

    return model_3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a stl file from a 3D binary image in tif format')
    parser.add_argument('-i', type=str, help='input tif image file to process')
    parser.add_argument('-o', type=str, help='output file')
    parser.add_argument('-pixel_dimensions', type=str, default="2.0,1.015625,1.015625",
                        help='Pixel dimensions in [z, y, x]')
    parser.add_argument('-dilation', type=bool, default=False, help='set to True to do binary dilation')
    args = parser.parse_args()

    img = tifffile.imread(args.i)
    pixelDims = [float(item) for item in args.pixel_dimensions.split(',')]

    if args.dilation:
        img = binary_dilation(img)

    imgMesh = get_mesh(img, tuple(pixelDims))

    imgMesh.save(args.o)



