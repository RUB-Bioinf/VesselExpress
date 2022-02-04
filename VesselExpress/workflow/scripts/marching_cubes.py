import functools
import argparse
import os
import time
from pathlib import Path

from PIL import Image, ImageSequence
import numpy as np
from stl import mesh
from tqdm import tqdm
from scipy import ndimage
from multiprocessing import Pool

import sys
import os

# import modules
package = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'dependencies/'))
sys.path.append(package)

import triangulation_table as tri_table


def apply_factor(coordinates: object, factor_x: float = 1.0, factor_y: float = 1.0, factor_z: float = 1.0) -> list:
    """
    Function to add independent factors to the x, y and z coordinates.

    :param coordinates: Coordinates as a list
    :param factor_x: x-coordinate
    :param factor_y: y-coordinate
    :param factor_z: z-coordinate
    :return: list of coordinates
    """
    return [coordinates[0] * factor_x, coordinates[1] * factor_y, coordinates[2] * factor_z]


def get_triangle_stl_data(edge_indices, cube_position, cube_size, factor_x: float = 1.0,
                          factor_y: float = 1.0, factor_z: float = 1.0):
    """
    Function to create triangle-vertices and the face (connection of vertices) from
    three edge_indices (from triangulation-table), the root position of the cube,
    the size of the cube and the list of all triangle_vertices before.

    :param factor_z: Scaling factor for z-coordinate
    :param factor_y: Scaling factor for y-coordinate
    :param factor_x: Scaling factor for x-coordinate
    :param edge_indices: List of three edge indices
    :param cube_position: Tuple in form of (x,y,z)
    :param cube_size: Int value of the cubes size
    :return: list of all coordinates of a triangle and the definition of the triangle
    """
    f_x, f_y, f_z = factor_x, factor_y, factor_z

    triangle_coordinates \
        = [apply_factor(get_triangle_coordinate(edge_indices[0], cube_position, cube_size), f_x, f_y, f_z),
           apply_factor(get_triangle_coordinate(edge_indices[1], cube_position, cube_size), f_x, f_y, f_z),
           apply_factor(get_triangle_coordinate(edge_indices[2], cube_position, cube_size), f_x, f_y, f_z)]
    return triangle_coordinates


def save_stl(stl_vertices, stl_faces, path, debug=False):
    """
    Function to save a .stl file to given path using a list of faces and vertices.

    :param stl_vertices: list of vertices
    :param stl_faces: list of faces in connection with stl_vertices
    :param path: path to save the file to
    """
    if debug: print('Saving...')
    stl_faces = stl_faces.astype(int)
    cube = mesh.Mesh(np.zeros(stl_faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(stl_faces):
        for j in range(3):
            cube.vectors[i][j] = stl_vertices[f[j], :]
    cube.save(path)
    if debug: print('Finished!')


def get_triangle_coordinate(edge_index, cube_position, cube_size):
    """
    Calculates x,y,z of a triangle vertex from the root position of the cube, the size of
    the marching cube and the edge index from the triangulation-table.

    :param edge_index: index for edge from triangulation table, int value of 0-11
    :param cube_position: tuple of cube position in form of (x,y,z)
    :param cube_size: size of cube, int value
    :return: coordinate of triangle vertex for given edge index
    """
    x, y, z = cube_position[0], cube_position[1], cube_position[2]
    hc = cube_size / 2

    if edge_index == 0:
        return x, y + hc, z
    elif edge_index == 1:
        return x + hc, y + cube_size, z
    elif edge_index == 2:
        return x + cube_size, y + hc, z
    elif edge_index == 3:
        return x + hc, y, z

    elif edge_index == 4:
        return x, y + hc, z + cube_size
    elif edge_index == 5:
        return x + hc, y + cube_size, z + cube_size
    elif edge_index == 6:
        return x + cube_size, y + hc, z + cube_size
    elif edge_index == 7:
        return x + hc, y, z + cube_size

    elif edge_index == 8:
        return x, y, z + hc
    elif edge_index == 9:
        return x, y + cube_size, z + hc
    elif edge_index == 10:
        return x + cube_size, y + cube_size, z + hc
    elif edge_index == 11:
        return x + cube_size, y, z + hc

    else:
        raise ValueError("Edge index must be a value between 0 and 11")


def create_image_array(path: str):
    """
    Function to create a 3D-numpy array from a multilayer tif-file.

    :param path: Path to multilayer tif-file
    :return: multilevel tif-file as stacked numpy array
    """
    im = Image.open(path)
    array = []
    for i, img in enumerate(ImageSequence.Iterator(im)):
        if i == 0:
            array = np.asarray(img)
        else:
            array = np.dstack((array, np.asarray(img)))
    return array


def get_triangles_definition(index):
    """
    Function to create list of triangle (indices) from triangulation-table index.

    :param index: Triangulation-Table index
    :return: numpy-array of individual triangles
    """
    tri_vert = tri_table.triangulation_table[index]
    i = 0
    n = tri_vert[0]
    triangles = np.empty((0, 3))
    while i < 13 and n >= 0:
        triangles = np.append(triangles, [tri_vert[i:i + 3]], axis=0)
        i += 3
        n = tri_vert[i]
    return triangles


def tri_index(cube_vertices, isolevel=1):
    """
    Calculates triangulation-table index from cubes vertices.

    :param isolevel: Minimum value for a pixel to be interpreted as solid
    :param cube_vertices: list of 8 cube vertices
    :param threshold: value for a vertex to be interpreted as inside of model
    :return: triangulation-table index as int value
    """

    cubeindex = 0
    if cube_vertices[0] >= isolevel: cubeindex += 1
    if cube_vertices[1] >= isolevel: cubeindex += 2
    if cube_vertices[2] >= isolevel: cubeindex += 4
    if cube_vertices[3] >= isolevel: cubeindex += 8
    if cube_vertices[4] >= isolevel: cubeindex += 16
    if cube_vertices[5] >= isolevel: cubeindex += 32
    if cube_vertices[6] >= isolevel: cubeindex += 64
    if cube_vertices[7] >= isolevel: cubeindex += 128

    return cubeindex


def march_cube(cube_size, array, stl_path, isolevel: float = 1.0, factor_x: float = 1.0, factor_y: float = 1.0,
               factor_z: float = 1.0, debug=False):
    """
    Function to apply marching cubes to a 3d numpy array.

    :param isolevel: Minimum value for a pixel to be interpreted as solid
    :param factor_z: Scaling factor for z-coordinate
    :param factor_y: Scaling factor for y-coordinate
    :param factor_x: Scaling factor for x-coordinate
    :param cube_size: Int value of the size of the cube
    :param array: 3D-numpy array, the array to apply marching cubes
    :param stl_path: the path for the stl-file to safe
    :param debug: Boolean value to enable or disable debuging and progress information
    :return: stl_vertices and stl_faces
    """
    shape = array.shape
    cubes = np.array(np.meshgrid(np.arange(start=0, stop=shape[2] - cube_size - 1, step=cube_size),
                                 np.arange(start=0, stop=shape[1] - cube_size - 1, step=cube_size),
                                 np.arange(start=0, stop=shape[0] - cube_size - 1, step=cube_size))).T.reshape(-1,
                                                                                                               3)

    if debug:
        print('Marching Cubes progress:')
        bar_size = len(cubes)
        pbar = tqdm(total=bar_size)

    stl_vertices = []

    for i in range(len(cubes)):
        x = cubes[i][0]
        y = cubes[i][1]
        z = cubes[i][2]

        vert_values = [array[z, y, x],
                       array[z, y + cube_size, x],
                       array[z, y + cube_size, x + cube_size],
                       array[z, y, x + cube_size],
                       array[z + cube_size, y, x],
                       array[z + cube_size, y + cube_size, x],
                       array[z + cube_size, y + cube_size, x + cube_size],
                       array[z + cube_size, y, x + cube_size]]

        tri_table_index = tri_index(vert_values, isolevel)
        tri_vertices = get_triangles_definition(tri_table_index)

        for tri_vertex in tri_vertices:
            vertices = get_triangle_stl_data(tri_vertex, [x, y, z], cube_size, factor_x, factor_y,
                                             factor_z)
            stl_vertices.extend(vertices)
        if debug: pbar.update(1)

    if debug: pbar.close()

    stl_vertices = np.array(stl_vertices)
    stl_faces = np.arange(len(stl_vertices)).reshape(-1, 3)

    save_stl(stl_vertices, stl_faces, stl_path, debug)
    return stl_vertices, stl_faces


def march_cube_multi(cube_size, array, stl_path, isolevel: float = 1.0, factor_x: float = 1.0, factor_y: float = 1.0,
                     factor_z: float = 1.0, debug=False, workers: int = 2):
    """
    Function to apply marching cubes to a 3d numpy array.

    :param workers: Number of workers
    :param isolevel: Minimum value for a pixel to be interpreted as solid
    :param factor_z: Scaling factor for z-coordinate
    :param factor_y: Scaling factor for y-coordinate
    :param factor_x: Scaling factor for x-coordinate
    :param cube_size: Int value of the size of the cube
    :param array: 3D-numpy array, the array to apply marching cubes
    :param stl_path: the path for the stl-file to safe
    :param debug: Boolean value to enable or disable debuging and progress information
    :return: stl_vertices and stl_faces
    """
    shape = array.shape
    cubes = np.array(np.meshgrid(np.arange(start=0, stop=shape[2] - cube_size - 1, step=cube_size),
                                 np.arange(start=0, stop=shape[1] - cube_size - 1, step=cube_size),
                                 np.arange(start=0, stop=shape[0] - cube_size - 1, step=cube_size))).T.reshape(-1, 3)
    if debug: print('Marching Cubes progress:')

    with Pool(workers) as p:
        stl_vertices = p.map(
            functools.partial(get_vertices, data=[array, cube_size, factor_x, factor_y, factor_z, isolevel]), cubes)
    stl_vertices = np.array([val for sublist in stl_vertices for val in sublist])
    stl_faces = np.arange(len(stl_vertices)).reshape(-1, 3)

    save_stl(stl_vertices, stl_faces, stl_path, debug)
    return stl_vertices, stl_faces


def get_vertices(coordinate, data):
    """
    Function to calculate the stl vertices for a given coordinate.

    :param coordinate: list of coordinates
    :param data: list with the array, cube_size, factor_x, factor_y, factor_z and isolevel
    :return: stl vertices
    """
    x = coordinate[0]
    y = coordinate[1]
    z = coordinate[2]
    array, cube_size, factor_x, factor_y, factor_z, isolevel = data

    stl_vertices = []
    vert_values = [array[z, y, x],
                   array[z, y + cube_size, x],
                   array[z, y + cube_size, x + cube_size],
                   array[z, y, x + cube_size],
                   array[z + cube_size, y, x],
                   array[z + cube_size, y + cube_size, x],
                   array[z + cube_size, y + cube_size, x + cube_size],
                   array[z + cube_size, y, x + cube_size]]
    tri_table_index = tri_index(vert_values, isolevel)
    tri_vertices = get_triangles_definition(tri_table_index)
    for tri_vertex in tri_vertices:
        vertices = get_triangle_stl_data(tri_vertex, [x, y, z], cube_size, factor_x, factor_y,
                                         factor_z)
        stl_vertices.extend(vertices)
    return stl_vertices


def array_dilation(array, debug, dilation_iterations):
    """
    Dilates an 3D-array to make it feasible for Marching Cubes.

    :param array: The 3D-array to dilate
    :param debug: Boolean value to enable or disable debugging
    :param dilation_iterations: number of dilatation to apply to the array (for skeletonized data
    this should be equal to the cube size)
    :return: the dilated array
    """
    if debug: print('Dilation of array...')
    array = ndimage.binary_dilation(array, iterations=dilation_iterations)
    if debug: print('Array dilated!')
    return array


def tif_to_stl(image_path: str, cube_size: int, output_path: str, scaling_x: int, scaling_y: int, scaling_z: int, isolevel: float = 1.0, dilation: bool = False,
               dilation_iterations: int = None,
               debug=False, factor_x: float = 1, factor_y: float = 1, factor_z: float = 1, parallel: bool = False,
               workers: int = 2):
    """
    Function to generate an STL-File from an TIF-Image using an implementation of the marching cubes algorithm.

    :param scaling_x: Parameter to scale x-axis of the array
    :param scaling_y: Parameter to scale y-axis of the array
    :param scaling_z: Parameter to scale z-axis of the array
    :param parallel: Boolean value to control if execution is parallel
    :param workers: Number of workers in parallel execution
    :param image_path: Path to the input TIF
    :param dilation_iterations: Number of Iterations for the dilation if dilation is set to True, defaults to cube_size
    :param output_path: Path to save the STL-File to
    :param dilation: If True dilation of the input TIF
    :param isolevel: Minimum value for a pixel to be interpreted as solid
    :param factor_z: Scaling factor for z-coordinate
    :param factor_y: Scaling factor for y-coordinate
    :param factor_x: Scaling factor for x-coordinate
    :param cube_size: Int value of the size of the cube
    :param array: 3D-numpy array, the array to apply marching cubes
    :param stl_path: the path for the stl-file to safe
    :param debug: Boolean value to enable or disable debugging and progress information
    """
    if debug: print('Creating array from TIF-Image...')
    array = create_image_array(image_path)
    if debug: print('Array created!')
    if dilation:
        if dilation_iterations is None: dilation_iterations = cube_size
        array = array_dilation(array, debug, dilation_iterations)

    if debug: print('Scaling array...')
    array = array.repeat(scaling_x, axis=0).repeat(scaling_y, axis=1).repeat(scaling_z, axis=2)
    if debug: print('Scaling complete!')
    if parallel:
        march_cube_multi(cube_size, array, output_path, isolevel, factor_x, factor_y, factor_z, debug, workers)
    else:
        march_cube(cube_size, array, output_path, isolevel, factor_x, factor_y, factor_z, debug)


def parse_arguments():
    """
    Function to parse the arguments.

    :return: dictionary of argument values
    """
    my_parser = argparse.ArgumentParser(description='Applies the Marching Cubes algorithm to a series of layered '
                                                    'images.')

    my_parser.add_argument('-i',
                           '--input',
                           action='store',
                           type=str,
                           required=True,
                           metavar='input_path')
    my_parser.add_argument('-o',
                           '--output',
                           action='store',
                           type=str,
                           required=True,
                           metavar='output_path')
    my_parser.add_argument('-c',
                           '--cube_size',
                           action='store',
                           type=int,
                           required=True,
                           metavar='size')
    my_parser.add_argument('-l',
                           '--isolevel',
                           action='store',
                           type=float,
                           required=False,
                           default=1.0,
                           metavar='iso_level')
    my_parser.add_argument('-d',
                           '--dilation',
                           action='store_true')
    my_parser.add_argument('-j',
                           '--dilation_iteration',
                           action='store',
                           type=int,
                           required=False,
                           default=None,
                           metavar='iterations')
    my_parser.add_argument('-X',
                           '--scaling_x',
                           action='store',
                           type=float,
                           required=False,
                           default=1.0,
                           metavar='workers')
    my_parser.add_argument('-Y',
                           '--scaling_y',
                           action='store',
                           type=float,
                           required=False,
                           default=1.0,
                           metavar='workers')
    my_parser.add_argument('-Z',
                           '--scaling_z',
                           action='store',
                           type=float,
                           required=False,
                           default=1.0,
                           metavar='workers')
    my_parser.add_argument('-D',
                           '--debug',
                           action='store_true')
    my_parser.add_argument('-x',
                           '--factor_x',
                           action='store',
                           type=int,
                           required=False,
                           default=1,
                           metavar='factor')
    my_parser.add_argument('-y',
                           '--factor_y',
                           action='store',
                           type=int,
                           required=False,
                           default=1,
                           metavar='factor')
    my_parser.add_argument('-z',
                           '--factor_z',
                           action='store',
                           type=int,
                           required=False,
                           default=1,
                           metavar='factor')
    my_parser.add_argument('-P',
                           '--parallel',
                           action='store_true')
    my_parser.add_argument('-w',
                           '--workers',
                           action='store',
                           type=int,
                           required=False,
                           default=2,
                           metavar='workers')
    my_parser.add_argument('-p',
                           '--pixel_dimensions',
                           action='store',
                           type=str,
                           required=False,
                           default="1.0,1.0,1.0",
                           metavar='pixel dimensions')

    args = my_parser.parse_args()
    filepath = os.path.dirname(vars(args)['input'])
    if not Path(vars(args)['input']).is_file() or not Path(filepath).is_dir():
        raise IOError('Inputfile or Output-Path does not exist!')

    return vars(args)


def main():
    """
    Method to apply the Marching Cubes algorithm.
    """
    arguments = parse_arguments()
    if arguments['debug']: start = time.time()

    pixel_dims = [float(item) for item in arguments['pixel_dimensions'].split(',')]
    tif_to_stl(
        image_path=arguments['input'],
        cube_size=arguments['cube_size'],
        output_path=arguments['output'],
        isolevel=arguments['isolevel'],
        dilation=arguments['dilation'],
        dilation_iterations=arguments['dilation_iteration'],
        debug=arguments['debug'],
        factor_x=arguments['factor_x'],
        factor_y=arguments['factor_y'],
        factor_z=arguments['factor_z'],
        parallel=arguments['parallel'],
        workers=arguments['workers'],
        scaling_x=pixel_dims[2],
        scaling_y=pixel_dims[1],
        scaling_z=pixel_dims[0]

    )

    if arguments['debug']:
        end = time.time()
        print("The run took {:.4f} Seconds to complete!".format((end - start)))


if __name__ == "__main__":
    main()
