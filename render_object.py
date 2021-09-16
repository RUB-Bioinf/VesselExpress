import bpy
import os
from math import radians
import argparse
import sys


# Expected image_color_mode: BW, RGB, RGBA
# Expected render_engine: CYCLES, BLENDER_EEVEE, BLENDER_WORKBENCH
# Expected render_device: CPU, GPU
# Expected file_format: JPEG, PNG, TIFF, IRIS, JPEG2000, BMP, TARGA, TARGA_RAW

# Expected RGB Values: Float between 0.0 and 1.0

expected_image_color_mode = ['BW', 'RGB', 'RGBA']
expected_render_engine = ['CYCLES', 'BLENDER_EEVEE', 'BLENDER_WORKBENCH']
expected_render_device = ['CPU', 'GPU']
expected_image_bit_depth = [8, 16]
expected_file_format = ['JPEG', 'PNG', 'TIFF', 'IRIS', 'JPEG2000', 'BMP', 'TARGA', 'TARGA_RAW']


def render_object(model_file_path: str, out_dir: str,
                  save_raw: int = 1,
                  # Camera Params
                  camera_pos_x: float = -2.0, camera_pos_y: float = 3.0, camera_pos_z: float = 3.0,
                  camera_euler_angle_x: float = 422.0, camera_euler_angle_y: float = 0.0,
                  camera_euler_angle_z: float = 149.0,
                  # Background Color Params
                  background_r: float = 1.0, background_g: float = 1.0, background_b: float = 1.0,
                  background_a: float = 1.0, background_intensity: float = 1.2,
                  # Mesh Color Params
                  mesh_r: float = 0.425, mesh_g: float = 0.002, mesh_b: float = 0.0013,
                  mesh_a: float = 1.0, mesh_roughness: float = 0.65,
                  # Render & Hardware Params
                  render_engine: str = 'CYCLES', render_device='GPU', render_distance: int = 500000,
                  # Rendered Image Params
                  image_resolution_x: int = 1920, image_resolution_y: int = 1080, image_resolution_scale: float = 30.0,
                  image_color_mode: str = 'RGBA', image_bit_depth: int = 8, file_format: str = 'PNG',
                  image_compression: float = 0.0
                  ) -> (str, str):
    ########################################
    #### Setup #############################
    ########################################

    # Finding the model file on the file system
    if not os.path.exists(model_file_path):
        raise Exception('The requested model file does not exist: ' + model_file_path)
    file_name = os.path.basename(model_file_path)
    object_name = os.path.basename(os.path.splitext(model_file_path)[0])

    # Setting up the file system for the output
    os.makedirs(out_dir, exist_ok=True)

    # Sanitising string parameters
    if render_engine not in expected_render_engine:
        raise Exception('Invalid parameter for "render_engine". Got: ' + render_engine + '. Expected: ' + str(
            expected_render_engine))
    if render_device not in expected_render_device:
        raise Exception('Invalid parameter for "render_device". Got: ' + render_device + '. Expected: ' + str(
            expected_render_device))
    if image_color_mode not in expected_image_color_mode:
        raise Exception('Invalid parameter for "image_color_mode". Got: ' + image_color_mode + '. Expected: ' + str(
            expected_image_color_mode))
    if file_format not in expected_file_format:
        raise Exception(
            'Invalid parameter for "file_format". Got: ' + file_format + '. Expected: ' + str(expected_file_format))
    if image_bit_depth not in expected_image_bit_depth:
        raise Exception('Invalid parameter for "image_bit_depth". Got: ' + str(image_bit_depth) + '. Expected: ' + str(
            expected_image_bit_depth))

    ########################################
    #### Running Blender Pipeline ##########
    ########################################

    # Setting up Blender context
    print('Setting up Blender context for: ' + file_name)
    context = bpy.context
    bpy.context.scene.render.engine = render_engine
    bpy.context.scene.cycles.device = render_device

    # create a new scene
    print('Creating a new empty scene.')
    scene = bpy.data.scenes.new("Scene")

    # Creating a new camera for the final render
    camera_data = bpy.data.cameras.new("Camera")

    # Select objects by type "Mesh"
    for o in bpy.context.scene.objects:
        if o.type == 'MESH' or o.type == 'CAMERA' or o.type == 'LAMP':
            o.select_set(True)
        else:
            o.select_set(False)

    # Delete all selected objects (e.g. the default cube, light and camera)
    # (so they are not in the way later)
    bpy.ops.object.delete()

    # Setting up a new camera for later
    print('Setting up camera for render.')
    camera = bpy.data.objects.new("Camera", camera_data)
    camera.location = (camera_pos_x, camera_pos_y, camera_pos_z)
    camera.rotation_euler = ([radians(a) for a in (camera_euler_angle_x, camera_euler_angle_y, camera_euler_angle_z)])
    bpy.context.collection.objects.link(camera)

    # setting up scene lighting
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (
        background_r, background_g, background_b, background_a)
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = background_intensity

    # setting the scene
    print('Attaching camera to the scene.')
    scene.camera = camera
    bpy.ops.scene.new(type='LINK_COPY')
    context.scene.name = model_file_path

    print('Loading the model file.')
    bpy.ops.import_mesh.stl(filepath=model_file_path, axis_forward='-Z', axis_up='Y', filter_glob="*.obj;*.mtl")
    print('Finished loading.')

    # Deselecting every object in the scene
    bpy.ops.object.select_all(action='DESELECT')

    # Select objects by type "Mesh"
    active_object = None
    for o in bpy.context.scene.objects:
        if o.type == 'MESH':
            o.select_set(True)
            active_object = o
        if o.type == 'Light' or o.type == 'LIGHT':
            # o.select_set(True)
            pass
        else:
            o.select_set(False)

    if active_object is None:
        raise Exception('Failed to detect the model in the scene.')

    # Creating the material for the rendered object
    materials = bpy.data.materials
    mat_name = 'skeleton'
    material = materials.get(mat_name)
    if not material:
        material = materials.new(mat_name)
    clear_material(material)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    output = nodes.new(type='ShaderNodeOutputMaterial')
    diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
    diffuse.inputs[0].default_value = ([mesh_r, mesh_g, mesh_b, mesh_a])
    diffuse.inputs[1].default_value = mesh_roughness

    # Applying the material to the imported object
    link = links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])
    active_object.active_material = bpy.data.materials[mat_name]

    # Setting the render clipping distance for all cameras
    # Anything further away from this distance, will not be drawn
    for c in bpy.data.cameras:
        c.clip_end = render_distance

    # Setting the active render camera
    context.scene.camera = camera
    # hashing the scene parameters
    h = hash(object_name)
    h = abs(h)
    print('Parameter hash: ' + str(h))

    render_out_filename = out_dir + os.sep + object_name + '-render.' + file_format
    print('Rendering: ' + render_out_filename)
    # Aligning camera with object
    active_object.select_set(True)
    bpy.ops.view3d.camera_to_view_selected()
    # Setting up camera and render parameters
    context.scene.render.filepath = render_out_filename
    context.scene.render.resolution_x = image_resolution_x
    context.scene.render.resolution_y = image_resolution_y
    context.scene.render.resolution_percentage = image_resolution_scale
    context.scene.render.image_settings.file_format = file_format
    context.scene.render.image_settings.compression = image_compression
    context.scene.render.image_settings.quality = image_compression
    context.scene.render.image_settings.color_mode = image_color_mode
    context.scene.render.image_settings.color_depth = str(image_bit_depth)

    # Rendering the image to the saved location
    bpy.ops.render.render(write_still=True)

    # Saving the scene to device
    scene_out_file_name = None
    if save_raw == 1:
        scene_out_file_name = out_dir + os.sep + object_name + '.blend'
        print('Saving scene to: ' + scene_out_file_name)
        bpy.ops.wm.save_as_mainfile(filepath=scene_out_file_name)

    # Updating the scene, in case this runs not in headless mode
    bpy.context.view_layer.update()
    print('Render done.')

    return render_out_filename, scene_out_file_name


# Clear all nodes in a material
def clear_material(material):
    if material.node_tree:
        material.node_tree.links.clear()
        material.node_tree.nodes.clear()


# Create a node corresponding to a defined group
def instantiate_group(nodes, group_name):
    group = nodes.new(type='ShaderNodeGroup')
    group.node_tree = bpy.data.node_groups[group_name]


if __name__ == '__main__':
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]

        parser = argparse.ArgumentParser(
            description='Loads a .stl 3D model into blender and renders it to a bitmap image file.')

        # Required Arguments
        # Paths:
        parser.add_argument('-model_file_path', required=True, type=str, help='Path to the model to use.')
        parser.add_argument('-out_dir', required=True, type=str, help='Path to the directory to save the renders in.')

        # Optional Arguments
        # Raw:
        parser.add_argument('-save_raw', type=int, default=1, help='Save the .blend file after rendering.')

        # Camera Position
        parser.add_argument('-camera_pos_x', type=float, default=-2.0, help='Original camera x-position.')
        parser.add_argument('-camera_pos_y', type=float, default=3.0, help='Original camera y-position.')
        parser.add_argument('-camera_pos_z', type=float, default=3.0, help='Original camera z-position.')

        # Camera Angle
        parser.add_argument('-camera_angle_x', type=float, default=422.0, help='Original camera x-euler-angle.')
        parser.add_argument('-camera_angle_y', type=float, default=0.0, help='Original camera y-euler-angle.')
        parser.add_argument('-camera_angle_z', type=float, default=149.0, help='Original camera z-euler-angle.')

        # Background Color
        parser.add_argument('-background_r', type=float, default=1.0,
                            help='Background color: Red. Expected values: 0.0-1.0')
        parser.add_argument('-background_g', type=float, default=1.0,
                            help='Background color: Green. Expected values: 0.0-1.0')
        parser.add_argument('-background_b', type=float, default=1.0,
                            help='Background color: Blue. Expected values: 0.0-1.0')
        parser.add_argument('-background_a', type=float, default=1.0,
                            help='Background color: Alpha (Experimental). Expected values: 0.0-1.0')
        parser.add_argument('-background_intensity', type=float, default=1.2, help='Background color intensity.')

        # Mesh Color and Material
        parser.add_argument('-mesh_r', type=float, default=0.425,
                            help='3D Object: Mesh color: Red Expected values: 0.0-1.0.')
        parser.add_argument('-mesh_g', type=float, default=0.002,
                            help='3D Object: Mesh color: Green Expected values: 0.0-1.0.')
        parser.add_argument('-mesh_b', type=float, default=0.0013,
                            help='3D Object: Mesh color: Blue Expected values: 0.0-1.0.')
        parser.add_argument('-mesh_a', type=float, default=1.0,
                            help='3D Object: Mesh color: Alpha (Experimental). Expected values: 0.0-1.0.')
        parser.add_argument('-mesh_roughness', type=float, default=0.65,
                            help='3D Object: Mesh roughness. Expected values: 0.0-1.0.')

        # Rendering & Hardware Parameter
        parser.add_argument('-render_engine', type=str, default='CYCLES',
                            help='Rendering engine chosen. Expected Values: ' + str(expected_render_engine))
        parser.add_argument('-render_device', type=str, default='GPU',
                            help='Hardware device to render. Expected Values: ' + str(expected_render_device))
        parser.add_argument('-render_distance', type=int, default=500000,
                            help='Render distance. If your object is too large, increase this value.')

        # Rendered Image Params
        parser.add_argument('-image_resolution_x', type=int, default=960, help='Output image pixel resolution: x')
        parser.add_argument('-image_resolution_y', type=int, default=540, help='Output image pixel resolution: y')
        parser.add_argument('-image_resolution_scale', type=float, default=100.0,
                            help='Percentage image resolution scaling.')
        parser.add_argument('-image_color_mode', type=str, default='RGBA',
                            help='Output image color mode. Expected values: ' + str(expected_image_color_mode))
        parser.add_argument('-image_bit_depth', type=int, default=8,
                            help='Output image bit depth. Expected values: ' + str(expected_image_bit_depth))
        parser.add_argument('-file_format', type=str, default='PNG',
                            help='Output image file format. Expected values: ' + str(expected_file_format))
        parser.add_argument('-image_compression', type=float, default=0.0,
                            help='Output image compression. Expected values: 0.0-1.0.')

        # Parsing the arguments
        args = parser.parse_known_args(argv)[0]

        # Starting the algorithm
        render_object(model_file_path=args.model_file_path,
                      out_dir=args.out_dir,
                      save_raw=args.save_raw,
                      camera_pos_x=args.camera_pos_x,
                      camera_pos_y=args.camera_pos_y,
                      camera_pos_z=args.camera_pos_z,
                      camera_euler_angle_x=args.camera_angle_x,
                      camera_euler_angle_y=args.camera_angle_y,
                      camera_euler_angle_z=args.camera_angle_z,
                      background_r=args.background_r,
                      background_g=args.background_g,
                      background_b=args.background_b,
                      background_a=args.background_a,
                      background_intensity=args.background_intensity,
                      mesh_r=args.mesh_r,
                      mesh_g=args.mesh_g,
                      mesh_b=args.mesh_b,
                      mesh_a=args.mesh_a,
                      mesh_roughness=args.mesh_roughness,
                      render_engine=args.render_engine,
                      render_device=args.render_device,
                      render_distance=args.render_distance,
                      image_resolution_x=args.image_resolution_x,
                      image_resolution_y=args.image_resolution_y,
                      image_resolution_scale=args.image_resolution_scale,
                      image_color_mode=args.image_color_mode,
                      image_bit_depth=args.image_bit_depth,
                      file_format=args.file_format,
                      image_compression=args.image_compression)
