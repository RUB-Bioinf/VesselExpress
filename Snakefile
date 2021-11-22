import os
import platform


def get_files_and_extensions(path):
    files = [os.path.splitext(f) for f in os.listdir(path) if (f.endswith('tif') or f.endswith('.tiff')
             or f.endswith('.jpg')) or f.endswith('.png') and not f.startswith('.')
             and os.path.isfile(os.path.join(path,f))]
    file_names = [f[0] for f in files]
    file_extensions = [f[1].split('.')[1] for f in files]

    new_file_extensions = []
    for file_name, file_ext in zip(file_names, file_extensions):
        if file_ext == 'tif':
            os.rename(os.path.join(path, file_name + '.tif'), os.path.join(os.path.join(path, file_name + '.tiff')))
            file_ext = 'tiff'
        new_file_extensions.append(file_ext)

    return file_names, new_file_extensions


def get_graphAnalysis_command(os):
    command_str = \
        "python graphAnalysis.py -skel_img \"{input.skelImg}\" -bin_img \"{input.binImg}\" \
        -output_dir \"{output}\" -pixel_dimensions {config[graphAnalysis][pixel_dimensions]} \
        -pruning_scale {config[graphAnalysis][pruning_scale]} -length_limit {config[graphAnalysis][length_limit]} \
        -branching_threshold {config[graphAnalysis][branching_threshold]} \
        -all_stats {config[graphAnalysis][all_stats]} -experimental_flag {config[graphAnalysis][experimental_flag]}"
    if os == 'Linux' or os == 'Darwin':
        command_str = command_str + "\nchmod ugo+rwx \"{output}\""
    return command_str


def get_input(imgs, exts, blender_exists, input_list=[]):
    input_list.append(expand(PATH + "/{img}/{img}.{ext}_Statistics", zip, img=imgs, ext=exts))
    if config['3D'] == 1 and config['render'] == 1 and blender_exists:
        input_list.append(expand(PATH + "/{img}/Binary_{img}-render.PNG", img=imgs))
        input_list.append(expand(PATH + "/{img}/Binary_{img}.blend",img=imgs))
        input_list.append(expand(PATH + "/{img}/Binary_{img}.glb",img=imgs))
        input_list.append(expand(PATH + "/{img}/Skeleton_{img}-render.PNG",img=imgs))
        input_list.append(expand(PATH + "/{img}/Skeleton_{img}.blend",img=imgs))
        input_list.append(expand(PATH + "/{img}/Skeleton_{img}.glb",img=imgs))
    return input_list


configfile: "data/config.json"

PATH = config["imgFolder"]
IMGS, EXTS = get_files_and_extensions(path=PATH)
OS = platform.system()

if OS == 'Linux':
    ENV_PATH = 'Envs/Linux/'
    BLENDER_PATH = '/usr/bin/blender'
else:
    ENV_PATH = 'Envs/Mac/'
    BLENDER_PATH = '/Applications/Blender.app/Contents/MacOS/Blender'
if os.path.exists(BLENDER_PATH):
    BLENDER = True
else:
    BLENDER = False

if config["skeletonization"] == "ClearMap" and OS == 'Linux':
    ruleorder: skeletonize_ClearMap > skeletonize_scikit
else:
    ruleorder: skeletonize_scikit > skeletonize_ClearMap

if config["3D"] == 1:
    ruleorder: frangi3D > frangi2D
else:
    ruleorder: frangi2D > frangi3D


rule all:
    input: get_input(IMGS, EXTS, BLENDER)

rule makeImgDir:
    input: PATH + "/{img}.{ext}"
    output: PATH + "/{img}/{img}.{ext}"
    shell: "mv \"{input}\" \"{output}\""

rule frangi2D:
    input: PATH + "/{img}/{img}.{ext}"
    output: PATH + "/{img}/Frangi_{img}.{ext}"
    conda: ENV_PATH + "Pipeline.yml"
    shell:
        """
            python frangi2D.py -i \"{input}\" \
            -sigma_min {config[frangi][sigma_min]} -sigma_max {config[frangi][sigma_max]} \
            -sigma_steps {config[frangi][sigma_steps]} -alpha {config[frangi][alpha]} -beta {config[frangi][beta]} \
            -gamma {config[frangi][gamma]} -denoise {config[threshold][denoise]}
        """

rule frangi3D:
    input: PATH + "/{img}/{img}.{ext}"
    output: PATH + "/{img}/Frangi_{img}.{ext}"
    conda: ENV_PATH + "vmtk.yml"
    # benchmark: PATH + "/{img}/benchmarks/{img}.frangi.benchmark.txt"
    shell:
        """
            python frangi3D.py -i \"{input}\" -sigma_min {config[frangi][sigma_min]} -sigma_max {config[frangi][sigma_max]} \
            -sigma_steps {config[frangi][sigma_steps]} -alpha {config[frangi][alpha]} -beta {config[frangi][beta]} \
            -gamma {config[frangi][gamma]}
        """

rule thresholding:
    input: PATH + "/{img}/Frangi_{img}.{ext}"
    output: PATH + "/{img}/Binary_{img}.{ext}"
    wildcard_constraints:
        ext="(tiff|png|jpg)"
    conda: ENV_PATH + "Thresholding.yml"
    # benchmark: PATH + "/{img}/benchmarks/{img}.threshold.benchmark.txt"
    shell:
        """
            python thresholding.py -i \"{input}\" -pixel_dimensions {config[graphAnalysis][pixel_dimensions]} \
            -ball_radius {config[threshold][ball_radius]} -artifact_size {config[threshold][artifact_size]} \
            -block_size {config[threshold][block_size]}
        """

rule renderBinary:
    input: PATH + "/{img}/Binary_{img}.stl"
    output: PATH + "/{img}/Binary_{img}-render.PNG", PATH + "/{img}/Binary_{img}.glb", PATH + "/{img}/Binary_{img}.blend"
    conda: ENV_PATH + "Pipeline.yml"
    shell:
            BLENDER_PATH + " --background --python render_object.py -- -model_file_path \"{input}\" -out_dir \"{PATH}/{wildcards.img}/\" \
            -save_raw {config[rendering][save_raw]} -render_device {config[rendering][render_device]} \
            -render_distance {config[rendering][render_distance]} \
            -image_resolution_x {config[rendering][image_resolution_x]} \
            -image_resolution_y {config[rendering][image_resolution_y]} \
            -image_compression {config[rendering][image_compression]}"

rule skeletonize_ClearMap:
    input: PATH + "/{img}/Binary_{img}.{ext}"
    output: PATH + "/{img}/Skeleton_{img}.{ext}"
    wildcard_constraints:
        ext="(tiff)"
    conda: ENV_PATH + "ClearMap.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.skeletonize.benchmark.txt"
    shell: "python skeletonize_ClearMap.py -i \"{input}\""

rule skeletonize_scikit:
    input: PATH + "/{img}/Binary_{img}.{ext}"
    output: PATH + "/{img}/Skeleton_{img}.{ext}"
    wildcard_constraints:
        ext="(tiff|png|jpg)"
    conda: ENV_PATH + "Pipeline.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.skeletonize.benchmark.txt"
    shell: "python skeletonize_scikit.py -i \"{input}\" -pixel_dimensions {config[graphAnalysis][pixel_dimensions]}"

rule renderSkeleton:
    input: PATH + "/{img}/Skeleton_{img}.stl"
    output: PATH + "/{img}/Skeleton_{img}-render.PNG", PATH + "/{img}/Skeleton_{img}.glb", PATH + "/{img}/Skeleton_{img}.blend"
    conda: ENV_PATH + "Pipeline.yml"
    shell:
            BLENDER_PATH + " --background --python render_object.py -- -model_file_path \"{input}\" -out_dir \"{PATH}/{wildcards.img}/\" \
            -save_raw {config[rendering][save_raw]} -render_device {config[rendering][render_device]} \
            -render_distance {config[rendering][render_distance]} \
            -image_resolution_x {config[rendering][image_resolution_x]} \
            -image_resolution_y {config[rendering][image_resolution_y]} \
            -image_compression {config[rendering][image_compression]}"

rule graphAnalysis:
    input: skelImg = PATH + "/{img}/Skeleton_{img}.{ext}", binImg = PATH + "/{img}/Binary_{img}.{ext}"
    output: directory(PATH + "/{img}/{img}.{ext}_Statistics/")
    conda: ENV_PATH + "Pipeline.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.graphAnalysis.benchmark.txt"
    shell: get_graphAnalysis_command(OS)

if config['3D'] == 1:
    rule createBinaryObj:
        input: PATH + "/{img}/Binary_{img}.tiff"
        output: PATH + "/{img}/Binary_{img}.stl"
        conda: ENV_PATH + "Pipeline.yml"
        shell:
            """
                python -W ignore create_stl.py -i \"{input}\" -o \"{output}\" -pixel_dimensions {config[graphAnalysis][pixel_dimensions]}
            """

    rule createSkeletonObj:
        input: PATH + "/{img}/Skeleton_{img}.tiff"
        output: PATH + "/{img}/Skeleton_{img}.stl"
        conda: ENV_PATH + "Pipeline.yml"
        shell:
            """
                python -W ignore create_stl.py -i \"{input}\" -o \"{output}\" -pixel_dimensions {config[graphAnalysis][pixel_dimensions]} \
                -dilation True
            """