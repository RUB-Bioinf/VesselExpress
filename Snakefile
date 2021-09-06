import os

configfile: "./data/config.json"

PATH = config["imgFolder"]
IMGS = [os.path.splitext(f)[0] for f in os.listdir(PATH) if (f.endswith('tif') or f.endswith('.jpg'))
        and not f.startswith('.') and os.path.isfile(os.path.join(PATH, f))]

if config["skeletonization"] == "ClearMap":
    ruleorder: skeletonize_ClearMap > skeletonize_scikit
else:
    ruleorder: skeletonize_scikit > skeletonize_ClearMap

rule all:
    input: expand(PATH + "/{img}/Statistics", img=IMGS)

rule makeImgDir:
    input: PATH + "/{img}.tif"
    output: PATH + "/{img}/{img}.tif"
    shell: "mv {input} {output}"

rule jpgMakeImgDir:
    input: PATH + "/{img}.jpg"
    output: PATH + "/{img}/{img}.jpg"
    shell: "mv {input} {output}"

rule frangi:
    input: PATH + "/{img}/{img}.tif"
    output: PATH + "/{img}/Frangi_{img}.tiff"
    conda: "Envs/vmtk.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.frangi.benchmark.txt"
    shell:
        """
            python frangi.py -i {input} -sigma_min {config[frangi][sigma_min]} -sigma_max {config[frangi][sigma_max]} \
            -sigma_steps {config[frangi][sigma_steps]} -alpha {config[frangi][alpha]} -beta {config[frangi][beta]} \
            -gamma {config[frangi][gamma]}
        """

rule threshold:
    input: PATH + "/{img}/Frangi_{img}.tiff"
    output: PATH + "/{img}/Binary_{img}.tif"
    conda: "Envs/Pipeline.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.threshold.benchmark.txt"
    shell:
        """
            python thresholding.py -i {input} -pixel_dimensions {config[graphAnalysis][pixel_dimensions]} \
            -ball_radius {config[threshold][ball_radius]} -artifact_size {config[threshold][artifact_size]}
        """

rule threshold_2D_jpg:
    input: PATH + "/{img}/{img}.jpg"
    output: PATH + "/{img}/Binary_{img}.tif"
    conda: "Envs/Thresholding.yml"
    shell: "python threshold2D.py -i {input} -artifact_size {config[threshold][artifact_size]}"

rule skeletonize_ClearMap:
    input: PATH + "/{img}/Binary_{img}.tif"
    output: PATH + "/{img}/Skeleton_{img}.tif"
    conda: "Envs/ClearMap.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.skeletonize.benchmark.txt"
    shell: "python skeletonize_ClearMap.py -i {input}"

rule skeletonize_scikit:
    input: PATH + "/{img}/Binary_{img}.tif"
    output: PATH + "/{img}/Skeleton_{img}.tif"
    conda: "Envs/Pipeline.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.skeletonize.benchmark.txt"
    shell: "python skeletonize_scikit.py -i {input} -pixel_dimensions {config[graphAnalysis][pixel_dimensions]}"

rule graphAnalysis:
    input: PATH + "/{img}/Skeleton_{img}.tif"
    output: directory(PATH + "/{img}/Statistics")
    conda: "Envs/Pipeline.yml"
    #benchmark: PATH + "/{img}/benchmarks/{img}.graphAnalysis.benchmark.txt"
    shell:
        """
            python graphAnalysis.py -i {input} -pixel_dimensions {config[graphAnalysis][pixel_dimensions]} \
            -pruning_scale {config[graphAnalysis][pruning_scale]} \
            -length_limit {config[graphAnalysis][length_limit]} \
            -branching_threshold {config[graphAnalysis][branching_threshold]} \
            -all_stats {config[graphAnalysis][all_stats]} \
            -experimental_flag {config[graphAnalysis][experimental_flag]}
        """


