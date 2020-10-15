configfile: "config.json"

# IMGS, = glob_wildcards(expand("{folder}{img}.tif", folder=config["global"]["imgFolder"]))
#
# rule all:
#     input: expand("{folder}/{img}/{img}.tif", img=IMGS, folder=config["global"]["imgFolder"])
#
# rule makeImgDir:
#     input: expand("{folder}/{img}.tif", folder=config["global"]["imgFolder"])
#     output: expand("{folder}/{img}/{img}.tif", folder=config["global"]["imgFolder"])
#     shell: "mv {input} {output}"

# IMGS, = glob_wildcards("/bph/puredata1/bioinfdata/user/phispa/Test/{img}.tif")
#
# rule all:
#     input: expand("/bph/puredata1/bioinfdata/user/phispa/Test/{img}/{img}.tif", img=IMGS)
#
# rule makeImgDir:
#     input: "/bph/puredata1/bioinfdata/user/phispa/Test/{img}.tif"
#     output: "/bph/puredata1/bioinfdata/user/phispa/Test/{img}/{img}.tif"
#     shell: "mv {input} {output}"


IMGS, = glob_wildcards("data/{img}.tif")

rule all:
    input: expand("data/{img}/Statistics", img=IMGS)

rule makeImgDir:
    input: "data/{img}.tif"
    output: "data/{img}/{img}.tif"
    shell: "mv {input} {output}"

rule frangi:
    input: "data/{img}/{img}.tif"
    output: "data/{img}/Frangi_{img}.tiff"
    conda: "Envs/PVAP.yml"
    benchmark: "data/benchmarks/{img}.frangi.benchmark.txt"
    shell:
        """
            python frangi.py -i {input} -sigma_min {config[frangi][sigma_min]} -sigma_max {config[frangi][sigma_max]} \
            -sigma_steps {config[frangi][sigma_steps]} -alpha {config[frangi][alpha]} -beta {config[frangi][beta]} \
            -gamma {config[frangi][gamma]}
        """

rule threshold:
    input: "data/{img}/Frangi_{img}.tiff"
    output: "data/{img}/Binary_{img}.tif"
    conda: "Envs/Thresholding.yml"
    benchmark: "data/benchmarks/{img}.threshold.benchmark.txt"
    shell:
        """
            python thresholding.py -i {input} -ball_radius {config[threshold][ball_radius]} \
            -artifact_size {config[threshold][artifact_size]}
        """

rule skeletonize:
    input: "data/{img}/Binary_{img}.tif"
    output: "data/{img}/Skeleton_{img}.tif"
    conda: "Envs/ClearMap.yml"
    benchmark: "data/benchmarks/{img}.skeletonize.benchmark.txt"
    shell: "python skeletonize.py -i {input}"

rule graphAnalysis:
    input: "data/{img}/Skeleton_{img}.tif"
    output: "data/{img}/Statistics"
    conda: "Envs/PVAP.yml"
    benchmark: "data/benchmarks/{img}.graphAnalysis.benchmark.txt"
    shell:
        """
            python graphAnalysis.py -i {input} -pixel_dimensions {config[graphAnalysis][pixel_dimensions]} \
            -pruning_scale {config[graphAnalysis][pruning_scale]} -length_limit {config[graphAnalysis][length_limit]}
        """


