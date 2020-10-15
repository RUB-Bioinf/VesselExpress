# **Automated Analysis of 3D Light Sheet Brain Vessel Images**

This repository contains Python scripts for extracting features of 3D light sheet brain vessel images which have been
automated with the workflow management system [Snakemake](https://github.com/snakemake/snakemake).

The automated analysis pipeline contains the following steps:

1. Segmentation
2. Skeleton Extraction
3. Graph Construction
4. Statistical Analysis

The skeletonization of the [ClearMap](https://github.com/ChristophKirst/ClearMap2) repository is used to extract the
vessels' centerlines. This repository must be included into the project folder to run the pipeline via snakemake.
For further information please read their paper
[Mapping the Fine-Scale Organization and Plasticity of the Brain Vasculature](https://www.sciencedirect.com/science/article/abs/pii/S0092867420301094).
For the graph construction the Python script
[networkx_graph_from_array](https://github.com/3Scan/3scan-skeleton/blob/master/skeleton/networkx_graph_from_array.py)
from the
[3scan-skeleton repository](https://github.com/3Scan/3scan-skeleton#3d-image-skeletonization-tools) is used. This is
downloaded from GitHub and copied into the Graph folder. Many thanks to GitHub and the contributors!









 



