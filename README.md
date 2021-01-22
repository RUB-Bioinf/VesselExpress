[![GitHub stars](https://img.shields.io/github/stars/RUB-Bioinf/LightSheetBrainVesselSkeletonization.svg?style=social&label=Star)](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization) 

[![License](https://img.shields.io/github/license/RUB-Bioinf/LightSheetBrainVesselSkeletonization?color=green&style=plastic)](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/LICENSE.txt)
![Size](https://img.shields.io/github/repo-size/RUB-Bioinf/LightSheetBrainVesselSkeletonization?style=plastic)
[![Language](https://img.shields.io/github/languages/top/RUB-Bioinf/LightSheetBrainVesselSkeletonization?style=plastic)](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization)
[![](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/workflows/RepoTracker/badge.svg)](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/actions)

# **Automated Analysis of 3D Light Sheet Brain Vessel Images**

This repository contains Python scripts for extracting features of 3D light sheet brain vessel images which have been
automated with the workflow management system [Snakemake](https://github.com/snakemake/snakemake).

The automated analysis pipeline contains the following steps:

1. Segmentation
2. Skeleton extraction
3. Graph construction
4. Statistical analysis

***

![Pipeline](pipeline.png)

***

For processing larger images it is recommended to use the ClearMap skeletonization by Kirst et al. To use that please clone their
[repository](https://github.com/MartinFinkenflugel/ClearMap2/tree/3617414d6d56709b452b2c5253631eecbede1b85)
and copy the ClearMap folder into the projects root folder. For further information please read their 
[paper](https://www.sciencedirect.com/science/article/abs/pii/S0092867420301094).
For the graph construction the Python script
[networkx_graph_from_array](https://github.com/3Scan/3scan-skeleton/blob/master/skeleton/networkx_graph_from_array.py)
from the
[3scan-skeleton repository](https://github.com/3Scan/3scan-skeleton#3d-image-skeletonization-tools) is used. This is
downloaded from GitHub and copied into the Graph folder. Many thanks to GitHub and the contributors!

## Correspondence

[**Prof. Dr. Axel Mosig**](mailto:axel.mosig@rub.de): Bioinformatics, Center for Protein Diagnostics (ProDi), Ruhr-University Bochum, Bochum, Germany

http://www.bioinf.rub.de/

[**Prof. Dr. Matthias Gunzer**](mailto:matthias.gunzer@uni-due.de): Institute for Experimental Immunology and Imaging, University Hospital Essen, University of Duisburg-Essen, Essen, Germany

https://www.uni-due.de/experimental-immunology

# Download and Install

See the [Releases](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/releases) page on how to download the latest version of the pipeline.
Then refer to [this guide in the wiki](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/wiki/Running-the-Pipeline) on how to set up and run the pipeline.

## Example Data

Please follow [this guide](https://github.com/RUB-Bioinf/HT-PropagatedNeuriteSkeletonization/wiki/Example-Data) on how to download example data.

# Feedback & Bug Reports

We strive to always improve and make this pipeline accessible to the public.
We hope to make it as easy to use as possible.

Should you encounter an error, bug or need help, please feel free to reach out to us via the [Issues](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/issues) page.
Thank you for your help. Your feedback is much appreciated.

# Misc

Visit the [wiki](https://github.com/RUB-Bioinf/LightSheetBrainVesselSkeletonization/wiki) for additional information.

****

**Keywords**: stroke; Neuronal Vessel Morphology; High Throughput Light Sheet Microscopy; 3D Skeletonization

****

**Funding**: This research received no external funding.

**Conflicts of Interest**: The authors declare no conflict of interest
