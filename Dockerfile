FROM python:3.7
RUN python3.7 -m pip install snakemake==6.7.0

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get update
RUN apt-get upgrade

RUN apt-get install blender -y
RUN wget https://download.blender.org/release/Blender2.83/blender-2.83.4-linux64.tar.xz
RUN tar xvf blender-2.83.4-linux64.tar.xz -C /usr/bin/

ENV PATH /opt/conda/bin:$PATH

WORKDIR /home/user/

COPY . /home/user/VesselExpress

WORKDIR /home/user/VesselExpress/

RUN ln -sf /bin/bash /bin/sh

RUN conda update conda

RUN snakemake --use-conda --cores all --conda-frontend conda --conda-create-envs-only

CMD snakemake --use-conda --cores all --conda-frontend conda