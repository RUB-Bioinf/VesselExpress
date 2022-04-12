# Dockerfile for VesselExpress-Flask

# Pull base image
FROM python:3.7

# Pull conda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.3.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ENV VIRTUAL_ENV=/opt/venv
RUN python3.7 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN apt-get update
#RUN apt-get upgrade

# Setup blender
RUN apt-get install blender -y
RUN wget https://download.blender.org/release/Blender2.83/blender-2.83.4-linux64.tar.xz
RUN tar xvf blender-2.83.4-linux64.tar.xz -C /usr/bin/

# Copy project
COPY ./ /home/user/VesselExpress_server/

# Install website packages
WORKDIR /home/user/VesselExpress_server/
RUN pip install -r Webinterface/requirements.txt

# Setup conda
ENV PATH /opt/conda/bin:$PATH
RUN ln -sf /bin/bash /bin/sh
RUN conda update conda

# Setup envs for pipeline
RUN snakemake --use-conda --cores all --conda-frontend conda --snakefile "./VesselExpress/workflow/Snakefile" --conda-create-envs-only

RUN rm VesselExpress/data/test.tiff

EXPOSE 5000
CMD ["python3.7", "Webinterface/server/app.py"]
