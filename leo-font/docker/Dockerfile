# Use the appropriate CUDA and cuDNN base image
FROM nvidia/cuda:11.8.0-base-ubuntu20.04 AS base

# Set up some environment variables
ENV TORCH_VERSION=1.3.1
ENV TORCH_CUDA_ARCH_LIST="3.7;5.0;6.0;7.0+PTX"

# Install Miniconda to manage Python packages
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    git \
    mercurial \
    subversion \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

# Create a conda environment and install PyTorch
RUN conda create -y --name torch python=3.7 && \
    echo "source activate torch" > ~/.bashrc && \
    conda install -y pytorch=$TORCH_VERSION torchvision cudatoolkit=11.1 -c pytorch

# Set working directory
WORKDIR /app

# Install Python dependencies
FROM base AS dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Final image with Python dependencies and application code
FROM base AS final
COPY --from=dependencies /opt/conda /opt/conda
COPY --from=dependencies /usr/local/cuda /usr/local/cuda
COPY . /app

# Expose port 80 if needed
# EXPOSE 80

# Command to run your application
# For example:
# CMD ["python", "app.py"]
