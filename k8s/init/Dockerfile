# # Use an official Python runtime as a parent image
# FROM python:3.8

# # Set the working directory in the container
# WORKDIR /usr/src/app

# # Install kubectl and other dependencies
# RUN apt-get update && apt-get install -y curl && \
#     curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
#     install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
#     apt-get install -y git && \
#     apt-get install -y libsm6 libxext6 libgl1-mesa-glx && \
#     pip install --upgrade pip && \
#     pip install --upgrade setuptools six && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# # Copy the requirements and pickle files
# COPY requirements.txt .
# COPY generate_init_result.py .
# COPY test_sub.tif . 

# # Download FIOLA and CaImAn
# RUN git clone https://github.com/nel-lab/FIOLA.git && \
#     pip install -r FIOLA/requirements.txt && \
#     pip install -e FIOLA

# RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
#     pip install -e CaImAn

# # Install packages from requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Install telnet
# RUN apt-get update && apt-get install -y telnet
 
# # The command to run the script, you'll overwrite this in Kubernetes deployment
# CMD ["python", "./generate_init_result.py"]
# Use an official CUDA runtime with Ubuntu 20.04 and Python 3.8
FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python 3.8 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    build-essential \
    curl \
    git \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# # Install kubectl
# RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
#     install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Copy the requirements and pickle files
COPY requirements.txt .
COPY generate_init_result.py .
COPY test_sub.tif . 

# Install Python packages
RUN python3.8 -m pip install --upgrade pip setuptools six && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt

# Download and install FIOLA and CaImAn
RUN git clone https://github.com/nel-lab/FIOLA.git && \
    python3.8 -m pip install -r FIOLA/requirements.txt && \
    python3.8 -m pip install -e FIOLA

RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
    python3.8 -m pip install -e CaImAn

# Install telnet
RUN apt-get update && apt-get install -y telnet

# The command to run the script, you'll overwrite this in Kubernetes deployment
CMD ["python3.8", "./generate_init_result.py"]
