# # # # # # # # Use an official Python runtime as a parent image
# # # # # # # FROM python:3.8

# # # # # # # # Set the working directory in the container
# # # # # # # WORKDIR /usr/src/app

# # # # # # # # Install kubectl and other dependencies
# # # # # # # RUN apt-get update && apt-get install -y curl && \
# # # # # # #     curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
# # # # # # #     install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
# # # # # # #     apt-get install -y git && \
# # # # # # #     apt-get install -y libsm6 libxext6 libgl1-mesa-glx && \
# # # # # # #     pip install --upgrade pip && \
# # # # # # #     pip install --upgrade setuptools six && \
# # # # # # #     apt-get clean && \
# # # # # # #     rm -rf /var/lib/apt/lists/*

# # # # # # # # Copy the requirements and pickle files
# # # # # # # COPY requirements.txt .
# # # # # # # COPY generate_init_result.py .
# # # # # # # COPY test_sub.tif . 

# # # # # # # # Download FIOLA and CaImAn
# # # # # # # RUN git clone https://github.com/nel-lab/FIOLA.git && \
# # # # # # #     pip install -r FIOLA/requirements.txt && \
# # # # # # #     pip install -e FIOLA

# # # # # # # RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
# # # # # # #     pip install -e CaImAn

# # # # # # # # Install packages from requirements.txt
# # # # # # # RUN pip install --no-cache-dir -r requirements.txt

# # # # # # # # Install telnet
# # # # # # # RUN apt-get update && apt-get install -y telnet
 
# # # # # # # # The command to run the script, you'll overwrite this in Kubernetes deployment
# # # # # # # CMD ["python", "./generate_init_result.py"]
# # # # # # # Use an official CUDA runtime with Ubuntu 20.04 and Python 3.8
# # # # # # FROM nvidia/cuda:11.2.0-cudnn8-runtime-ubuntu20.04

# # # # # # # Set the working directory in the container
# # # # # # WORKDIR /usr/src/app

# # # # # # # Install Python 3.8 and other dependencies
# # # # # # RUN apt-get update && apt-get install -y \
# # # # # #     python3.8 \
# # # # # #     python3.8-venv \
# # # # # #     python3.8-dev \
# # # # # #     curl \
# # # # # #     git \
# # # # # #     libsm6 \
# # # # # #     libxext6 \
# # # # # #     libgl1-mesa-glx \
# # # # # #     && apt-get clean \
# # # # # #     && rm -rf /var/lib/apt/lists/*

# # # # # # # Install pip for Python 3.8
# # # # # # RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# # # # # # # Install kubectl
# # # # # # RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
# # # # # #     install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# # # # # # # Copy the requirements and pickle files
# # # # # # COPY requirements.txt .
# # # # # # COPY generate_init_result.py .
# # # # # # COPY test_sub.tif . 

# # # # # # # Install Python packages
# # # # # # RUN python3.8 -m pip install --upgrade pip setuptools six && \
# # # # # #     python3.8 -m pip install --no-cache-dir -r requirements.txt

# # # # # # # Download and install FIOLA and CaImAn
# # # # # # RUN git clone https://github.com/nel-lab/FIOLA.git && \
# # # # # #     python3.8 -m pip install -r FIOLA/requirements.txt && \
# # # # # #     python3.8 -m pip install -e FIOLA

# # # # # # RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
# # # # # #     python3.8 -m pip install -e CaImAn


# # # # # # # The command to run the script, you'll overwrite this in Kubernetes deployment
# # # # # # CMD ["python3.8", "./generate_init_result.py"]
# # # # # # Use an official CUDA runtime with Python 3.8
# # # # # FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

# # # # # # Set the working directory in the container
# # # # # WORKDIR /usr/src/app

# # # # # # Install Python 3.8 and other dependencies
# # # # # RUN apt-get update && apt-get install -y \
# # # # #     python3.8 \
# # # # #     python3.8-venv \
# # # # #     python3.8-dev \
# # # # #     build-essential \
# # # # #     curl \
# # # # #     git \
# # # # #     libsm6 \
# # # # #     libxext6 \
# # # # #     libgl1-mesa-glx \
# # # # #     && apt-get clean \
# # # # #     && rm -rf /var/lib/apt/lists/*

# # # # # # Install pip for Python 3.8
# # # # # RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# # # # # # Install kubectl
# # # # # RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
# # # # #     install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# # # # # # Copy the requirements and scripts files
# # # # # COPY requirements.txt .
# # # # # COPY generate_init_result.py .
# # # # # COPY test_sub.tif .

# # # # # # Install Python packages from requirements.txt
# # # # # RUN python3.8 -m pip install --upgrade pip setuptools six && \
# # # # #     python3.8 -m pip install --no-cache-dir -r requirements.txt

# # # # # # Download and install FIOLA
# # # # # RUN git clone https://github.com/nel-lab/FIOLA.git && \
# # # # #     python3.8 -m pip install -r FIOLA/requirements.txt && \
# # # # #     python3.8 -m pip install -e FIOLA

# # # # # # Download and install CaImAn
# # # # # RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
# # # # #     python3.8 -m pip install -r CaImAn/requirements.txt && \
# # # # #     python3.8 -m pip install -e CaImAn

# # # # # # Install telnet
# # # # # RUN apt-get update && apt-get install -y telnet

# # # # # # The command to run the script, you'll overwrite this in Kubernetes deployment
# # # # # CMD ["python3.8", "./generate_init_result.py"]
# # # # # Use an official CUDA runtime with Python 3.8
# # # FROM nvidia/cuda:12.0.0-runtime-ubuntu20.04

# # # # Set the working directory in the container
# # # WORKDIR /usr/src/app

# # # # Set environment variables to configure tzdata non-interactively
# # # ENV DEBIAN_FRONTEND=noninteractive
# # # ENV TZ=Etc/UTC

# # # # Install Python 3.8 and other dependencies
# # # RUN apt-get update && apt-get install -y \
# # #     python3.8 \
# # #     python3.8-venv \
# # #     python3.8-dev \
# # #     build-essential \
# # #     curl \
# # #     git \
# # #     libsm6 \
# # #     libxext6 \
# # #     libgl1-mesa-glx \
# # #     libglib2.0-0 \
# # #     && apt-get clean \
# # #     && rm -rf /var/lib/apt/lists/*

# # # # Install pip for Python 3.8
# # # RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# # # # Copy the requirements and scripts files
# # # COPY requirements.txt .
# # # COPY generate_init_result.py .
# # # COPY test_sub.tif .

# # # # Install Python packages from requirements.txt
# # # RUN pip install --upgrade pip setuptools six && \
# # #     pip install --no-cache-dir -r requirements.txt

# # # # Download and install FIOLA
# # # RUN git clone https://github.com/nel-lab/FIOLA.git && \
# # #     cd FIOLA && \
# # #     pip install -r requirements.txt && \
# # #     pip install -e . && \
# # #     cd ..

# # # # Download and install CaImAn
# # # RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
# # #     cd CaImAn && \
# # #     pip install -r requirements.txt && \
# # #     pip install -e . && \
# # #     cd ..

# # # # Set PYTHONPATH to include FIOLA and CaImAn
# # # ENV PYTHONPATH="/usr/src/app/FIOLA:/usr/src/app/CaImAn:/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"



# # # # The command to run the script, you'll overwrite this in Kubernetes deployment
# # # CMD ["python", "./generate_init_result.py"]



# # # # Use an official Python runtime as a parent image
# # # # FROM python:3.8

# # # # # Set the working directory in the container
# # # # WORKDIR /usr/src/app

# # # # # Install CUDA and other dependencies
# # # # RUN apt-get update && \
# # # #     apt-get install -y --no-install-recommends \
# # # #     cuda-command-line-tools-11-2 \
# # # #     libcudnn8=8.1.1.33-1+cuda11.2 \
# # # #     libcudnn8-dev=8.1.1.33-1+cuda11.2 \
# # # #     libnccl2=2.8.4-1+cuda11.2 \
# # # #     libnccl-dev=2.8.4-1+cuda11.2 \
# # # #     curl \
# # # #     git \
# # # #     libsm6 \
# # # #     libxext6 \
# # # #     libgl1-mesa-glx && \
# # # #     apt-get clean && \
# # # #     rm -rf /var/lib/apt/lists/*

# # # # # Set environment variables for CUDA
# # # # ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# # # # ENV PATH /usr/local/cuda/bin:$PATH

# # # # # Install pip for Python 3.8
# # # # RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# # # # # Install kubectl
# # # # RUN curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
# # # #     install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# # # # # Copy the requirements and scripts files
# # # # COPY requirements.txt .
# # # # COPY generate_init_result.py .
# # # # COPY test_sub.tif .

# # # # # Install Python packages
# # # # RUN pip install --upgrade pip setuptools six && \
# # # #     pip install --no-cache-dir -r requirements.txt

# # # # # Download and install FIOLA
# # # # RUN git clone https://github.com/nel-lab/FIOLA.git && \
# # # #     pip install -r FIOLA/requirements.txt && \
# # # #     pip install -e FIOLA

# # # # # Download and install CaImAn
# # # # RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
# # # #     pip install -r CaImAn/requirements.txt && \
# # # #     pip install -e CaImAn

# # # # # Print installed packages and Python path for debugging
# # # # RUN pip list && \
# # # #     python -c "import sys; print(sys.path)"

# # # # # The command to run the script, you'll overwrite this in Kubernetes deployment
# # # # CMD ["python3", "./generate_init_result.py"]
# # FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

# # # Set the working directory in the container
# # WORKDIR /usr/src/app

# # # Set environment variables to configure tzdata non-interactively
# # ENV DEBIAN_FRONTEND=noninteractive
# # ENV TZ=Etc/UTC

# # # Install Python 3.8 and other dependencies
# # RUN apt-get update && apt-get install -y \
# #     python3.8 \
# #     python3.8-venv \
# #     python3.8-dev \
# #     build-essential \
# #     curl \
# #     git \
# #     libsm6 \
# #     libxext6 \
# #     libgl1-mesa-glx \
# #     libglib2.0-0 \
# #     && apt-get clean \
# #     && rm -rf /var/lib/apt/lists/*

# # # Install pip for Python 3.8
# # RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py




# # # Copy the requirements and scripts files
# # COPY requirements.txt .
# # COPY generate_init_result.py .
# # COPY test_sub.tif .

# # # Install Python packages from requirements.txt
# # RUN pip install --upgrade pip setuptools six && \
# #     pip install --no-cache-dir -r requirements.txt

# # # Download and install FIOLA
# # RUN git clone https://github.com/nel-lab/FIOLA.git && \
# #     cd FIOLA && \
# #     pip install -r requirements.txt && \
# #     pip install -e . && \
# #     cd ..

# # # Download and install CaImAn
# # RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
# #     cd CaImAn && \
# #     pip install -r requirements.txt && \
# #     pip install -e . && \
# #     cd ..

# # # Set PYTHONPATH to include FIOLA and CaImAn
# # ENV PYTHONPATH="/usr/src/app/FIOLA:/usr/src/app/CaImAn:/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"

# # # The command to run the script, you'll overwrite this in Kubernetes deployment
# # CMD ["python", "./generate_init_result.py"]
# FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

# # Set the working directory in the container
# WORKDIR /usr/src/app

# # Set environment variables to configure tzdata non-interactively
# ENV DEBIAN_FRONTEND=noninteractive
# ENV TZ=Etc/UTC

# # Install Python 3.8 and other dependencies
# RUN apt-get update && apt-get install -y \
#     python3.8 \
#     python3.8-venv \
#     python3.8-dev \
#     python3.8-distutils \
#     build-essential \
#     curl \
#     git \
#     libsm6 \
#     libxext6 \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

# # Install pip for Python 3.8
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# # Add the NVIDIA package repositories
# RUN apt-get update && apt-get install -y gnupg && \
#     curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
#     curl -sL https://nvidia.github.io/nvidia-docker/ubuntu20.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
#     apt-get update && \
#     apt-get install -y nvidia-container-toolkit

# # Install cuDNN
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libcudnn8 \
#     libcudnn8-dev \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*


# # Copy the requirements and scripts files
# COPY requirements.txt .
# COPY generate_init_result.py .
# COPY test_sub.tif .

# # Install Python packages from requirements.txt
# RUN pip install --upgrade pip setuptools six && \
#     pip install --no-cache-dir -r requirements.txt

# # Download and install FIOLA
# RUN git clone https://github.com/nel-lab/FIOLA.git && \
#     cd FIOLA && \
#     pip install -r requirements.txt && \
#     pip install -e . && \
#     cd ..

# # Download and install CaImAn
# RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
#     cd CaImAn && \
#     pip install -r requirements.txt && \
#     pip install -e . && \
#     cd ..

# # Set PYTHONPATH to include FIOLA and CaImAn
# ENV PYTHONPATH="/usr/src/app/FIOLA:/usr/src/app/CaImAn:/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"

# # Set LD_LIBRARY_PATH to include CUDA and cuDNN libraries
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu

# # The command to run the script, you'll overwrite this in Kubernetes deployment
# CMD ["python", "./generate_init_result.py"]



FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Set environment variables to configure tzdata non-interactively
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install Python 3.8 and other dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    python3-distutils \
    build-essential \
    curl \
    git \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# Add the NVIDIA package repositories
RUN apt-get update && apt-get install -y gnupg && \
    curl -sL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -sL https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install -y nvidia-container-toolkit

# Install cuDNN
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn8=8.0.5.39-1+cuda11.0 \
    libcudnn8-dev=8.0.5.39-1+cuda11.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements and scripts files
COPY requirements.txt .
COPY generate_init_result.py .
COPY test_sub.tif .

# Install Python packages from requirements.txt
RUN pip install --upgrade pip setuptools six && \
    pip install --no-cache-dir -r requirements.txt

# Install specific version of TensorFlow compatible with CUDA 11.0
RUN pip install tensorflow==2.4.1

# Download and install FIOLA
RUN git clone https://github.com/nel-lab/FIOLA.git && \
    cd FIOLA && \
    pip install -r requirements.txt && \
    pip install -e . && \
    cd ..

# Download and install CaImAn
RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
    cd CaImAn && \
    pip install -r requirements.txt && \
    pip install -e . && \
    cd ..

# Set PYTHONPATH to include FIOLA and CaImAn
ENV PYTHONPATH="/usr/src/app/FIOLA:/usr/src/app/CaImAn:/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"

# Set LD_LIBRARY_PATH to include CUDA and cuDNN libraries
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu

# The command to run the script, you'll overwrite this in Kubernetes deployment
CMD ["python", "./generate_init_result.py"]
