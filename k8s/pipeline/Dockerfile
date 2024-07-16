FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

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
    libboost-all-dev \
    libssl-dev \
    libffi-dev \
    libtool \
    pkg-config \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libcudnn8=8.0.5.39-1+cuda11.0 \
    libcudnn8-dev=8.0.5.39-1+cuda11.0 \
    libcupti-dev \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py && rm get-pip.py

# Install PyCUDA
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"
RUN pip3 install pycuda --verbose

# Add NVIDIA package repositories
RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - \
    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list \
    && apt-get update

# Install nvidia-container-toolkit
RUN apt-get update && apt-get install -y nvidia-container-toolkit

# Install Python packages
COPY requirements.txt .
RUN pip3 install --upgrade pip setuptools six && \
    pip3 install --no-cache-dir -r requirements.txt

# Install specific version of TensorFlow compatible with CUDA 11.0
RUN pip3 install tensorflow==2.4.1 typing-extensions==3.7.4.3 pytools==2020.4.4 

# Download and install FIOLA
RUN git clone https://github.com/nel-lab/FIOLA.git && \
    cd FIOLA && \
    pip3 install -r requirements.txt && \
    pip3 install -e . && \
    cd ..

# Download and install CaImAn
RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
    cd CaImAn && \
    pip3 install -r requirements.txt && \
    pip3 install -r requirements_gpu.txt && \
    pip3 install -e . && \
    cd ..

# Set PYTHONPATH to include FIOLA and CaImAn
ENV PYTHONPATH="/usr/src/app/FIOLA:/usr/src/app/CaImAn:/usr/local/lib/python3.8/dist-packages:${PYTHONPATH}"

# Set LD_LIBRARY_PATH to include CUDA and cuDNN libraries
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu"

# Copy the script and data files
COPY generate_init_result.py .
COPY receive_then_init.py .
COPY receive_then_fiola.py .


# The command to run the script, you'll overwrite this in Kubernetes deployment
CMD ["python3.8", "./receive_then_fiola.py"]
