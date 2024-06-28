# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory in the container
WORKDIR /usr/src/app

# Install kubectl and other dependencies
RUN apt-get update && apt-get install -y curl && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && \
    apt-get install -y git && \
    apt-get install -y libsm6 libxext6 libgl1-mesa-glx && \
    pip install --upgrade pip && \
    pip install --upgrade setuptools six && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements and pickle files
COPY requirements.txt .
COPY fiola_state_msCam.pkl .
COPY receive_then_fiola.py .

# Download FIOLA and CaImAn
RUN git clone https://github.com/nel-lab/FIOLA.git && \
    pip install -r FIOLA/requirements.txt && \
    pip install -e FIOLA

RUN git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13 && \
    pip install -e CaImAn

# Install packages from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install telnet
RUN apt-get update && apt-get install -y telnet

# The command to run the script, you'll overwrite this in Kubernetes deployment
CMD ["python", "./receive_then_fiola.py"]

# Volume for profile data
VOLUME /usr/src/app/profile_data
