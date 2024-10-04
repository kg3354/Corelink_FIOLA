# Corelink_FIOLA Instructions

To use Corelink, FIOLA pipeline, and CaImAn locally, please refer to the files under Corelink_FIOLA main directory.
For Kubernetes support, refer to k8s/pipeline.

## Initialization:

To set up Corelink_FIOLA pipeline locally, please follow the following
``` 
git clone https://github.com/kg3354/Corelink_FIOLA.git
git clone https://github.com/nel-lab/FIOLA.git
git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13

conda create --name CF python==3.8
conda activate CF

cd Corelink_FIOLA
pip install -r requirements.txt
cd FIOLA
pip install -e .
cd ../CaImAn
pip install -e . 
cd ..
```

The sender side should use watch_and_send.py, and the receiver should use receive_then_fiola.py.


# Files to review:

- receive_then_fiola.py
- receive_then_init.py
- generate_init_result.py
- Dockerfile
- fiola-process.yaml

## Current bottole neck:
- Offline Motion Correction
- Online FIOLA Process using CPU
