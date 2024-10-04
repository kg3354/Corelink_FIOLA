# Corelink_FIOLA Instructions

Computational steering is a cutting-edge research workflow that enhances innovation potential of scientific studies. This involves “closing the loop” on the research workflow, allowing for adjustment of parameters during the experiment at very low latencies using the latest technologies that allow for real-time research computing jobs. With the introduction of new High Performance Computing (HPC) resources into the Center for Neuroscience building such as the High Speed Research Network (HSRN) and HPCs edge Kubernetes cluster, we hope to explore this new approach to research for the department. Our goal is to develop and deploy a pipeline that processes calcium imaging data captured in real time from devices monitoring neuronal activity in rats, to detect and visualize the rats' ability to react to their surrounding environments, including brightness, moisture, smell, and beyond.


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

Dockerfile can also be referred to in order to use the processing side logic.


# DAQ side:
- watch_and_send.py
# Kubernetes side:
- receive_then_fiola.py
- receive_then_init.py
- generate_init_result.py
- Dockerfile
- fiola-process.yaml


