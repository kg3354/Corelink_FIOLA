# Initialization:
``` 

git clone https://github.com/kg3354/Corelink_FIOLA.git
git clone https://github.com/nel-lab/FIOLA.git
git clone https://github.com/flatironinstitute/CaImAn.git -b v1.9.13

conda create --name Corelink_FIOLA python==3.8
conda activate Corelink_FIOLA

pip install -r requirements.txt
cd FIOLA
pip install -r requirements.txt 
pip install -e .
cd ../CaImAn
pip install -e . 
cd ..

```


