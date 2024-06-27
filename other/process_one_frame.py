
#import caiman as cm
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import pyximport
pyximport.install()
from time import time
import pickle    
from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, to_2D, movie_iterator
from tensorflow.python.client import device_lib
import tifffile

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)    
logging.info(device_lib.list_local_devices())  # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0

startT = time()

def load_fiola_state(filepath):
    with open(filepath, 'rb') as f:
        fio_state = pickle.load(f)
    params = fio_state['params']
    trace_fiola = np.array(fio_state['trace_fiola'], dtype=np.float32)
    template = np.array(fio_state['template'], dtype=np.float32)
    Ab = np.array(fio_state['Ab'], dtype=np.float32)
    min_mov = fio_state['min_mov']
    mc_nn_mov = np.array(fio_state['mov_data'], dtype=np.float32)
    
    fio = FIOLA(params=params) 
    fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
    return fio

def main():
    mode = 'calcium'  # 'voltage' or 'calcium' fluorescence indicator
    # folder = 'C:/Users/Research/desktop/fiola/CaImAn/example_movies/frame_sample'
    # fnames = folder + '/combined_frames_15.tif'
    folder = '../results'
    fnames = '../trash/combined_output_20.tif'
    num_frames_init = 1   # number of frames used for initialization
    num_frames_total = 20 # estimated total number of frames for processing
    batch = 1  # number of frames processing at the same time using GPU 


    fio = load_fiola_state('fiola_state_msCam.pkl')

  
    online_trace = np.zeros((fio.Ab.shape[-1], num_frames_total - num_frames_init), dtype=np.float32)
    online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], num_frames_total - num_frames_init), dtype=np.float32)

    start = time()
 
    memmap_image = tifffile.memmap(fnames, mode='r')
    for idx in range(num_frames_init, num_frames_total, batch):
        frame_batch = memmap_image[idx:idx + batch].astype(np.float32)
       
      
        fio.fit_online_frame(frame_batch)
        online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx - batch:idx]
        online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx - batch - fio.params.retrieve['lag']:idx - fio.params.retrieve['lag']]
      
    fio.pipeline.saoz.online_trace = online_trace
    fio.pipeline.saoz.online_trace_deconvolved = online_trace_deconvolved
    logging.info(f'total time online: {time() - start}')


    fio.compute_estimates()

  


    # Save result
    if True:
        np.save(f"{folder}/fiola_result_pof_{time()}.npy", fio.estimates)
    logging.info(f'Total seconds used to process are: {time() - startT}')

if __name__ == "__main__":
    main()
