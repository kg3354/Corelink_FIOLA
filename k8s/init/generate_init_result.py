# #!/usr/bin/env python
# """
# Illustration of the usage of FIOLA with calcium and voltage imaging data. 
# For Calcium USE THE demo_initialize_calcium.py FILE TO GENERATE THE HDF5 files necessary for 
# initialize FIOLA. 
# For voltage this demo is self contained.   
# copyright in license file
# authors: @agiovann @changjia
# """
# #%%
# import caiman as cm
# import logging
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
# import pyximport
# pyximport.install()
# import scipy
# from tensorflow.python.client import device_lib
# from time import time
    
# from fiola.demo_initialize_calcium import run_caiman_init
# from fiola.fiolaparams import fiolaparams
# from fiola.fiola import FIOLA
# from fiola.utilities import download_demo, load, to_2D, movie_iterator

# logging.basicConfig(format=
#                     "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
#                     "[%(process)d] %(message)s",
#                     level=logging.INFO)    
# logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0
# #%% 
# def main():
# #%%
#     mode = 'calcium'                    # 'voltage' or 'calcium' fluorescence indicator
#     # Parameter setting
    
#     if mode == 'calcium':
#         # folder = cm.paths.caiman_datadir() + '/example_movies'
#         folder = 'C:/Users/29712/fiola/CaImAn/example_movies'
#         #fnames = folder + '/output_multi_frame99.tif'
#         #fnames = folder + '/output_multi_frame_big_endian.tif'
#         # fnames = folder + '/result_update270.tif'
#         fnames = folder + '/msCam_continuous.tif'
#         fr = 30                         # sample rate of the movie
        
#         mode = 'calcium'                # 'voltage' or 'calcium' fluorescence indicator
#         num_frames_init =   1000       # number of frames used for initialization
#                                          # estimated total number of frames for processing, this is used for generating matrix to store data
#         offline_batch = 5               # number of frames for one batch to perform offline motion correction
#         batch= 1                        # number of frames processing at the same time using gpu 
#         flip = False                    # whether to flip signal to find spikes   
#         detrend = False                 # whether to remove the slow trend in the fluorescence data
#         dc_param = 0.9995               # DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
#                                         # 0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
#         do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
#         ms = [55, 55]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
#         center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
#         hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
#                                         # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
#         n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
#         nb = 1                          # number of background components
#         trace_with_neg=True             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
#         lag = 5                         # lag for retrieving the online result.
                        
     
        
#         mov = cm.load(fnames, subindices=range(num_frames_init))
#         fnames_init = fnames.split('.')[0] + '_init.tif'
#         mov.save(fnames_init)
#         print('here')
#         # run caiman initialization. User might need to change the parameters 
#         # inside the file to get good initialization result
#         caiman_file = run_caiman_init(fnames_init, pw_rigid=True, 
#                                       max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
#         print(f'Caiman intialization file is ready: ' + caiman_file)
#         print()
        
# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
"""
Illustration of the usage of FIOLA with calcium and voltage imaging data. 
For Calcium USE THE demo_initialize_calcium.py FILE TO GENERATE THE HDF5 files necessary for 
initialize FIOLA. 
For voltage this demo is self contained.   
copyright in license file
authors: @agiovann @changjia
"""
#%%
import caiman as cm
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
import pyximport
pyximport.install()
import scipy
from tensorflow.python.client import device_lib
from time import time

import pickle

from fiola.demo_initialize_calcium import run_caiman_init
from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, to_2D, movie_iterator

logging.basicConfig(format=
                    "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
                    "[%(process)d] %(message)s",
                    level=logging.INFO)    
logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0
#%% 

def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
    fio_state = {
        'params': params,
        'trace_fiola': trace_fiola,
        'template': template,
        'Ab': Ab,
        'min_mov': min_mov,
        
        'mov_data': mc_nn_mov  # Convert the movie data to a list to make it serializable
    }
    with open(filepath, 'wb') as f:
        pickle.dump(fio_state, f)



#%% 
def main():
#%%
    mode = 'calcium'                    # 'voltage' or 'calcium' fluorescence indicator
    # Parameter setting
    
    if mode == 'calcium':
        # folder = cm.paths.caiman_datadir() + '/example_movies'
#        folder = 'C:/Users/29712/fiola/CaImAn/example_movies'
        folder = os.getenv('MOVIE_FOLDER', '/app')
        #fnames = folder + '/output_multi_frame99.tif'
        #fnames = folder + '/output_multi_frame_big_endian.tif'
        # fnames = folder + '/result_update270.tif'
        fnames = folder + '/test.tif'
        fr = 30                         # sample rate of the movie
        
        mode = 'calcium'                # 'voltage' or 'calcium' fluorescence indicator
        num_frames_init =   2000       # number of frames used for initialization
        num_frames_total =  30000        # estimated total number of frames for processing, this is used for generating matrix to store data
        offline_batch = 5               # number of frames for one batch to perform offline motion correction
        batch= 1                        # number of frames processing at the same time using gpu 
        flip = False                    # whether to flip signal to find spikes   
        detrend = False                 # whether to remove the slow trend in the fluorescence data
        dc_param = 0.9995               # DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
                                        # 0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
        do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
        # ms = [55,55]    
        ms = [3,3]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
        center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
        hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
                                        # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
        n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
        nb = 1                          # number of background components
        trace_with_neg=True             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
        lag = 5                         # lag for retrieving the online result.
                        
        options = {
            'fnames': fnames,
            'fr': fr,
            'mode': mode, 
            'num_frames_init': num_frames_init,     
            'num_frames_total':num_frames_total,
            'offline_batch': offline_batch,
            'batch':batch,
            'flip': flip,
            'detrend': detrend,
            'dc_param': dc_param,            
            'do_deconvolve': do_deconvolve,
            'ms': ms,
            'hals_movie': hals_movie,
            'center_dims':center_dims, 
            'n_split': n_split,
            'nb' : nb, 
            'trace_with_neg':trace_with_neg, 
            'lag': lag}
        
        mov = cm.load(fnames, subindices=range(num_frames_init))
        fnames_init = fnames.split('.')[0] + '_init.tif'
        mov.save(fnames_init)
        
        # run caiman initialization. User might need to change the parameters 
        # inside the file to get good initialization result
        caiman_file = run_caiman_init(fnames_init, pw_rigid=True, 
                                      max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
        logging.info(caiman_file)
        # load results of initialization
        cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
        estimates = cnm2.estimates
        template = cnm2.estimates.template
        Cn = cnm2.estimates.Cn
        logging.info('Finished run caiman init')   
    else: 
        raise Exception('mode must be either calcium')
          
    #%% Run FIOLA
    #example motion correction
    motion_correct = True
    #example source separation
    do_nnls = True
    #%% Mot corr only
    if motion_correct:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        # run motion correction on GPU on the initialization movie
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())             
        plt.plot(shifts_fiola)
        plt.xlabel('frames')
        plt.ylabel('pixels')                 
        plt.legend(['x shifts', 'y shifts'])
    else:    
        mc_nn_mov = mov
    logging.info('Finished mc')       
    #%% NNLS only
    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        Ab = np.hstack((estimates.A.toarray(), estimates.b))
        trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch']) 
        plt.plot(trace_fiola[:-nb].T)
        plt.xlabel('frames')
        plt.ylabel('fluorescence signal')              

    else:        
        if trace_with_neg == True:
            trace_fiola = np.vstack((estimates.C+estimates.YrA, estimates.f))
        else:
            trace_fiola = estimates.C+estimates.YrA
            trace_fiola[trace_fiola < 0] = 0
            trace_fiola = np.vstack((trace_fiola, estimates.f))
    logging.info('Finished nnls')   
    #%% set up online pipeline
    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    Ab = np.hstack((estimates.A.toarray(), estimates.b))
    Ab = Ab.astype(np.float32)        
    #fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())

   # After creating the FIOLA pipeline

# Save the FIOLA state
    save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, 'fiola_state_msCam.pkl')


if __name__ == "__main__":
    main()
