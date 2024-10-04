#!/usr/bin/env python
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

from fiola.fiolaparams import fiolaparams
from fiola.fiola import FIOLA
from fiola.utilities import download_demo, load, to_2D, movie_iterator

import tensorflow as tf
tf.debugging.set_log_device_placement(True)

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU device found")

logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
                    level=logging.INFO)
logging.info(device_lib.list_local_devices())  # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0
startTime = time()
import cv2
import glob
from time import time
from caiman.motion_correction import MotionCorrect #Subject to change
from caiman.source_extraction.cnmf import cnmf as cnmf
from caiman.source_extraction.cnmf import params as params
from caiman.utils.utils import download_demo
from caiman.summary_images import local_correlations_movie_offline
from caiman.source_extraction.cnmf.utilities import get_file_size

#%%    f
def run_caiman_init(fnames, pw_rigid=True, max_shifts=[6, 6], gnb=2, rf=15, K=5, gSig=[4, 4]):
    c, dview, n_processes = cm.cluster.setup_cluster(
        backend='local', n_processes=None, single_thread=False)  # Adjust n_processes as needed

    timing = {}
    timing['start'] = time()

    # dataset dependent parameters
    display_images = False

    fr = 30  # imaging rate in frames per second
    decay_time = 0.4  # length of a typical transient in seconds
    dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
    patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
    strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
    overlaps = (24, 24)
    max_deviation_rigid = 3

    mc_dict = {
        'fnames': fnames,
        'fr': fr,
        'decay_time': decay_time,
        'dxy': dxy,
        'pw_rigid': pw_rigid,
        'max_shifts': max_shifts,
        'strides': strides,
        'overlaps': overlaps,
        'max_deviation_rigid': max_deviation_rigid,
        'border_nan': 'copy',
    }

    opts = params.CNMFParams(params_dict=mc_dict)

    # Motion correction and memory mapping
    time_init = time()
    motion_opts = opts.get_group('motion')
    # motion_opts['use_cuda'] = True

    print(motion_opts)


    logging.info(motion_opts)

    mc = MotionCorrect(fnames, dview=dview, **motion_opts)
    

    mc.motion_correct(save_movie=True)

    logging.info('Finished mc, before cm.save_memmap')
    border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
    fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders
    logging.info('Finished exclude borders')

    Yr, dims, T = cm.load_memmap(fname_new)
    images = np.reshape(Yr.T, [T] + list(dims), order='F')
    logging.info('Finished reshaping borders')

    # Restart cluster to clean up memory
    cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)  # Adjust n_processes as needed
    logging.info('Finished restarting cluster')

    f_F_mmap = mc.mmap_file[0]
    Cns = local_correlations_movie_offline(f_F_mmap, remove_baseline=True, window=1000, stride=1000, winSize_baseline=100, quantil_min_baseline=10, dview=dview)
    logging.info('Finished local_correlations_movie_offline')
    Cn = Cns.max(axis=0)
    Cn[np.isnan(Cn)] = 0
    plt.imshow(Cn, vmax=0.5)

    # Parameters for source extraction and deconvolution
    p = 1
    merge_thr = 0.85
    stride_cnmf = 6
    method_init = 'greedy_roi'
    ssub = 2
    tsub = 2

    opts_dict = {'fnames': fnames,
                 'p': p,
                 'fr': fr,
                 'nb': gnb,
                 'rf': rf,
                 'K': K,
                 'gSig': gSig,
                 'stride': stride_cnmf,
                 'method_init': method_init,
                 'rolling_sum': True,
                 'merge_thr': merge_thr,
                 'n_processes': n_processes,
                 'only_init': True,
                 'ssub': ssub,
                 'tsub': tsub}

    opts.change_params(params_dict=opts_dict)
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    logging.info('Finished CNFM on patches')

    # Component evaluation
    min_SNR = 1.0
    rval_thr = 0.75
    cnn_thr = 0.3
    cnn_lowest = 0.0

    cnm.params.set('quality', {'decay_time': decay_time,
                               'min_SNR': min_SNR,
                               'rval_thr': rval_thr,
                               'use_cnn': False,
                               'min_cnn_thr': cnn_thr,
                               'cnn_lowest': cnn_lowest})
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
    print(len(cnm.estimates.idx_components))
    time_patch = time()

    if display_images:
        cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

    cnm.estimates.select_components(use_object=True)
    cnm2 = cnm.refit(images, dview=dview)
    time_end = time()
    print('Total time until cnm refit is: ')
    print(time_end - time_init)

    min_SNR = 2
    rval_thr = 0.85
    cnn_thr = 0.15
    cnn_lowest = 0.0

    cnm2.params.set('quality', {'decay_time': decay_time,
                                'min_SNR': min_SNR,
                                'rval_thr': rval_thr,
                                'use_cnn': False,
                                'min_cnn_thr': cnn_thr,
                                'cnn_lowest': cnn_lowest})
    cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
    print(len(cnm2.estimates.idx_components))

    cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

    if display_images:
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components)
        cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)

    cnm2.estimates.select_components(use_object=True)
    cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
    cnm2.mmap_F = f_F_mmap
    cnm2.estimates.Cn = Cn
    cnm2.estimates.template = mc.total_template_rig
    cnm2.estimates.shifts = mc.shifts_rig
    save_name = cnm2.mmap_file[:-5] + '_caiman_init.hdf5'

    timing['end'] = time()
    print(timing)
    cnm2.save(save_name)
    print(save_name)
    output_file = save_name

    cm.stop_server(dview=dview)
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
    plt.close('all')
    return output_file

# def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
#     fio_state = {
#         'params': params,
#         'trace_fiola': trace_fiola,
#         'template': template,
#         'Ab': Ab,
#         'min_mov': min_mov,
#         'mov_data': mc_nn_mov
#     }
#     persistent_volume_path = "/persistent_storage"
#     timestamp = time().strftime("%Y%m%d%H%M%S")
#     filename = os.path.join(persistent_volume_path, f"{filepath}_{timestamp}.pkl")

#     with open(filename, 'wb') as f:
#         pickle.dump(fio_state, f)

#     logging.info(f"Saved state to {filename}")
def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
    fio_state = {
        'params': params,
        'trace_fiola': trace_fiola,
        'template': template,
        'Ab': Ab,
        'min_mov': min_mov,
        'mov_data': mc_nn_mov
    }
    persistent_volume_path = "/persistent_storage"
    timestamp = time().strftime("%Y%m%d%H%M%S")
    filename = os.path.join(persistent_volume_path, f"{filepath}_{timestamp}.pkl")

    with open(filename, 'wb') as f:
        pickle.dump(fio_state, f)

    # Save the latest filename to a known location
    latest_file_path = os.path.join(persistent_volume_path, 'latest_fiola_state.pkl')
    with open(latest_file_path, 'w') as f:
        f.write(filename)

    logging.info(f"Saved state to {filename}")
    logging.info(f"Updated latest state path to {latest_file_path}")


def caiman_process(fnames):
    mode = 'calcium'
    if mode == 'calcium':
        #folder = os.getenv('MOVIE_FOLDER', '/usr/src/app')
        # fnames = folder + '/test_sub.tif'
        fr = 30
        num_frames_init = 2000
        num_frames_total = 30000
        offline_batch = 100
        batch = 100
        flip = False
        detrend = False
        dc_param = 0.9995
        do_deconvolve = True
        ms = [3, 3]
        center_dims = None
        hals_movie = 'hp_thresh'
        n_split = 1
        nb = 1
        trace_with_neg = True
        lag = 5

        options = {
            'fnames': fnames,
            'fr': fr,
            'mode': mode,
            'num_frames_init': num_frames_init,
            'num_frames_total': num_frames_total,
            'offline_batch': offline_batch,
            'batch': batch,
            'flip': flip,
            'detrend': detrend,
            'dc_param': dc_param,
            'do_deconvolve': do_deconvolve,
            'ms': ms,
            'hals_movie': hals_movie,
            'center_dims': center_dims,
            'n_split': n_split,
            'nb': nb,
            'trace_with_neg': trace_with_neg,
            'lag': lag
        }

        mov = cm.load(fnames, subindices=range(num_frames_init))
        fnames_init = fnames.split('.')[0] + '_init.tif'
        mov.save(fnames_init)

        print(tf.test.gpu_device_name())

        caiman_file = run_caiman_init(fnames_init, pw_rigid=True, max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
        logging.info(caiman_file)
        cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
        estimates = cnm2.estimates
        template = cnm2.estimates.template
        Cn = cnm2.estimates.Cn
        logging.info('Finished run caiman init')
    else:
        raise Exception('mode must be either calcium')

    motion_correct = True
    do_nnls = True
    if motion_correct:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())
        plt.plot(shifts_fiola)
        plt.xlabel('frames')
        plt.ylabel('pixels')
        plt.legend(['x shifts', 'y shifts'])
    else:
        mc_nn_mov = mov
    logging.info('Finished mc')

    if do_nnls:
        params = fiolaparams(params_dict=options)
        fio = FIOLA(params=params)
        Ab = np.hstack((estimates.A.toarray(), estimates.b))
        trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch'])
        plt.plot(trace_fiola[:-nb].T)
        plt.xlabel('frames')
        plt.ylabel('fluorescence signal')

    else:
        if trace_with_neg:
            trace_fiola = np.vstack((estimates.C + estimates.YrA, estimates.f))
        else:
            trace_fiola = estimates.C + estimates.YrA
            trace_fiola[trace_fiola < 0] = 0
            trace_fiola = np.vstack((trace_fiola, estimates.f))
    logging.info('Finished nnls')

    params = fiolaparams(params_dict=options)
    fio = FIOLA(params=params)
    Ab = np.hstack((estimates.A.toarray(), estimates.b))
    Ab = Ab.astype(np.float32)
    fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())

    save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, fnames.split('.')[0] + 'pkl')
    print(f'The Total time for initialization is: {time() - startTime}')


# #!/usr/bin/env python
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
# import pickle

# from fiola.fiolaparams import fiolaparams
# from fiola.fiola import FIOLA
# from fiola.utilities import download_demo, load, to_2D, movie_iterator

# import tensorflow as tf
# tf.debugging.set_log_device_placement(True)

# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# else:
#     print("No GPU device found")

# logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
#                     level=logging.INFO)
# logging.info(device_lib.list_local_devices())  # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0

# import cv2
# import glob
# from time import time
# from caiman.motion_correction import MotionCorrect
# from caiman.source_extraction.cnmf import cnmf as cnmf
# from caiman.source_extraction.cnmf import params as params
# from caiman.utils.utils import download_demo
# from caiman.summary_images import local_correlations_movie_offline
# from caiman.source_extraction.cnmf.utilities import get_file_size

# # Start TensorFlow profiler
# logdir = "/usr/src/app/logs"
# tf.profiler.experimental.start(logdir)

# #%%    
# def run_caiman_init(fnames, pw_rigid=True, max_shifts=[6, 6], gnb=2, rf=15, K=5, gSig=[4, 4]):
#     c, dview, n_processes = cm.cluster.setup_cluster(
#         backend='local', n_processes=4, single_thread=False)  # Adjust n_processes as needed

#     timing = {}
#     timing['start'] = time()

#     # dataset dependent parameters
#     display_images = False

#     fr = 30  # imaging rate in frames per second
#     decay_time = 0.4  # length of a typical transient in seconds
#     dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
#     patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
#     strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
#     overlaps = (24, 24)
#     max_deviation_rigid = 3

#     mc_dict = {
#         'fnames': fnames,
#         'fr': fr,
#         'decay_time': decay_time,
#         'dxy': dxy,
#         'pw_rigid': pw_rigid,
#         'max_shifts': max_shifts,
#         'strides': strides,
#         'overlaps': overlaps,
#         'max_deviation_rigid': max_deviation_rigid,
#         'border_nan': 'copy',
#     }

#     opts = params.CNMFParams(params_dict=mc_dict)

#     # Motion correction and memory mapping
#     time_init = time()
#     mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
#     mc.motion_correct(save_movie=True)
#     border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
#     fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders
#     logging.info('Finished exclude borders')

#     Yr, dims, T = cm.load_memmap(fname_new)
#     images = np.reshape(Yr.T, [T] + list(dims), order='F')
#     logging.info('Finished reshaping borders')

#     # Restart cluster to clean up memory
#     cm.stop_server(dview=dview)
#     c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=4, single_thread=False)  # Adjust n_processes as needed
#     logging.info('Finished restarting cluster')

#     f_F_mmap = mc.mmap_file[0]
#     Cns = local_correlations_movie_offline(f_F_mmap, remove_baseline=True, window=1000, stride=1000, winSize_baseline=100, quantil_min_baseline=10, dview=dview)
#     logging.info('Finished local_correlations_movie_offline')
#     Cn = Cns.max(axis=0)
#     Cn[np.isnan(Cn)] = 0
#     plt.imshow(Cn, vmax=0.5)

#     # Parameters for source extraction and deconvolution
#     p = 1
#     merge_thr = 0.85
#     stride_cnmf = 6
#     method_init = 'greedy_roi'
#     ssub = 2
#     tsub = 2

#     opts_dict = {'fnames': fnames,
#                  'p': p,
#                  'fr': fr,
#                  'nb': gnb,
#                  'rf': rf,
#                  'K': K,
#                  'gSig': gSig,
#                  'stride': stride_cnmf,
#                  'method_init': method_init,
#                  'rolling_sum': True,
#                  'merge_thr': merge_thr,
#                  'n_processes': n_processes,
#                  'only_init': True,
#                  'ssub': ssub,
#                  'tsub': tsub}

#     opts.change_params(params_dict=opts_dict)
#     cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
#     cnm = cnm.fit(images)
#     logging.info('Finished CNFM on patches')

#     # Component evaluation
#     min_SNR = 1.0
#     rval_thr = 0.75
#     cnn_thr = 0.3
#     cnn_lowest = 0.0

#     cnm.params.set('quality', {'decay_time': decay_time,
#                                'min_SNR': min_SNR,
#                                'rval_thr': rval_thr,
#                                'use_cnn': False,
#                                'min_cnn_thr': cnn_thr,
#                                'cnn_lowest': cnn_lowest})
#     cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
#     print(len(cnm.estimates.idx_components))
#     time_patch = time()

#     if display_images:
#         cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

#     cnm.estimates.select_components(use_object=True)
#     cnm2 = cnm.refit(images, dview=dview)
#     time_end = time()
#     print(time_end - time_init)

#     min_SNR = 2
#     rval_thr = 0.85
#     cnn_thr = 0.15
#     cnn_lowest = 0.0

#     cnm2.params.set('quality', {'decay_time': decay_time,
#                                 'min_SNR': min_SNR,
#                                 'rval_thr': rval_thr,
#                                 'use_cnn': False,
#                                 'min_cnn_thr': cnn_thr,
#                                 'cnn_lowest': cnn_lowest})
#     cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
#     print(len(cnm2.estimates.idx_components))

#     cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

#     if display_images:
#         cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components)
#         cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)

#     cnm2.estimates.select_components(use_object=True)
#     cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
#     cnm2.mmap_F = f_F_mmap
#     cnm2.estimates.Cn = Cn
#     cnm2.estimates.template = mc.total_template_rig
#     cnm2.estimates.shifts = mc.shifts_rig
#     save_name = cnm2.mmap_file[:-5] + '_caiman_init.hdf5'

#     timing['end'] = time()
#     print(timing)
#     cnm2.save(save_name)
#     print(save_name)
#     output_file = save_name

#     cm.stop_server(dview=dview)
#     log_files = glob.glob('*_LOG_*')
#     for log_file in log_files:
#         os.remove(log_file)
#     plt.close('all')
#     return output_file

# def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
#     fio_state = {
#         'params': params,
#         'trace_fiola': trace_fiola,
#         'template': template,
#         'Ab': Ab,
#         'min_mov': min_mov,
#         'mov_data': mc_nn_mov
#     }
#     with open(filepath, 'wb') as f:
#         pickle.dump(fio_state, f)

# def main():
#     mode = 'calcium'
#     if mode == 'calcium':
#         folder = os.getenv('MOVIE_FOLDER', '/usr/src/app')
#         fnames = folder + '/test_sub.tif'
#         fr = 30
#         num_frames_init = 2000
#         num_frames_total = 30000
#         offline_batch = 5
#         batch = 1
#         flip = False
#         detrend = False
#         dc_param = 0.9995
#         do_deconvolve = True
#         ms = [3, 3]
#         center_dims = None
#         hals_movie = 'hp_thresh'
#         n_split = 1
#         nb = 1
#         trace_with_neg = True
#         lag = 5

#         options = {
#             'fnames': fnames,
#             'fr': fr,
#             'mode': mode,
#             'num_frames_init': num_frames_init,
#             'num_frames_total': num_frames_total,
#             'offline_batch': offline_batch,
#             'batch': batch,
#             'flip': flip,
#             'detrend': detrend,
#             'dc_param': dc_param,
#             'do_deconvolve': do_deconvolve,
#             'ms': ms,
#             'hals_movie': hals_movie,
#             'center_dims': center_dims,
#             'n_split': n_split,
#             'nb': nb,
#             'trace_with_neg': trace_with_neg,
#             'lag': lag
#         }

#         mov = cm.load(fnames, subindices=range(num_frames_init))
#         fnames_init = fnames.split('.')[0] + '_init.tif'
#         mov.save(fnames_init)

#         print(tf.test.gpu_device_name())

#         caiman_file = run_caiman_init(fnames_init, pw_rigid=True, max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
#         logging.info(caiman_file)
#         cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
#         estimates = cnm2.estimates
#         template = cnm2.estimates.template
#         Cn = cnm2.estimates.Cn
#         logging.info('Finished run caiman init')
#     else:
#         raise Exception('mode must be either calcium')

#     motion_correct = True
#     do_nnls = True
#     if motion_correct:
#         params = fiolaparams(params_dict=options)
#         fio = FIOLA(params=params)
#         mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())
#         plt.plot(shifts_fiola)
#         plt.xlabel('frames')
#         plt.ylabel('pixels')
#         plt.legend(['x shifts', 'y shifts'])
#     else:
#         mc_nn_mov = mov
#     logging.info('Finished mc')

#     if do_nnls:
#         params = fiolaparams(params_dict=options)
#         fio = FIOLA(params=params)
#         Ab = np.hstack((estimates.A.toarray(), estimates.b))
#         trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch'])
#         plt.plot(trace_fiola[:-nb].T)
#         plt.xlabel('frames')
#         plt.ylabel('fluorescence signal')

#     else:
#         if trace_with_neg:
#             trace_fiola = np.vstack((estimates.C + estimates.YrA, estimates.f))
#         else:
#             trace_fiola = estimates.C + estimates.YrA
#             trace_fiola[trace_fiola < 0] = 0
#             trace_fiola = np.vstack((trace_fiola, estimates.f))
#     logging.info('Finished nnls')

#     params = fiolaparams(params_dict=options)
#     fio = FIOLA(params=params)
#     Ab = np.hstack((estimates.A.toarray(), estimates.b))
#     Ab = Ab.astype(np.float32)
#     fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())

#     save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, 'fiola_state_msCam.pkl')

# if __name__ == "__main__":
#     main()

#     # Stop TensorFlow profiler
#     tf.profiler.experimental.stop()

# # #!/usr/bin/env python
# # import caiman as cm
# # import logging
# # import matplotlib.pyplot as plt
# # import numpy as np
# # import os
# # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
# # import pyximport
# # pyximport.install()
# # import scipy
# # from tensorflow.python.client import device_lib
# # from time import time
# # import pickle

# # from fiola.fiolaparams import fiolaparams
# # from fiola.fiola import FIOLA
# # from fiola.utilities import download_demo, load, to_2D, movie_iterator

# # import tensorflow as tf
# # tf.debugging.set_log_device_placement(True)

# # physical_devices = tf.config.list_physical_devices('GPU')
# # if physical_devices:
# #     tf.config.experimental.set_memory_growth(physical_devices[0], True)
# # else:
# #     print("No GPU device found")

# # logging.basicConfig(format="%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s][%(process)d] %(message)s",
# #                     level=logging.INFO)
# # logging.info(device_lib.list_local_devices())  # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0

# # import cv2
# # import glob
# # from time import time
# # from caiman.motion_correction import MotionCorrect
# # from caiman.source_extraction.cnmf import cnmf as cnmf
# # from caiman.source_extraction.cnmf import params as params
# # from caiman.utils.utils import download_demo
# # from caiman.summary_images import local_correlations_movie_offline
# # from caiman.source_extraction.cnmf.utilities import get_file_size

# # # Set up TensorFlow Profiler
# # options = tf.profiler.experimental.ProfilerOptions(
# #     host_tracer_level=2, python_tracer_level=1, device_tracer_level=1)
# # tf.profiler.experimental.start('logdir', options=options)

# # #%%    
# # def run_caiman_init(fnames, pw_rigid=True, max_shifts=[6, 6], gnb=2, rf=15, K=5, gSig=[4, 4]):
# #     c, dview, n_processes = cm.cluster.setup_cluster(
# #         backend='local', n_processes=4, single_thread=False)  # Adjust n_processes as needed

# #     timing = {}
# #     timing['start'] = time()

# #     # dataset dependent parameters
# #     display_images = False

# #     fr = 30  # imaging rate in frames per second
# #     decay_time = 0.4  # length of a typical transient in seconds
# #     dxy = (2., 2.)  # spatial resolution in x and y in (um per pixel)
# #     patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
# #     strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
# #     overlaps = (24, 24)
# #     max_deviation_rigid = 3

# #     mc_dict = {
# #         'fnames': fnames,
# #         'fr': fr,
# #         'decay_time': decay_time,
# #         'dxy': dxy,
# #         'pw_rigid': pw_rigid,
# #         'max_shifts': max_shifts,
# #         'strides': strides,
# #         'overlaps': overlaps,
# #         'max_deviation_rigid': max_deviation_rigid,
# #         'border_nan': 'copy',
# #     }

# #     opts = params.CNMFParams(params_dict=mc_dict)

# #     # Motion correction and memory mapping
# #     time_init = time()
# #     mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# #     mc.motion_correct(save_movie=True)
# #     border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
# #     fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C', border_to_0=border_to_0)  # exclude borders
# #     logging.info('Finished exclude borders')

# #     Yr, dims, T = cm.load_memmap(fname_new)
# #     images = np.reshape(Yr.T, [T] + list(dims), order='F')
# #     logging.info('Finished reshaping borders')

# #     # Restart cluster to clean up memory
# #     cm.stop_server(dview=dview)
# #     c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=4, single_thread=False)  # Adjust n_processes as needed
# #     logging.info('Finished restarting cluster')

# #     f_F_mmap = mc.mmap_file[0]
# #     Cns = local_correlations_movie_offline(f_F_mmap, remove_baseline=True, window=1000, stride=1000, winSize_baseline=100, quantil_min_baseline=10, dview=dview)
# #     logging.info('Finished local_correlations_movie_offline')
# #     Cn = Cns.max(axis=0)
# #     Cn[np.isnan(Cn)] = 0
# #     plt.imshow(Cn, vmax=0.5)

# #     # Parameters for source extraction and deconvolution
# #     p = 1
# #     merge_thr = 0.85
# #     stride_cnmf = 6
# #     method_init = 'greedy_roi'
# #     ssub = 2
# #     tsub = 2

# #     opts_dict = {'fnames': fnames,
# #                  'p': p,
# #                  'fr': fr,
# #                  'nb': gnb,
# #                  'rf': rf,
# #                  'K': K,
# #                  'gSig': gSig,
# #                  'stride': stride_cnmf,
# #                  'method_init': method_init,
# #                  'rolling_sum': True,
# #                  'merge_thr': merge_thr,
# #                  'n_processes': n_processes,
# #                  'only_init': True,
# #                  'ssub': ssub,
# #                  'tsub': tsub}

# #     opts.change_params(params_dict=opts_dict)
# #     cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
# #     cnm = cnm.fit(images)
# #     logging.info('Finished CNFM on patches')

# #     # Component evaluation
# #     min_SNR = 1.0
# #     rval_thr = 0.75
# #     cnn_thr = 0.3
# #     cnn_lowest = 0.0

# #     cnm.params.set('quality', {'decay_time': decay_time,
# #                                'min_SNR': min_SNR,
# #                                'rval_thr': rval_thr,
# #                                'use_cnn': False,
# #                                'min_cnn_thr': cnn_thr,
# #                                'cnn_lowest': cnn_lowest})
# #     cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
# #     print(len(cnm.estimates.idx_components))
# #     time_patch = time()

# #     if display_images:
# #         cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)

# #     cnm.estimates.select_components(use_object=True)
# #     cnm2 = cnm.refit(images, dview=dview)
# #     time_end = time()
# #     print(time_end - time_init)

# #     min_SNR = 2
# #     rval_thr = 0.85
# #     cnn_thr = 0.15
# #     cnn_lowest = 0.0

# #     cnm2.params.set('quality', {'decay_time': decay_time,
# #                                 'min_SNR': min_SNR,
# #                                 'rval_thr': rval_thr,
# #                                 'use_cnn': False,
# #                                 'min_cnn_thr': cnn_thr,
# #                                 'cnn_lowest': cnn_lowest})
# #     cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
# #     print(len(cnm2.estimates.idx_components))

# #     cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)

# #     if display_images:
# #         cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components)
# #         cnm2.estimates.view_components(images, img=Cn, idx=cnm2.estimates.idx_components_bad)

# #     cnm2.estimates.select_components(use_object=True)
# #     cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
# #     cnm2.mmap_F = f_F_mmap
# #     cnm2.estimates.Cn = Cn
# #     cnm2.estimates.template = mc.total_template_rig
# #     cnm2.estimates.shifts = mc.shifts_rig
# #     save_name = cnm2.mmap_file[:-5] + '_caiman_init.hdf5'

# #     timing['end'] = time()
# #     print(timing)
# #     cnm2.save(save_name)
# #     print(save_name)
# #     output_file = save_name

# #     cm.stop_server(dview=dview)
# #     log_files = glob.glob('*_LOG_*')
# #     for log_file in log_files:
# #         os.remove(log_file)
# #     plt.close('all')
# #     return output_file

# # def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
# #     fio_state = {
# #         'params': params,
# #         'trace_fiola': trace_fiola,
# #         'template': template,
# #         'Ab': Ab,
# #         'min_mov': min_mov,
# #         'mov_data': mc_nn_mov
# #     }
# #     with open(filepath, 'wb') as f:
# #         pickle.dump(fio_state, f)

# # def main():
# #     mode = 'calcium'
# #     if mode == 'calcium':
# #         folder = os.getenv('MOVIE_FOLDER', '/usr/src/app')
# #         fnames = folder + '/test_sub.tif'
# #         fr = 30
# #         num_frames_init = 2000
# #         num_frames_total = 30000
# #         offline_batch = 5
# #         batch = 1
# #         flip = False
# #         detrend = False
# #         dc_param = 0.9995
# #         do_deconvolve = True
# #         ms = [3, 3]
# #         center_dims = None
# #         hals_movie = 'hp_thresh'
# #         n_split = 1
# #         nb = 1
# #         trace_with_neg = True
# #         lag = 5

# #         options = {
# #             'fnames': fnames,
# #             'fr': fr,
# #             'mode': mode,
# #             'num_frames_init': num_frames_init,
# #             'num_frames_total': num_frames_total,
# #             'offline_batch': offline_batch,
# #             'batch': batch,
# #             'flip': flip,
# #             'detrend': detrend,
# #             'dc_param': dc_param,
# #             'do_deconvolve': do_deconvolve,
# #             'ms': ms,
# #             'hals_movie': hals_movie,
# #             'center_dims': center_dims,
# #             'n_split': n_split,
# #             'nb': nb,
# #             'trace_with_neg': trace_with_neg,
# #             'lag': lag
# #         }

# #         mov = cm.load(fnames, subindices=range(num_frames_init))
# #         fnames_init = fnames.split('.')[0] + '_init.tif'
# #         mov.save(fnames_init)

# #         print(tf.test.gpu_device_name())

# #         caiman_file = run_caiman_init(fnames_init, pw_rigid=True, max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
# #         logging.info(caiman_file)
# #         cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
# #         estimates = cnm2.estimates
# #         template = cnm2.estimates.template
# #         Cn = cnm2.estimates.Cn
# #         logging.info('Finished run caiman init')
# #     else:
# #         raise Exception('mode must be either calcium')

# #     motion_correct = True
# #     do_nnls = True
# #     if motion_correct:
# #         params = fiolaparams(params_dict=options)
# #         fio = FIOLA(params=params)
# #         mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())
# #         plt.plot(shifts_fiola)
# #         plt.xlabel('frames')
# #         plt.ylabel('pixels')
# #         plt.legend(['x shifts', 'y shifts'])
# #     else:
# #         mc_nn_mov = mov
# #     logging.info('Finished mc')

# #     if do_nnls:
# #         params = fiolaparams(params_dict=options)
# #         fio = FIOLA(params=params)
# #         Ab = np.hstack((estimates.A.toarray(), estimates.b))
# #         trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch'])
# #         plt.plot(trace_fiola[:-nb].T)
# #         plt.xlabel('frames')
# #         plt.ylabel('fluorescence signal')

# #     else:
# #         if trace_with_neg:
# #             trace_fiola = np.vstack((estimates.C + estimates.YrA, estimates.f))
# #         else:
# #             trace_fiola = estimates.C + estimates.YrA
# #             trace_fiola[trace_fiola < 0] = 0
# #             trace_fiola = np.vstack((trace_fiola, estimates.f))
# #     logging.info('Finished nnls')

# #     params = fiolaparams(params_dict=options)
# #     fio = FIOLA(params=params)
# #     Ab = np.hstack((estimates.A.toarray(), estimates.b))
# #     Ab = Ab.astype(np.float32)
# #     fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())

# #     save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, 'fiola_state_msCam.pkl')

# # if __name__ == "__main__":
# #     main()

# # # Stop TensorFlow Profiler
# # tf.profiler.experimental.stop()

# # # #!/usr/bin/env python
# # # import caiman as cm
# # # import logging
# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import os
# # # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # # os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
# # # import pyximport
# # # pyximport.install()
# # # import scipy
# # # from tensorflow.python.client import device_lib
# # # from time import time

# # # import pickle

# # # from fiola.fiolaparams import fiolaparams
# # # from fiola.fiola import FIOLA
# # # from fiola.utilities import download_demo, load, to_2D, movie_iterator

# # # import tensorflow as tf
# # # tf.debugging.set_log_device_placement(True)

# # # logging.basicConfig(format=
# # #                     "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
# # #                     "[%(process)d] %(message)s",
# # #                     level=logging.INFO)    
# # # logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0


# # # import cv2
# # # import glob

# # # from time import time

# # # from caiman.motion_correction import MotionCorrect
# # # from caiman.source_extraction.cnmf import cnmf as cnmf
# # # from caiman.source_extraction.cnmf import params as params
# # # from caiman.utils.utils import download_demo
# # # from caiman.summary_images import local_correlations_movie_offline
# # # from caiman.source_extraction.cnmf.utilities import get_file_size


# # # #%%    
# # # def run_caiman_init(fnames, pw_rigid = True, max_shifts=[6, 6], gnb=2, rf=15, K = 5, gSig = [4, 4]):
# # #     """
# # #     Run caiman for initialization.
    
# # #     Parameters
# # #     ----------
# # #     fnames : string
# # #         file name
# # #     pw_rigid : bool, 
# # #         flag to select rigid vs pw_rigid motion correction. The default is True.
# # #     max_shifts: list
# # #         maximum shifts allowed for x axis and y axis. The default is [6, 6].
# # #     gnb : int
# # #         number of background components. The default is 2.
# # #     rf: int
# # #         half-size of the patches in pixels. e.g., if rf=25, patches are 50x50. The default value is 15.
# # #     K : int
# # #         number of components per patch. The default is 5.
# # #     gSig : list
# # #         expected half size of neurons in pixels. The default is [4, 4].

# # #     Returns
# # #     -------
# # #     output_file : string
# # #         file with caiman output

# # #     """
# # #     c, dview, n_processes = cm.cluster.setup_cluster(
# # #             backend='local', n_processes=None, single_thread=False)
    
# # #     timing = {}
# # #     timing['start'] = time()

# # #     # dataset dependent parameters
# # #     display_images = False
    
# # #     fr = 30             # imaging rate in frames per second
# # #     decay_time = 0.4    # length of a typical transient in seconds
# # #     dxy = (2., 2.)      # spatial resolution in x and y in (um per pixel)
# # #     # note the lower than usual spatial resolution here
# # #     patch_motion_um = (100., 100.)  # patch size for non-rigid correction in um
# # #     # motion correction parameters
# # #     # start a new patch for pw-rigid motion correction every x pixels
# # #     strides = tuple([int(a/b) for a, b in zip(patch_motion_um, dxy)])
# # #     # overlap between pathes (size of patch in pixels: strides+overlaps)
# # #     overlaps = (24, 24)
# # #     # maximum deviation allowed for patch with respect to rigid shifts
# # #     max_deviation_rigid = 3
    
# # #     mc_dict = {
# # #         'fnames': fnames,
# # #         'fr': fr,
# # #         'decay_time': decay_time,
# # #         'dxy': dxy,
# # #         'pw_rigid': pw_rigid,
# # #         'max_shifts': max_shifts,
# # #         'strides': strides,
# # #         'overlaps': overlaps,
# # #         'max_deviation_rigid': max_deviation_rigid,
# # #         'border_nan': 'copy',
# # #     }
    
# # #     opts = params.CNMFParams(params_dict=mc_dict)
    
# # #     # Motion correction and memory mapping
# # #     time_init = time()
# # #     mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# # #     mc.motion_correct(save_movie=True)
# # #     border_to_0 = 0 if mc.border_nan == 'copy' else mc.border_to_0
# # #     fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
# # #                                border_to_0=border_to_0)  # exclude borders
# # #     logging.info('Finished exclude borders')
# # #     #
# # #     Yr, dims, T = cm.load_memmap(fname_new)
# # #     images = np.reshape(Yr.T, [T] + list(dims), order='F')
# # #     logging.info('Finished reshapping borders')


# # #     #  restart cluster to clean up memory
# # #     cm.stop_server(dview=dview)
# # #     c, dview, n_processes = cm.cluster.setup_cluster(
# # #         backend='local', n_processes=None, single_thread=False)
# # #     logging.info('Finished restarting cluster')

# # #     #
# # #     f_F_mmap = mc.mmap_file[0]
# # #     Cns = local_correlations_movie_offline(f_F_mmap,
# # #                                        remove_baseline=True, window=1000, stride=1000,
# # #                                        winSize_baseline=100, quantil_min_baseline=10,
# # #                                        dview=dview)
# # #     logging.info('Finished local_correlations_movie_offline')
# # #     Cn = Cns.max(axis=0)
# # #     Cn[np.isnan(Cn)] = 0
# # #     # if display_images: 
       
# # #     plt.imshow(Cn,vmax=0.5)
# # #     #   parameters for source extraction and deconvolution
# # #     p = 1                    # order of the autoregressive system
# # #     merge_thr = 0.85         # merging threshold, max correlation allowed
# # #     stride_cnmf = 6          # amount of overlap between the patches in pixels
# # #     # initialization method (if analyzing dendritic data using 'sparse_nmf')
# # #     method_init = 'greedy_roi'
# # #     ssub = 2                     # spatial subsampling during initialization
# # #     tsub = 2                     # temporal subsampling during intialization
    
# # #     # parameters for component evaluation
# # #     opts_dict = {'fnames': fnames,
# # #                  'p': p,
# # #                  'fr': fr,
# # #                  'nb': gnb,
# # #                  'rf': rf,
# # #                  'K': K,
# # #                  'gSig': gSig,
# # #                  'stride': stride_cnmf,
# # #                  'method_init': method_init,
# # #                  'rolling_sum': True,
# # #                  'merge_thr': merge_thr,
# # #                  'n_processes': n_processes,
# # #                  'only_init': True,
# # #                  'ssub': ssub,
# # #                  'tsub': tsub}
    
# # #     opts.change_params(params_dict=opts_dict)
# # #     #  RUN CNMF ON PATCHES
# # #     cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
# # #     cnm = cnm.fit(images)
# # #     logging.info('Finished CNFM on patches')

# # #     #  COMPONENT EVALUATION
# # #     min_SNR = 1.0  # signal to noise ratio for accepting a component
# # #     rval_thr = 0.75  # space correlation threshold for accepting a component
# # #     cnn_thr = 0.3  # threshold for CNN based classifier
# # #     cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected
    
# # #     cnm.params.set('quality', {'decay_time': decay_time,
# # #                            'min_SNR': min_SNR,
# # #                            'rval_thr': rval_thr,
# # #                            'use_cnn': False,
# # #                            'min_cnn_thr': cnn_thr,
# # #                            'cnn_lowest': cnn_lowest})
# # #     cnm.estimates.evaluate_components(images, cnm.params, dview=dview)
# # #     print(len(cnm.estimates.idx_components))
# # #     time_patch = time()
# # #     #
# # #     if display_images:
# # #         cnm.estimates.plot_contours(img=Cn, idx=cnm.estimates.idx_components)
# # #     #
# # #     cnm.estimates.select_components(use_object=True)
# # #     # RE-RUN seeded CNMF on accepted patches to refine and perform deconvolution
# # #     cnm2 = cnm.refit(images, dview=dview)
# # #     time_end = time() 
# # #     print(time_end- time_init)
# # #     #  COMPONENT EVALUATION
# # #     min_SNR = 2  # signal to noise ratio for accepting a component
# # #     rval_thr = 0.85  # space correlation threshold for accepting a component
# # #     cnn_thr = 0.15  # threshold for CNN based classifier
# # #     cnn_lowest = 0.0 # neurons with cnn probability lower than this value are rejected
    
# # #     cnm2.params.set('quality', {'decay_time': decay_time,
# # #                                'min_SNR': min_SNR,
# # #                                'rval_thr': rval_thr,
# # #                                'use_cnn': False,
# # #                                'min_cnn_thr': cnn_thr,
# # #                                'cnn_lowest': cnn_lowest})
# # #     cnm2.estimates.evaluate_components(images, cnm2.params, dview=dview)
# # #     print(len(cnm2.estimates.idx_components))
    
# # #     #  PLOT COMPONENTS
# # #     cnm2.estimates.plot_contours(img=Cn, idx=cnm2.estimates.idx_components)
# # #     #  VIEW TRACES (accepted and rejected)
# # #     if display_images:
# # #         cnm2.estimates.view_components(images, img=Cn,
# # #                                       idx=cnm2.estimates.idx_components)
# # #         cnm2.estimates.view_components(images, img=Cn,
# # #                                       idx=cnm2.estimates.idx_components_bad)
# # #     # update object with selected components
# # #     cnm2.estimates.select_components(use_object=True)
# # #     # Extract DF/F values
# # #     cnm2.estimates.detrend_df_f(quantileMin=8, frames_window=250)
# # #     # Show final traces
# # #     #cnm2.estimates.view_components(img=Cn)
# # #     #
# # #     cnm2.mmap_F = f_F_mmap 
# # #     cnm2.estimates.Cn = Cn
# # #     cnm2.estimates.template = mc.total_template_rig
# # #     cnm2.estimates.shifts = mc.shifts_rig
# # #     save_name = cnm2.mmap_file[:-5] + '_caiman_init.hdf5'
    
# # #     timing['end'] = time()
# # #     print(timing)
# # #     cnm2.save(save_name)
# # #     print(save_name)
# # #     output_file = save_name
# # #     # STOP CLUSTER and clean up log files
# # #     cm.stop_server(dview=dview)
# # #     log_files = glob.glob('*_LOG_*')
# # #     for log_file in log_files:
# # #         os.remove(log_file)
# # #     plt.close('all')        
# # #     return output_file


# # # #%% 

# # # def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
# # #     fio_state = {
# # #         'params': params,
# # #         'trace_fiola': trace_fiola,
# # #         'template': template,
# # #         'Ab': Ab,
# # #         'min_mov': min_mov,
        
# # #         'mov_data': mc_nn_mov  # Convert the movie data to a list to make it serializable
# # #     }
# # #     with open(filepath, 'wb') as f:
# # #         pickle.dump(fio_state, f)

# # # #%% 
# # # def main():
# # # #%%
# # #     mode = 'calcium'                    # 'voltage' or 'calcium' fluorescence indicator
# # #     # Parameter setting
    
# # #     if mode == 'calcium':
# # #         # folder = cm.paths.caiman_datadir() + '/example_movies'
# # # #        folder = 'C:/Users/29712/fiola/CaImAn/example_movies'
# # #         folder = os.getenv('MOVIE_FOLDER', '/usr/src/app')
# # #         #fnames = folder + '/output_multi_frame99.tif'
# # #         #fnames = folder + '/output_multi_frame_big_endian.tif'
# # #         # fnames = folder + '/result_update270.tif'
# # #         fnames = folder + '/test_sub.tif'
# # #         fr = 30                         # sample rate of the movie
        
# # #         mode = 'calcium'                # 'voltage' or 'calcium' fluorescence indicator
# # #         num_frames_init =   2000       # number of frames used for initialization
# # #         num_frames_total =  30000        # estimated total number of frames for processing, this is used for generating matrix to store data
# # #         offline_batch = 5               # number of frames for one batch to perform offline motion correction
# # #         batch= 1                        # number of frames processing at the same time using gpu 
# # #         flip = False                    # whether to flip signal to find spikes   
# # #         detrend = False                 # whether to remove the slow trend in the fluorescence data
# # #         dc_param = 0.9995               # DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
# # #                                         # 0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
# # #         do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
# # #         # ms = [55,55]    
# # #         ms = [3,3]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
# # #         center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
# # #         hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
# # #                                         # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
# # #         n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
# # #         nb = 1                          # number of background components
# # #         trace_with_neg=True             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
# # #         lag = 5                         # lag for retrieving the online result.
                        
# # #         options = {
# # #             'fnames': fnames,
# # #             'fr': fr,
# # #             'mode': mode, 
# # #             'num_frames_init': num_frames_init,     
# # #             'num_frames_total':num_frames_total,
# # #             'offline_batch': offline_batch,
# # #             'batch':batch,
# # #             'flip': flip,
# # #             'detrend': detrend,
# # #             'dc_param': dc_param,            
# # #             'do_deconvolve': do_deconvolve,
# # #             'ms': ms,
# # #             'hals_movie': hals_movie,
# # #             'center_dims':center_dims, 
# # #             'n_split': n_split,
# # #             'nb' : nb, 
# # #             'trace_with_neg':trace_with_neg, 
# # #             'lag': lag}
        
# # #         mov = cm.load(fnames, subindices=range(num_frames_init))
# # #         fnames_init = fnames.split('.')[0] + '_init.tif'
# # #         mov.save(fnames_init)
        
# # #         print(tf.test.gpu_device_name())
        
# # #         # run caiman initialization. User might need to change the parameters 
# # #         # inside the file to get good initialization result
# # #         caiman_file = run_caiman_init(fnames_init, pw_rigid=True, 
# # #                                       max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
# # #         logging.info(caiman_file)
# # #         # load results of initialization
# # #         cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
# # #         estimates = cnm2.estimates
# # #         template = cnm2.estimates.template
# # #         Cn = cnm2.estimates.Cn
# # #         logging.info('Finished run caiman init')   
# # #     else: 
# # #         raise Exception('mode must be either calcium')
          
# # #     #%% Run FIOLA
# # #     #example motion correction
# # #     motion_correct = True
# # #     #example source separation
# # #     do_nnls = True
# # #     #%% Mot corr only
# # #     if motion_correct:
# # #         params = fiolaparams(params_dict=options)
# # #         fio = FIOLA(params=params)
# # #         # run motion correction on GPU on the initialization movie
# # #         mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())             
# # #         plt.plot(shifts_fiola)
# # #         plt.xlabel('frames')
# # #         plt.ylabel('pixels')                 
# # #         plt.legend(['x shifts', 'y shifts'])
# # #     else:    
# # #         mc_nn_mov = mov
# # #     logging.info('Finished mc')       
# # #     #%% NNLS only
# # #     if do_nnls:
# # #         params = fiolaparams(params_dict=options)
# # #         fio = FIOLA(params=params)
# # #         Ab = np.hstack((estimates.A.toarray(), estimates.b))
# # #         trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch']) 
# # #         plt.plot(trace_fiola[:-nb].T)
# # #         plt.xlabel('frames')
# # #         plt.ylabel('fluorescence signal')              

# # #     else:        
# # #         if trace_with_neg == True:
# # #             trace_fiola = np.vstack((estimates.C+estimates.YrA, estimates.f))
# # #         else:
# # #             trace_fiola = estimates.C+estimates.YrA
# # #             trace_fiola[trace_fiola < 0] = 0
# # #             trace_fiola = np.vstack((trace_fiola, estimates.f))
# # #     logging.info('Finished nnls')   
# # #     #%% set up online pipeline
# # #     params = fiolaparams(params_dict=options)
# # #     fio = FIOLA(params=params)
# # #     Ab = np.hstack((estimates.A.toarray(), estimates.b))
# # #     Ab = Ab.astype(np.float32)        
# # #     fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())

# # #    # After creating the FIOLA pipeline

# # # # Save the FIOLA state
# # #     save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, 'fiola_state_msCam.pkl')


# # # if __name__ == "__main__":
# # #     main()



# # # # # #!/usr/bin/env python
# # # # # """
# # # # # Illustration of the usage of FIOLA with calcium and voltage imaging data. 
# # # # # For Calcium USE THE demo_initialize_calcium.py FILE TO GENERATE THE HDF5 files necessary for 
# # # # # initialize FIOLA. 
# # # # # For voltage this demo is self contained.   
# # # # # copyright in license file
# # # # # authors: @agiovann @changjia
# # # # # """
# # # # # #%%
# # # # # import caiman as cm
# # # # # import logging
# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np
# # # # # import os
# # # # # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # # # # os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
# # # # # import pyximport
# # # # # pyximport.install()
# # # # # import scipy
# # # # # from tensorflow.python.client import device_lib
# # # # # from time import time
    
# # # # # from fiola.demo_initialize_calcium import run_caiman_init
# # # # # from fiola.fiolaparams import fiolaparams
# # # # # from fiola.fiola import FIOLA
# # # # # from fiola.utilities import download_demo, load, to_2D, movie_iterator

# # # # # logging.basicConfig(format=
# # # # #                     "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
# # # # #                     "[%(process)d] %(message)s",
# # # # #                     level=logging.INFO)    
# # # # # logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0
# # # # # #%% 
# # # # # def main():
# # # # # #%%
# # # # #     mode = 'calcium'                    # 'voltage' or 'calcium' fluorescence indicator
# # # # #     # Parameter setting
    
# # # # #     if mode == 'calcium':
# # # # #         # folder = cm.paths.caiman_datadir() + '/example_movies'
# # # # #         folder = 'C:/Users/29712/fiola/CaImAn/example_movies'
# # # # #         #fnames = folder + '/output_multi_frame99.tif'
# # # # #         #fnames = folder + '/output_multi_frame_big_endian.tif'
# # # # #         # fnames = folder + '/result_update270.tif'
# # # # #         fnames = folder + '/msCam_continuous.tif'
# # # # #         fr = 30                         # sample rate of the movie
        
# # # # #         mode = 'calcium'                # 'voltage' or 'calcium' fluorescence indicator
# # # # #         num_frames_init =   1000       # number of frames used for initialization
# # # # #                                          # estimated total number of frames for processing, this is used for generating matrix to store data
# # # # #         offline_batch = 5               # number of frames for one batch to perform offline motion correction
# # # # #         batch= 1                        # number of frames processing at the same time using gpu 
# # # # #         flip = False                    # whether to flip signal to find spikes   
# # # # #         detrend = False                 # whether to remove the slow trend in the fluorescence data
# # # # #         dc_param = 0.9995               # DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
# # # # #                                         # 0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
# # # # #         do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
# # # # #         ms = [55, 55]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
# # # # #         center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
# # # # #         hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
# # # # #                                         # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
# # # # #         n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
# # # # #         nb = 1                          # number of background components
# # # # #         trace_with_neg=True             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
# # # # #         lag = 5                         # lag for retrieving the online result.
                        
     
        
# # # # #         mov = cm.load(fnames, subindices=range(num_frames_init))
# # # # #         fnames_init = fnames.split('.')[0] + '_init.tif'
# # # # #         mov.save(fnames_init)
# # # # #         print('here')
# # # # #         # run caiman initialization. User might need to change the parameters 
# # # # #         # inside the file to get good initialization result
# # # # #         caiman_file = run_caiman_init(fnames_init, pw_rigid=True, 
# # # # #                                       max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
# # # # #         print(f'Caiman intialization file is ready: ' + caiman_file)
# # # # #         print()
        
# # # # # if __name__ == "__main__":
# # # # #     main()

# # # # #!/usr/bin/env python
# # # # """
# # # # Illustration of the usage of FIOLA with calcium and voltage imaging data. 
# # # # For Calcium USE THE demo_initialize_calcium.py FILE TO GENERATE THE HDF5 files necessary for 
# # # # initialize FIOLA. 
# # # # For voltage this demo is self contained.   
# # # # copyright in license file
# # # # authors: @agiovann @changjia
# # # # """
# # # # #%%
# # # # import caiman as cm
# # # # import logging
# # # # import matplotlib.pyplot as plt
# # # # import numpy as np
# # # # import os
# # # # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# # # # os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
# # # # import pyximport
# # # # pyximport.install()
# # # # import scipy
# # # # from tensorflow.python.client import device_lib
# # # # from time import time

# # # # import pickle

# # # # from fiola.demo_initialize_calcium import run_caiman_init
# # # # from fiola.fiolaparams import fiolaparams
# # # # from fiola.fiola import FIOLA
# # # # from fiola.utilities import download_demo, load, to_2D, movie_iterator

# # # # import tensorflow as tf
# # # # print(tf.test.gpu_device_name())
# # # # tf.debugging.set_log_device_placement(True)

# # # # logging.basicConfig(format=
# # # #                     "%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s]"\
# # # #                     "[%(process)d] %(message)s",
# # # #                     level=logging.INFO)    
# # # # logging.info(device_lib.list_local_devices()) # if GPU is not detected, try to reinstall tensorflow with pip install tensorflow==2.5.0
# # # # #%% 

# # # # def save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, min_mov, params, filepath):
# # # #     fio_state = {
# # # #         'params': params,
# # # #         'trace_fiola': trace_fiola,
# # # #         'template': template,
# # # #         'Ab': Ab,
# # # #         'min_mov': min_mov,
        
# # # #         'mov_data': mc_nn_mov  # Convert the movie data to a list to make it serializable
# # # #     }
# # # #     with open(filepath, 'wb') as f:
# # # #         pickle.dump(fio_state, f)



# # # # #%% 
# # # # def main():
# # # # #%%
# # # #     mode = 'calcium'                    # 'voltage' or 'calcium' fluorescence indicator
# # # #     # Parameter setting
    
# # # #     if mode == 'calcium':
# # # #         # folder = cm.paths.caiman_datadir() + '/example_movies'
# # # # #        folder = 'C:/Users/29712/fiola/CaImAn/example_movies'
# # # #         folder = os.getenv('MOVIE_FOLDER', '/usr/src/app')
# # # #         #fnames = folder + '/output_multi_frame99.tif'
# # # #         #fnames = folder + '/output_multi_frame_big_endian.tif'
# # # #         # fnames = folder + '/result_update270.tif'
# # # #         fnames = folder + '/test_sub.tif'
# # # #         fr = 30                         # sample rate of the movie
        
# # # #         mode = 'calcium'                # 'voltage' or 'calcium' fluorescence indicator
# # # #         num_frames_init =   2000       # number of frames used for initialization
# # # #         num_frames_total =  30000        # estimated total number of frames for processing, this is used for generating matrix to store data
# # # #         offline_batch = 5               # number of frames for one batch to perform offline motion correction
# # # #         batch= 1                        # number of frames processing at the same time using gpu 
# # # #         flip = False                    # whether to flip signal to find spikes   
# # # #         detrend = False                 # whether to remove the slow trend in the fluorescence data
# # # #         dc_param = 0.9995               # DC blocker parameter for removing the slow trend in the fluorescence data. It is usually between
# # # #                                         # 0.99 and 1. Higher value will remove less trend. No detrending will perform if detrend=False.
# # # #         do_deconvolve = True            # If True, perform spike detection for voltage imaging or deconvolution for calcium imaging.
# # # #         # ms = [55,55]    
# # # #         ms = [3,3]                     # maximum shift in x and y axis respectively. Will not perform motion correction if None.
# # # #         center_dims = None              # template dimensions for motion correction. If None, the input will the the shape of the FOV
# # # #         hals_movie = 'hp_thresh'        # apply hals on the movie high-pass filtered and thresholded with 0 (hp_thresh); movie only high-pass filtered (hp); 
# # # #                                         # original movie (orig); no HALS needed if the input is from CaImAn (when init_method is 'caiman' or 'weighted_masks')
# # # #         n_split = 1                     # split neuron spatial footprints into n_split portion before performing matrix multiplication, increase the number when spatial masks are larger than 2GB
# # # #         nb = 1                          # number of background components
# # # #         trace_with_neg=True             # return trace with negative components (noise) if True; otherwise the trace is cutoff at 0
# # # #         lag = 5                         # lag for retrieving the online result.
                        
# # # #         options = {
# # # #             'fnames': fnames,
# # # #             'fr': fr,
# # # #             'mode': mode, 
# # # #             'num_frames_init': num_frames_init,     
# # # #             'num_frames_total':num_frames_total,
# # # #             'offline_batch': offline_batch,
# # # #             'batch':batch,
# # # #             'flip': flip,
# # # #             'detrend': detrend,
# # # #             'dc_param': dc_param,            
# # # #             'do_deconvolve': do_deconvolve,
# # # #             'ms': ms,
# # # #             'hals_movie': hals_movie,
# # # #             'center_dims':center_dims, 
# # # #             'n_split': n_split,
# # # #             'nb' : nb, 
# # # #             'trace_with_neg':trace_with_neg, 
# # # #             'lag': lag}
        
# # # #         mov = cm.load(fnames, subindices=range(num_frames_init))
# # # #         fnames_init = fnames.split('.')[0] + '_init.tif'
# # # #         mov.save(fnames_init)
        
# # # #         print(tf.test.gpu_device_name())

# # # #         # run caiman initialization. User might need to change the parameters 
# # # #         # inside the file to get good initialization result
# # # #         caiman_file = run_caiman_init(fnames_init, pw_rigid=True, 
# # # #                                       max_shifts=ms, gnb=nb, rf=15, K=4, gSig=[3, 3])
# # # #         logging.info(caiman_file)
# # # #         # load results of initialization
# # # #         cnm2 = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_file)
# # # #         estimates = cnm2.estimates
# # # #         template = cnm2.estimates.template
# # # #         Cn = cnm2.estimates.Cn
# # # #         logging.info('Finished run caiman init')   
# # # #     else: 
# # # #         raise Exception('mode must be either calcium')
          
# # # #     #%% Run FIOLA
# # # #     #example motion correction
# # # #     motion_correct = True
# # # #     #example source separation
# # # #     do_nnls = True
# # # #     #%% Mot corr only
# # # #     if motion_correct:
# # # #         params = fiolaparams(params_dict=options)
# # # #         fio = FIOLA(params=params)
# # # #         # run motion correction on GPU on the initialization movie
# # # #         mc_nn_mov, shifts_fiola, _ = fio.fit_gpu_motion_correction(mov, template, fio.params.mc_nnls['offline_batch'], min_mov=mov.min())             
# # # #         plt.plot(shifts_fiola)
# # # #         plt.xlabel('frames')
# # # #         plt.ylabel('pixels')                 
# # # #         plt.legend(['x shifts', 'y shifts'])
# # # #     else:    
# # # #         mc_nn_mov = mov
# # # #     logging.info('Finished mc')       
# # # #     #%% NNLS only
# # # #     if do_nnls:
# # # #         params = fiolaparams(params_dict=options)
# # # #         fio = FIOLA(params=params)
# # # #         Ab = np.hstack((estimates.A.toarray(), estimates.b))
# # # #         trace_fiola, _ = fio.fit_gpu_nnls(mc_nn_mov, Ab, batch_size=fio.params.mc_nnls['offline_batch']) 
# # # #         plt.plot(trace_fiola[:-nb].T)
# # # #         plt.xlabel('frames')
# # # #         plt.ylabel('fluorescence signal')              

# # # #     else:        
# # # #         if trace_with_neg == True:
# # # #             trace_fiola = np.vstack((estimates.C+estimates.YrA, estimates.f))
# # # #         else:
# # # #             trace_fiola = estimates.C+estimates.YrA
# # # #             trace_fiola[trace_fiola < 0] = 0
# # # #             trace_fiola = np.vstack((trace_fiola, estimates.f))
# # # #     logging.info('Finished nnls')   
# # # #     #%% set up online pipeline
# # # #     params = fiolaparams(params_dict=options)
# # # #     fio = FIOLA(params=params)
# # # #     Ab = np.hstack((estimates.A.toarray(), estimates.b))
# # # #     Ab = Ab.astype(np.float32)        
# # # #     fio = fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=mov.min())

# # # #    # After creating the FIOLA pipeline

# # # # # Save the FIOLA state
# # # #     save_fiola_state(mc_nn_mov, trace_fiola, template, Ab, mov.min(), params, 'fiola_state_msCam.pkl')


# # # # if __name__ == "__main__":
# # # #     main()