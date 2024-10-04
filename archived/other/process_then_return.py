
import sys
import os
import numpy as np
import pickle
import logging
from time import time
from fiola.fiola import FIOLA
import warnings
import tifffile
import io
import queue
warnings.filterwarnings("ignore", message="no queue or thread to delete")

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

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
    fio.pipeline.saoz.update_q = queue.Queue()
    
    return fio
def memmap_from_buffer(tiff_buffer):
    buffer = io.BytesIO(tiff_buffer)
    try:
        with tifffile.TiffFile(buffer) as tif:
            tiff_series = tif.series[0]
            dtype = tiff_series.dtype
            shape = tiff_series.shape
            byte_order = tif.byteorder

            if len(shape) == 2:  # If only height and width are provided, assume single frame
                shape = (1, *shape)
            logging.info(f"Shape: {shape}")

            # Initialize image_data to hold the entire TIFF data
            image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))

            for page_index, page in enumerate(tif.pages):
                offsets = page.dataoffsets
                bytecounts = page.databytecounts
                for offset, bytecount in zip(offsets, bytecounts):
                    buffer.seek(offset)
                    data = np.frombuffer(buffer.read(bytecount), dtype=np.dtype(byte_order + dtype.char))
                    image_data[page_index, ...] = data.reshape(shape[1:])

        if image_data.size != np.prod(shape):
            logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
            raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

        logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
        return image_data
    except Exception as e:
        logging.error(f"Error processing TIFF buffer: {e}")
        raise

def process_frame(fio, memmap_image, frame_idx, batch):
    try:
        start_time = time()
        num_frames_total = memmap_image.shape[0]
        
        online_trace = np.zeros((fio.Ab.shape[-1], num_frames_total), dtype=np.float32)
        online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], num_frames_total), dtype=np.float32)

        for idx in range(0, num_frames_total, batch):
            frame_batch = memmap_image[idx:idx + batch].astype(np.float32)
            fio.fit_online_frame(frame_batch)
            online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx:idx + batch]
            online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx:idx + batch]

        fio.compute_estimates()
        fio.pipeline.saoz.online_trace = online_trace
        fio.pipeline.saoz.online_trace_deconvolved = online_trace_deconvolved

        # serialized_estimates = pickle.dumps(fio.estimates)
        # sys.stdout.buffer.write(b'--PICKLE-START--')
        # sys.stdout.buffer.write(serialized_estimates)
        # sys.stdout.buffer.write(b'--PICKLE-END--')
        # sys.stdout.flush()
        np.save(f'./results/fiola_result_ptr_{frame_idx}', fio.estimates)
        end_time = time()
        total_time = end_time - start_time
        logging.info(f"Total time spent processing frame {frame_idx}: {total_time}")

    except Exception as e:
        logging.error(f"Error processing frame {frame_idx}: {e}")

def process_frame_with_buffer(fio, frame_data, frame_idx, batch):
    try:
        memmap_image = memmap_from_buffer(frame_data)
        process_frame(fio, memmap_image, frame_idx, batch)
    except Exception as e:
        logging.error(f"Failed to process frame with buffer: {e}")

def main():
    fio = load_fiola_state('fiola_state_msCam.pkl')
    frame_idx = 0
    batch = 1  # Increase batch size

    while True:
        try:
            frame_size_data = sys.stdin.buffer.read(4)
            if not frame_size_data:
                break
            frame_size = int.from_bytes(frame_size_data, byteorder='little')
            
            frame_data = sys.stdin.buffer.read(frame_size)
            if not frame_data or len(frame_data) != frame_size:
                break
            
            frame_idx += 1
            
            process_frame_with_buffer(fio, frame_data, frame_idx, batch)
        except Exception as e:
            logging.error(f"Error in main loop: {e}")
            continue

if __name__ == "__main__":
    main()