
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
    """Load the FIOLA state from a pickle file."""
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

def memmap_from_buffer(tiff_buffer):
    buffer = io.BytesIO(tiff_buffer)
    
    with tifffile.TiffFile(buffer) as tif:
        tiff_series = tif.series[0]
        dtype = tiff_series.dtype
        shape = tiff_series.shape
        byte_order = tif.byteorder

        # logging.info(f"Expected shape: {shape}, dtype: {dtype}")
        # logging.info(f"TIFF Series Details: {tiff_series}")

        image_data = np.zeros(np.prod(shape), dtype=np.dtype(byte_order + dtype.char))

        for page in tif.pages:
            # logging.info(f"Page {page.index}: offset={page.dataoffsets}, size={page.databytecounts}")
            # logging.info(f"Page {page.index} is memmappable: {page.is_memmappable}")

            for offset, size in zip(page.dataoffsets, page.databytecounts):
                buffer.seek(offset)
                data = np.frombuffer(buffer.read(size), dtype=np.dtype(byte_order + dtype.char))
                start = page.index * size
                image_data[start:start + size] = data

    # logging.info(f"Data array size: {image_data.size}, expected size: {np.prod(shape)}, buffer size: {len(tiff_buffer)}")

    if image_data.size != np.prod(shape):
        # logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
        raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

    try:
        image_data = image_data.reshape(shape)
    except ValueError as e:
        logging.error(f"Error reshaping array: {e}")
        raise

    return image_data

def process_frame(fio, memmap_image, frame_idx, batch, online_trace, online_trace_deconvolved):
    result_folder = "./CaImAn/example_movies/frame_sample/results"
    
    # Ensure the result directory exists
    os.makedirs(result_folder, exist_ok=True)
    
    start_time = time()
    logging.info('Retrieving shape')
    num_frames_total = memmap_image.shape[0]
    
    for idx in range(0, num_frames_total, batch):
        frame_batch = memmap_image[idx:idx + batch].astype(np.float32)
        logging.info(f'Fit online frame for batch {idx}')
        fio.fit_online_frame(frame_batch)
        # Uncomment and adjust the following lines if you need to update
        # traces
        # online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx - batch:idx]
        # online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx - batch - fio.params.retrieve['lag']:idx - fio.params.retrieve['lag']]

    fio.compute_estimates()
    
    # Save result
    result_file = os.path.join(result_folder, f"fiola_result_{frame_idx}_{time()}.npy")
    np.save(result_file, fio.estimates)
    
    end_time = time()
    total_time = end_time - start_time
    logging.info(f"Total time spent processing frame {frame_idx}: {total_time}")

def process_frame_with_buffer(fio, frame_data, frame_idx, batch, online_trace, online_trace_deconvolved):
    try:
        # Try to create a memory-mapped-like array from the buffer
        memmap_image = memmap_from_buffer(frame_data)
    except ValueError as e:
        logging.error(f"Failed to create memmap from buffer: {e}")
        return  # Skip processing this frame if there's an error
    
    # Process the frame
    process_frame(fio, memmap_image, frame_idx, batch, online_trace, online_trace_deconvolved)

def main():
    fio = load_fiola_state('fiola_state_msCam.pkl')
    frame_idx = 0

    batch = 1  # Number of frames processed at the same time using GPU
    online_trace = np.zeros((fio.Ab.shape[-1], 0), dtype=np.float32)
    online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], 0), dtype=np.float32)

    while True:
        # Read frame size from stdin
        frame_size_data = sys.stdin.buffer.read(4)
        if not frame_size_data:
            break
        frame_size = int.from_bytes(frame_size_data, byteorder='little')
        
        # Read frame data from stdin
        frame_data = sys.stdin.buffer.read(frame_size)
        if not frame_data or len(frame_data) != frame_size:
            break
        
        frame_idx += 1
        
        # Process the frame with buffer
        process_frame_with_buffer(fio, frame_data, frame_idx, batch, online_trace, online_trace_deconvolved)

if __name__ == "__main__":
    main()
