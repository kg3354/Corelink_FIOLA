
import os
import sys
import struct
import asyncio
import pickle
import logging
from collections import defaultdict
from time import time
import tifffile
import numpy as np
import warnings
from fiola.fiola import FIOLA
import io
import queue
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", message="no queue or thread to delete")

# sys.path.append("C:/Users/29712/corelink-client/python/package/Corelink/src")
import corelink

HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Dictionary to hold the incoming chunks for each frame
incoming_frames = defaultdict(lambda: {
    "timestamp": 0,
    "total_slices": 0,
    "received_slices": 0,
    "chunks": [],
    "start_time": time()
})

# Ensure the 'results' directory exists
os.makedirs('results', exist_ok=True)

# Thread pool executor for concurrent processing
executor = ThreadPoolExecutor(max_workers=7)  # Adjusted for 6 FIOLA objects

FIOLA_POOL_SIZE = 8  # Define the number of FIOLA objects to rotate through
fio_objects = []
fio_index = 0

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

async def memmap_from_buffer(tiff_buffer):
    buffer = io.BytesIO(tiff_buffer)
    try:
        with tifffile.TiffFile(buffer) as tif:
            tiff_series = tif.series[0]
            dtype = tiff_series.dtype
            shape = tiff_series.shape
            byte_order = tif.byteorder

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

async def process_frame(fio, memmap_image, frame_idx, timestamp, processtimestamp):
    try:
        frame_batch = memmap_image.astype(np.float32)
        fio.fit_online_frame(frame_batch)
        fio.compute_estimates()
        
        # np.save(f'./results/fiola_result_ptr_{frame_idx}', fio.estimates)
        
        end_time = time()
        total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds

        logging.info(f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}")
        logging.info(f"Total time spent processing frame {frame_idx}: {end_time - processtimestamp / 1000}")
    except Exception as e:
        logging.error(f"Error processing frame {frame_idx}: {e}")

async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp):
    try:
        memmap_image = await memmap_from_buffer(frame_data)
        await process_frame(fio, memmap_image, frame_idx, timestamp, processtimestamp)
    except Exception as e:
        logging.error(f"Failed to process frame with buffer: {e}")

async def callback(data_bytes, streamID, header):
    global incoming_frames, fio_index

    # Extract the header information
    timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
    chunk_data = data_bytes[HEADER_SIZE:]

    frame = incoming_frames[frame_number]
    frame["timestamp"] = timestamp

    # Initialize frame entry if receiving the first chunk
    if frame["received_slices"] == 0:
        frame["total_slices"] = total_chunks
        frame["chunks"] = [None] * total_chunks
        frame["start_time"] = int(time() * 1000)
    # Store the chunk data in the correct position
    if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
        frame["chunks"][chunk_index] = chunk_data
        frame["received_slices"] += 1
        print(f"Received slice {chunk_index} for frame {frame_number}")

        # Check if we have received all chunks for this frame
        if frame["received_slices"] == total_chunks:
            # Reconstruct the frame
            frame_data = b''.join(frame["chunks"])

            # Process the frame with a rotated FIOLA object concurrently
            fio = fio_objects[fio_index]
            fio_index = (fio_index + 1) % FIOLA_POOL_SIZE
            loop = asyncio.get_event_loop()
            loop.run_in_executor(executor, asyncio.run, process_frame_with_buffer(fio, frame_data, frame_number, frame["timestamp"], frame["start_time"]))

            # Clean up the completed frame entry
            del incoming_frames[frame_number]
    else:
        print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

async def update(response, key):
    print(f'Updating as new sender valid in the workspace: {response}')

async def stale(response, key):
    print(response)

async def main():
    global fio_objects
    for _ in range(FIOLA_POOL_SIZE):
        fio_objects.append(load_fiola_state('fiola_state_msCam.pkl'))

    await corelink.set_server_callback(update, 'update')
    await corelink.set_server_callback(stale, 'stale')
    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    await corelink.set_data_callback(callback)
    
    receiver_id = await corelink.create_receiver("FentonCtl", "ws", alert=True, echo=True)
    print(receiver_id)
    
    print("Start receiving")
    await corelink.keep_open()
    
    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        print('Receiver terminated.')

if __name__ == "__main__":
    corelink.run(main())
