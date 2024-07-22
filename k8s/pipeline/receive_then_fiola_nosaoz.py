import asyncio
import os
import sys
import struct
import pickle
import logging
from collections import defaultdict
from time import time
import tifffile
import numpy as np
import warnings
import psutil
from fiola.fiola import FIOLA
from fiola.signal_analysis_online import SignalAnalysisOnlineZ
import io
from numba import njit, prange
import corelink
from corelink.resources.control import subscribe_to_stream
import receive_then_init
import queue
warnings.filterwarnings("ignore", message="no queue or thread to delete")

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

LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'
fio_objects=[]
# Loading FIOLA state from a pickle file 
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
    if not hasattr(fio.pipeline, 'saoz'):
        fio.pipeline.saoz = SignalAnalysisOnlineZ()
    fio.pipeline.saoz.update_q = queue.Queue()
    
    return fio

# Reading each tiff frame in buffer, converting to the format that FIOLA wants
def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
    image_data = np.zeros(shape, dtype=dtype)
    for i in range(len(offsets)):
        buffer.seek(offsets[i])
        data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

# Using numba to speed up numerical processing
@njit(parallel=True)
def process_tiff_data(image_data, offsets, bytecounts, dtype, shape):
    for i in prange(len(offsets)):
        data = np.frombuffer(image_data[offsets[i]:offsets[i]+bytecounts[i]], dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

# Process the frame data in buffer, to avoid disk io
async def memmap_from_buffer(tiff_buffer):
    buffer = io.BytesIO(tiff_buffer)
    try:
        with tifffile.TiffFile(buffer) as tif:
            tiff_series = tif.series[0]
            dtype = tiff_series.dtype
            shape = tiff_series.shape
            byte_order = tif.byteorder

            # Accept 1 frame input only
            shape = (1, *shape)
            logging.info(f"Shape: {shape}")

            # Initialize image_data to hold the entire TIFF data
            image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))
            offsets = []
            bytecounts = []

            for page in tif.pages:
                offsets.extend(page.dataoffsets)
                bytecounts.extend(page.databytecounts)

            image_data = read_tiff_data(buffer, offsets, bytecounts, np.dtype(byte_order + dtype.char), shape)

        if image_data.size != np.prod(shape):
            logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
            raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

        logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
        return image_data
    except Exception as e:
        logging.error(f"Error processing TIFF buffer: {e}")
        raise

def process_frame_data(memmap_image):
    frame_batch = memmap_image.astype(np.float32)
    return frame_batch

async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp):
    try:
        start_time = time()
        memmap_image = await memmap_from_buffer(frame_data)
        buffer_time = time()
        frame_batch = process_frame_data(memmap_image)
        proc_time = time()
        fio.fit_online_frame(frame_batch)
        fio.compute_estimates()
        end_time = time()
        total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds
        message = f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}, total time processing is {end_time-proc_time}, buffering time: {buffer_time - start_time}, start to finish: {end_time - start_time}"
        print(message)
        await corelink.send(sender_id, message)
    except Exception as e:
        logging.error(f"Failed to process frame with buffer: {e}")

async def callback(data_bytes, streamID, header):
    global incoming_frames

    # Extract the header information
    timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
    chunk_data = data_bytes[HEADER_SIZE:]
    arrival_time = time()

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

        # Check if we have received all chunks for this frame
        if frame["received_slices"] == total_chunks:
            # Log transmission time
            transmission_time = time() - timestamp / 1000
            logging.info(f"Frame {frame_number} transmission time: {transmission_time:.6f}s")

            # Reconstruct the frame
            frame_data = b''.join(frame["chunks"])

            # Process the frame with the single FIOLA object
            asyncio.create_task(process_frame_with_buffer(fio_objects[0], frame_data, frame_number, frame["timestamp"], frame["start_time"]))

            # Clean up the completed frame entry
            del incoming_frames[frame_number]

            # Log arrival time and processing start time
            logging.info(f"Frame {frame_number} fully received at {arrival_time:.6f}, started processing at {time():.6f}")
    else:
        logging.info(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

async def update(response, key):
    logging.info(f'Updating as new sender valid in the workspace: {response}')
    await subscribe_to_stream(response['receiverID'], response['streamID'])

async def stale(response, key):
    logging.info(response)

async def subscriber(response, key):
    logging.info(f"subscriber: {response}")

async def dropped(response, key):    
    logging.info(f"dropped: {response}")

async def processing():
    global fio_objects, sender_id

    # Read the latest FIOLA state path
    if os.path.exists(LATEST_FIOLA_STATE_PATH):
        with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
            latest_fiola_state_file = f.read().strip()

        if os.path.exists(latest_fiola_state_file):
            logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
            fio_objects.append(load_fiola_state(latest_fiola_state_file))
        else:
            logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
            sys.exit(1)
    else:
        logging.info("Generating new FIOLA init file")
        terminate_event = asyncio.Event()
        await receive_then_init.receive_then_init(terminate_event)
        logging.info("Completed receive_then_init")

        if os.path.exists(LATEST_FIOLA_STATE_PATH):
            with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
                latest_fiola_state_file = f.read().strip()
            if os.path.exists(latest_fiola_state_file):
                fio_objects.append(load_fiola_state(latest_fiola_state_file))
            else:
                logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
                sys.exit(1)
        else:
            logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
            sys.exit(1)

    await corelink.set_server_callback(update, 'update')
    await corelink.set_server_callback(stale, 'stale')
    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    await corelink.set_data_callback(callback)
    
    receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
    logging.info(f"Receiver ID: {receiver_id}")
    
    logging.info("Start receive process frames")
    await corelink.keep_open()
    
    await corelink.set_server_callback(subscriber, 'subscriber')
    await corelink.set_server_callback(dropped, 'dropped')

    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")

    try:
        while True:
            await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logging.info('Receiver terminated.')

if __name__ == "__main__":
    corelink.run(processing())

# Assigning CPU affinity
p = psutil.Process(os.getpid())

# Assign all CPUs to the process
numa_nodes = psutil.cpu_count(logical=False)
numa_cpus = list(range(numa_nodes))
p.cpu_affinity(numa_cpus)
