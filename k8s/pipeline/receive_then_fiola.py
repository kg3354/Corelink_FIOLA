
# # # # import cProfile
# # # # import pstats
# # # # import asyncio
# # # # import os
# # # # import sys
# # # # import struct
# # # # import pickle
# # # # import logging
# # # # from collections import defaultdict
# # # # from time import time
# # # # import tifffile
# # # # import numpy as np
# # # # import warnings
# # # # from fiola.fiola import FIOLA
# # # # import io
# # # # import queue
# # # # from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# # # # from numba import njit, prange
# # # # import subprocess

# # # # warnings.filterwarnings("ignore", message="no queue or thread to delete")

# # # # import corelink
# # # # from corelink.resources.control import subscribe_to_stream


# # # # import receive_then_init

# # # # HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)

# # # # # Configure logging
# # # # logging.basicConfig(
# # # #     format="%(asctime)s [%(levelname)s] %(message)s",
# # # #     level=logging.INFO,
# # # #     handlers=[
# # # #         logging.StreamHandler(sys.stdout)
# # # #     ]
# # # # )

# # # # # Dictionary to hold the incoming chunks for each frame
# # # # incoming_frames = defaultdict(lambda: {
# # # #     "timestamp": 0,
# # # #     "total_slices": 0,
# # # #     "received_slices": 0,
# # # #     "chunks": [],
# # # #     "start_time": time()
# # # # })

# # # # # Ensure the 'results' directory exists
# # # # os.makedirs('results', exist_ok=True)

# # # # # Process pool executor for concurrent processing
# # # # executor = ProcessPoolExecutor(max_workers=8)  # Adjusted for parallel processing

# # # # FIOLA_POOL_SIZE = 8  # Define the number of FIOLA objects to rotate through
# # # # fio_objects = []
# # # # fio_index = 0

# # # # LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'

# # # # #%%    
# # # # # Loading fiola state from a pickle file 
# # # # def load_fiola_state(filepath):
# # # #     with open(filepath, 'rb') as f:
# # # #         fio_state = pickle.load(f)
# # # #     params = fio_state['params']
# # # #     trace_fiola = np.array(fio_state['trace_fiola'], dtype=np.float32)
# # # #     template = np.array(fio_state['template'], dtype=np.float32)
# # # #     Ab = np.array(fio_state['Ab'], dtype=np.float32)
# # # #     min_mov = fio_state['min_mov']
# # # #     mc_nn_mov = np.array(fio_state['mov_data'], dtype=np.float32)
    
# # # #     fio = FIOLA(params=params)
# # # #     fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
# # # #     fio.pipeline.saoz.update_q = queue.Queue()
    
# # # #     return fio

# # # # #%%    
# # # # # Reading each tiff frame in buffer, converting to the format that FIOLA wants
# # # # @njit(parallel=True)
# # # # def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
# # # #     image_data = np.zeros(shape, dtype=dtype)
# # # #     for i in prange(len(offsets)):
# # # #         buffer.seek(offsets[i])
# # # #         data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
# # # #         image_data[i, ...] = data.reshape(shape[1:])
# # # #     return image_data

# # # # #%%    
# # # # #
# # # # async def memmap_from_buffer(tiff_buffer):
# # # #     buffer = io.BytesIO(tiff_buffer)
# # # #     try:
# # # #         with tifffile.TiffFile(buffer) as tif:
# # # #             tiff_series = tif.series[0]
# # # #             dtype = tiff_series.dtype
# # # #             shape = tiff_series.shape
# # # #             byte_order = tif.byteorder

# # # #             # Accept 1 frame input only
# # # #             shape = (1, *shape)
# # # #             logging.info(f"Shape: {shape}")

# # # #             # Initialize image_data to hold the entire TIFF data
# # # #             image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))

# # # #             for page_index, page in enumerate(tif.pages):
# # # #                 offsets = page.dataoffsets
# # # #                 bytecounts = page.databytecounts
# # # #                 image_data[page_index, ...] = read_tiff_data(buffer, offsets, bytecounts, np.dtype(byte_order + dtype.char), shape)

# # # #         if image_data.size != np.prod(shape):
# # # #             logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
# # # #             raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

# # # #         logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
# # # #         return image_data
# # # #     except Exception as e:
# # # #         logging.error(f"Error processing TIFF buffer: {e}")
# # # #         raise

# # # # async def process_frame_parallel(fio, memmap_image, frame_idx, timestamp, processtimestamp):
# # # #     try:
# # # #         frame_batch = memmap_image.astype(np.float32)
# # # #         fio.fit_online_frame(frame_batch)
# # # #         fio.compute_estimates()
        
# # # #         # np.save(f'./results/fiola_result_ptr_{frame_idx}', fio.estimates)
        
# # # #         end_time = time()
# # # #         total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds

# # # #         message = f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}"
# # # #         logging.info(message)
# # # #         logging.info(f"Total time spent processing frame {frame_idx}: {end_time - processtimestamp / 1000}")

# # # #         await corelink.send(sender_id, message)


# # # #     except Exception as e:
# # # #         logging.error(f"Error processing frame {frame_idx}: {e}")

# # # # async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp):
# # # #     try:
# # # #         memmap_image = await memmap_from_buffer(frame_data)
# # # #         loop = asyncio.get_event_loop()
# # # #         await loop.run_in_executor(executor, process_frame_parallel, fio, memmap_image, frame_idx, timestamp, processtimestamp)
# # # #     except Exception as e:
# # # #         logging.error(f"Failed to process frame with buffer: {e}")

# # # # async def callback(data_bytes, streamID, header):
# # # #     global incoming_frames, fio_index

# # # #     # Extract the header information
# # # #     timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
# # # #     chunk_data = data_bytes[HEADER_SIZE:]

# # # #     frame = incoming_frames[frame_number]
# # # #     frame["timestamp"] = timestamp

# # # #     # Initialize frame entry if receiving the first chunk
# # # #     if frame["received_slices"] == 0:
# # # #         frame["total_slices"] = total_chunks
# # # #         frame["chunks"] = [None] * total_chunks
# # # #         frame["start_time"] = int(time() * 1000)
# # # #     # Store the chunk data in the correct position
# # # #     if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
# # # #         frame["chunks"][chunk_index] = chunk_data
# # # #         frame["received_slices"] += 1
# # # #         print(f"Received slice {chunk_index} for frame {frame_number}")

# # # #         # Check if we have received all chunks for this frame
# # # #         if frame["received_slices"] == total_chunks:
# # # #             # Reconstruct the frame
# # # #             frame_data = b''.join(frame["chunks"])

# # # #             # Process the frame with a rotated FIOLA object concurrently
# # # #             fio = fio_objects[fio_index]
# # # #             fio_index = (fio_index + 1) % FIOLA_POOL_SIZE
# # # #             loop = asyncio.get_event_loop()
# # # #             loop.run_in_executor(executor, asyncio.run, process_frame_with_buffer(fio, frame_data, frame_number, frame["timestamp"], frame["start_time"]))

# # # #             # Clean up the completed frame entry
# # # #             del incoming_frames[frame_number]
# # # #     else:
# # # #         print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

# # # # async def update(response, key):
# # # #     print(f'Updating as new sender valid in the workspace: {response}')
# # # #     await subscribe_to_stream(response['receiverID'], response['streamID'])

# # # # async def stale(response, key):
# # # #     print(response)

# # # # async def subscriber(response, key):
# # # #     print("subscriber: ", response)


# # # # async def dropped(response, key):    
# # # #     print("dropped", response)



# # # # async def processing():
# # # #     global fio_objects, sender_id

# # # #     # Read the latest FIOLA state path
# # # #     if os.path.exists(LATEST_FIOLA_STATE_PATH):
# # # #         with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
# # # #             latest_fiola_state_file = f.read().strip()

# # # #         if os.path.exists(latest_fiola_state_file):
# # # #             logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
# # # #             for _ in range(FIOLA_POOL_SIZE):
# # # #                 fio_objects.append(load_fiola_state(latest_fiola_state_file))
# # # #         else:
# # # #             logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
# # # #             sys.exit(1)
# # # #     else:
# # # #         logging.info("Generating new FIOLA init file")
        
# # # #         #subprocess.run(["python3.8", "./generate_init_result.py"], check=True)

# # # #         await receive_then_init.receive_then_init()


# # # #         if os.path.exists(LATEST_FIOLA_STATE_PATH):
# # # #             with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
# # # #                 latest_fiola_state_file = f.read().strip()
# # # #             if os.path.exists(latest_fiola_state_file):
# # # #                 for _ in range(FIOLA_POOL_SIZE):
# # # #                     fio_objects.append(load_fiola_state(latest_fiola_state_file))
# # # #             else:
# # # #                 logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
# # # #                 sys.exit(1)
# # # #         else:
# # # #             logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
# # # #             sys.exit(1)

# # # #     await corelink.set_server_callback(update, 'update')
# # # #     await corelink.set_server_callback(stale, 'stale')
# # # #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# # # #     await corelink.set_data_callback(callback)
    
# # # #     receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
# # # #     print(receiver_id)
    
# # # #     print("Start receive process frames")
# # # #     await corelink.keep_open()
     

# # # #     await corelink.set_server_callback(subscriber, 'subscriber')
# # # #     await corelink.set_server_callback(dropped, 'dropped')

# # # #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# # # #     sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")



# # # #     try:
# # # #         while True:
# # # #             await asyncio.sleep(3600)
# # # #     except KeyboardInterrupt:
# # # #         print('Receiver terminated.')

# # # # async def run_main_with_profile():
# # # #     profiler = cProfile.Profile()
# # # #     profiler.enable()

# # # #     main_task = asyncio.create_task(processing())

# # # #     while True:
# # # #         await asyncio.sleep(300)  # Wait for 5 minutes
# # # #         profiler.disable()
# # # #         with open("/usr/src/app/profile_data/profile_output.prof", "w") as f:
# # # #             ps = pstats.Stats(profiler, stream=f)
# # # #             ps.sort_stats("cumulative")
# # # #             ps.print_stats()
# # # #         profiler.enable()
# # # #         if main_task.done():
# # # #             break

# # # #     profiler.disable()
# # # #     with open("/usr/src/app/profile_data/profile_output.prof", "w") as f:
# # # #         ps = pstats.Stats(profiler, stream=f)
# # # #         ps.sort_stats("cumulative")
# # # #         ps.print_stats()

# # # # if __name__ == "__main__":
# # # #     corelink.run(run_main_with_profile())
# # # import asyncio
# # # import os
# # # import sys
# # # import struct
# # # import pickle
# # # import logging
# # # from collections import defaultdict
# # # from time import time
# # # import tifffile
# # # import numpy as np
# # # import warnings
# # # from fiola.fiola import FIOLA
# # # import io
# # # import queue
# # # from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
# # # from numba import njit, prange
# # # import subprocess

# # # warnings.filterwarnings("ignore", message="no queue or thread to delete")

# # # import corelink
# # # from corelink.resources.control import subscribe_to_stream


# # # import receive_then_init

# # # HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)

# # # # Configure logging
# # # logging.basicConfig(
# # #     format="%(asctime)s [%(levelname)s] %(message)s",
# # #     level=logging.INFO,
# # #     handlers=[
# # #         logging.StreamHandler(sys.stdout)
# # #     ]
# # # )

# # # # Dictionary to hold the incoming chunks for each frame
# # # incoming_frames = defaultdict(lambda: {
# # #     "timestamp": 0,
# # #     "total_slices": 0,
# # #     "received_slices": 0,
# # #     "chunks": [],
# # #     "start_time": time()
# # # })

# # # # Ensure the 'results' directory exists
# # # os.makedirs('results', exist_ok=True)

# # # # Process pool executor for concurrent processing
# # # executor = ProcessPoolExecutor(max_workers=8)  # Adjusted for parallel processing

# # # FIOLA_POOL_SIZE = 8  # Define the number of FIOLA objects to rotate through
# # # fio_objects = []
# # # fio_index = 0

# # # LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'

# # # #%%    
# # # # Loading fiola state from a pickle file 
# # # def load_fiola_state(filepath):
# # #     with open(filepath, 'rb') as f:
# # #         fio_state = pickle.load(f)
# # #     params = fio_state['params']
# # #     trace_fiola = np.array(fio_state['trace_fiola'], dtype=np.float32)
# # #     template = np.array(fio_state['template'], dtype=np.float32)
# # #     Ab = np.array(fio_state['Ab'], dtype=np.float32)
# # #     min_mov = fio_state['min_mov']
# # #     mc_nn_mov = np.array(fio_state['mov_data'], dtype=np.float32)
    
# # #     fio = FIOLA(params=params)
# # #     fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
# # #     fio.pipeline.saoz.update_q = queue.Queue()
    
# # #     return fio

# # # #%%    
# # # # Reading each tiff frame in buffer, converting to the format that FIOLA wants
# # # @njit(parallel=True)
# # # def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
# # #     image_data = np.zeros(shape, dtype=dtype)
# # #     for i in prange(len(offsets)):
# # #         buffer.seek(offsets[i])
# # #         data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
# # #         image_data[i, ...] = data.reshape(shape[1:])
# # #     return image_data

# # # #%%    
# # # #
# # # async def memmap_from_buffer(tiff_buffer):
# # #     buffer = io.BytesIO(tiff_buffer)
# # #     try:
# # #         with tifffile.TiffFile(buffer) as tif:
# # #             tiff_series = tif.series[0]
# # #             dtype = tiff_series.dtype
# # #             shape = tiff_series.shape
# # #             byte_order = tif.byteorder

# # #             # Accept 1 frame input only
# # #             shape = (1, *shape)
# # #             logging.info(f"Shape: {shape}")

# # #             # Initialize image_data to hold the entire TIFF data
# # #             image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))

# # #             for page_index, page in enumerate(tif.pages):
# # #                 offsets = page.dataoffsets
# # #                 bytecounts = page.databytecounts
# # #                 image_data[page_index, ...] = read_tiff_data(buffer, offsets, bytecounts, np.dtype(byte_order + dtype.char), shape)

# # #         if image_data.size != np.prod(shape):
# # #             logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
# # #             raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

# # #         logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
# # #         return image_data
# # #     except Exception as e:
# # #         logging.error(f"Error processing TIFF buffer: {e}")
# # #         raise

# # # def process_frame_parallel(fio, memmap_image, frame_idx, timestamp, processtimestamp):
# # #     try:
# # #         frame_batch = memmap_image.astype(np.float32)
# # #         fio.fit_online_frame(frame_batch)
# # #         fio.compute_estimates()
        
# # #         # np.save(f'./results/fiola_result_ptr_{frame_idx}', fio.estimates)
        
# # #         end_time = time()
# # #         total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds

# # #         message = f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}"
# # #         return message
# # #     except Exception as e:
# # #         logging.error(f"Error processing frame {frame_idx}: {e}")

# # # async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp):
# # #     try:
# # #         memmap_image = await memmap_from_buffer(frame_data)
# # #         loop = asyncio.get_event_loop()
# # #         message = await loop.run_in_executor(executor, process_frame_parallel, fio, memmap_image, frame_idx, timestamp, processtimestamp)
# # #         await corelink.send(sender_id, message)
# # #     except Exception as e:
# # #         logging.error(f"Failed to process frame with buffer: {e}")


# # # async def callback(data_bytes, streamID, header):
# # #     global incoming_frames, fio_index

# # #     # Extract the header information
# # #     timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
# # #     chunk_data = data_bytes[HEADER_SIZE:]

# # #     frame = incoming_frames[frame_number]
# # #     frame["timestamp"] = timestamp

# # #     # Initialize frame entry if receiving the first chunk
# # #     if frame["received_slices"] == 0:
# # #         frame["total_slices"] = total_chunks
# # #         frame["chunks"] = [None] * total_chunks
# # #         frame["start_time"] = int(time() * 1000)
# # #     # Store the chunk data in the correct position
# # #     if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
# # #         frame["chunks"][chunk_index] = chunk_data
# # #         frame["received_slices"] += 1
# # #         print(f"Received slice {chunk_index} for frame {frame_number}")

# # #         # Check if we have received all chunks for this frame
# # #         if frame["received_slices"] == total_chunks:
# # #             # Reconstruct the frame
# # #             frame_data = b''.join(frame["chunks"])

# # #             # Process the frame with a rotated FIOLA object concurrently
# # #             fio = fio_objects[fio_index]
# # #             fio_index = (fio_index + 1) % FIOLA_POOL_SIZE
# # #             loop = asyncio.get_event_loop()
# # #             loop.run_in_executor(executor, asyncio.run, process_frame_with_buffer(fio, frame_data, frame_number, frame["timestamp"], frame["start_time"]))

# # #             # Clean up the completed frame entry
# # #             del incoming_frames[frame_number]
# # #     else:
# # #         print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

# # # async def update(response, key):
# # #     print(f'Updating as new sender valid in the workspace: {response}')
# # #     await subscribe_to_stream(response['receiverID'], response['streamID'])

# # # async def stale(response, key):
# # #     print(response)

# # # async def subscriber(response, key):
# # #     print("subscriber: ", response)


# # # async def dropped(response, key):    
# # #     print("dropped", response)



# # # async def processing():
# # #     global fio_objects, sender_id

# # #     # Read the latest FIOLA state path
# # #     if os.path.exists(LATEST_FIOLA_STATE_PATH):
# # #         with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
# # #             latest_fiola_state_file = f.read().strip()

# # #         if os.path.exists(latest_fiola_state_file):
# # #             logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
# # #             for _ in range(FIOLA_POOL_SIZE):
# # #                 fio_objects.append(load_fiola_state(latest_fiola_state_file))
# # #         else:
# # #             logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
# # #             sys.exit(1)
# # #     else:
# # #         logging.info("Generating new FIOLA init file")
        
# # #         #subprocess.run(["python3.8", "./generate_init_result.py"], check=True)
# # #         terminate_event = asyncio.Event()
# # #         await receive_then_init.receive_then_init(terminate_event)
# # #         logging.info("Completed receive_then_init")

# # #         if os.path.exists(LATEST_FIOLA_STATE_PATH):
# # #             with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
# # #                 latest_fiola_state_file = f.read().strip()
# # #             if os.path.exists(latest_fiola_state_file):
# # #                 for _ in range(FIOLA_POOL_SIZE):
# # #                     fio_objects.append(load_fiola_state(latest_fiola_state_file))
# # #             else:
# # #                 logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
# # #                 sys.exit(1)
# # #         else:
# # #             logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
# # #             sys.exit(1)

# # #     await corelink.set_server_callback(update, 'update')
# # #     await corelink.set_server_callback(stale, 'stale')
# # #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# # #     await corelink.set_data_callback(callback)
    
# # #     receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
# # #     print(receiver_id)
    
# # #     print("Start receive process frames")
# # #     await corelink.keep_open()
     

# # #     await corelink.set_server_callback(subscriber, 'subscriber')
# # #     await corelink.set_server_callback(dropped, 'dropped')

# # #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# # #     sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")



# # #     try:
# # #         while True:
# # #             await asyncio.sleep(3600)
# # #     except KeyboardInterrupt:
# # #         print('Receiver terminated.')

# # # if __name__ == "__main__":
# # #     corelink.run(processing())
# # import asyncio
# # import os
# # import sys
# # import struct
# # import pickle
# # import logging
# # from collections import defaultdict
# # from time import time
# # import tifffile
# # import numpy as np
# # import warnings
# # from fiola.fiola import FIOLA
# # import io
# # import queue
# # from concurrent.futures import ProcessPoolExecutor
# # from numba import njit, prange

# # warnings.filterwarnings("ignore", message="no queue or thread to delete")

# # import corelink
# # from corelink.resources.control import subscribe_to_stream
# # import receive_then_init

# # HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)

# # # Configure logging
# # logging.basicConfig(
# #     format="%(asctime)s [%(levelname)s] %(message)s",
# #     level=logging.INFO,
# #     handlers=[
# #         logging.StreamHandler(sys.stdout)
# #     ]
# # )

# # # Dictionary to hold the incoming chunks for each frame
# # incoming_frames = defaultdict(lambda: {
# #     "timestamp": 0,
# #     "total_slices": 0,
# #     "received_slices": 0,
# #     "chunks": [],
# #     "start_time": time()
# # })

# # # Ensure the 'results' directory exists
# # os.makedirs('results', exist_ok=True)

# # # Process pool executor for concurrent processing
# # executor = ProcessPoolExecutor(max_workers=os.cpu_count())  # Adjusted for parallel processing

# # FIOLA_POOL_SIZE = 8  # Define the number of FIOLA objects to rotate through
# # fio_objects = []
# # fio_index = 0

# # LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'

# # #%%    
# # # Loading fiola state from a pickle file 
# # def load_fiola_state(filepath):
# #     with open(filepath, 'rb') as f:
# #         fio_state = pickle.load(f)
# #     params = fio_state['params']
# #     trace_fiola = np.array(fio_state['trace_fiola'], dtype=np.float32)
# #     template = np.array(fio_state['template'], dtype=np.float32)
# #     Ab = np.array(fio_state['Ab'], dtype=np.float32)
# #     min_mov = fio_state['min_mov']
# #     mc_nn_mov = np.array(fio_state['mov_data'], dtype=np.float32)
    
# #     fio = FIOLA(params=params)
# #     fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
# #     fio.pipeline.saoz.update_q = queue.Queue()
    
# #     return fio

# # #%%    
# # # Reading each tiff frame in buffer, converting to the format that FIOLA wants
# # @njit(parallel=True)
# # def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
# #     image_data = np.zeros(shape, dtype=dtype)
# #     for i in prange(len(offsets)):
# #         buffer.seek(offsets[i])
# #         data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
# #         image_data[i, ...] = data.reshape(shape[1:])
# #     return image_data

# # #%%    
# # #
# # async def memmap_from_buffer(tiff_buffer):
# #     buffer = io.BytesIO(tiff_buffer)
# #     try:
# #         with tifffile.TiffFile(buffer) as tif:
# #             tiff_series = tif.series[0]
# #             dtype = tiff_series.dtype
# #             shape = tiff_series.shape
# #             byte_order = tif.byteorder

# #             # Accept 1 frame input only
# #             shape = (1, *shape)
# #             logging.info(f"Shape: {shape}")

# #             # Initialize image_data to hold the entire TIFF data
# #             image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))

# #             for page_index, page in enumerate(tif.pages):
# #                 offsets = page.dataoffsets
# #                 bytecounts = page.databytecounts
# #                 image_data[page_index, ...] = read_tiff_data(buffer, offsets, bytecounts, np.dtype(byte_order + dtype.char), shape)

# #         if image_data.size != np.prod(shape):
# #             logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
# #             raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

# #         logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
# #         return image_data
# #     except Exception as e:
# #         logging.error(f"Error processing TIFF buffer: {e}")
# #         raise

# # def process_frame_parallel(fio, memmap_image, frame_idx, timestamp, processtimestamp):
# #     try:
# #         frame_batch = memmap_image.astype(np.float32)
# #         fio.fit_online_frame(frame_batch)
# #         fio.compute_estimates()
        
# #         # np.save(f'./results/fiola_result_ptr_{frame_idx}', fio.estimates)
        
# #         end_time = time()
# #         total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds

# #         message = f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}"
# #         return message
# #     except Exception as e:
# #         logging.error(f"Error processing frame {frame_idx}: {e}")

# # async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp):
# #     try:
# #         memmap_image = await memmap_from_buffer(frame_data)
# #         loop = asyncio.get_event_loop()
# #         message = await loop.run_in_executor(executor, process_frame_parallel, fio, memmap_image, frame_idx, timestamp, processtimestamp)
# #         await corelink.send(sender_id, message)
# #     except Exception as e:
# #         logging.error(f"Failed to process frame with buffer: {e}")


# # async def callback(data_bytes, streamID, header):
# #     global incoming_frames, fio_index

# #     # Extract the header information
# #     timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
# #     chunk_data = data_bytes[HEADER_SIZE:]

# #     frame = incoming_frames[frame_number]
# #     frame["timestamp"] = timestamp

# #     # Initialize frame entry if receiving the first chunk
# #     if frame["received_slices"] == 0:
# #         frame["total_slices"] = total_chunks
# #         frame["chunks"] = [None] * total_chunks
# #         frame["start_time"] = int(time() * 1000)
# #     # Store the chunk data in the correct position
# #     if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
# #         frame["chunks"][chunk_index] = chunk_data
# #         frame["received_slices"] += 1
# #         logging.info(f"Received slice {chunk_index} for frame {frame_number}")

# #         # Check if we have received all chunks for this frame
# #         if frame["received_slices"] == total_chunks:
# #             # Reconstruct the frame
# #             frame_data = b''.join(frame["chunks"])

# #             # Process the frame with a rotated FIOLA object concurrently
# #             fio = fio_objects[fio_index]
# #             fio_index = (fio_index + 1) % FIOLA_POOL_SIZE
# #             asyncio.create_task(process_frame_with_buffer(fio, frame_data, frame_number, frame["timestamp"], frame["start_time"]))

# #             # Clean up the completed frame entry
# #             del incoming_frames[frame_number]
# #     else:
# #         logging.info(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

# # async def update(response, key):
# #     logging.info(f'Updating as new sender valid in the workspace: {response}')
# #     await subscribe_to_stream(response['receiverID'], response['streamID'])

# # async def stale(response, key):
# #     logging.info(response)

# # async def subscriber(response, key):
# #     logging.info(f"subscriber: {response}")

# # async def dropped(response, key):    
# #     logging.info(f"dropped: {response}")

# # async def processing():
# #     global fio_objects, sender_id

# #     # Read the latest FIOLA state path
# #     if os.path.exists(LATEST_FIOLA_STATE_PATH):
# #         with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
# #             latest_fiola_state_file = f.read().strip()

# #         if os.path.exists(latest_fiola_state_file):
# #             logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
# #             for _ in range(FIOLA_POOL_SIZE):
# #                 fio_objects.append(load_fiola_state(latest_fiola_state_file))
# #         else:
# #             logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
# #             sys.exit(1)
# #     else:
# #         logging.info("Generating new FIOLA init file")
        
# #         #subprocess.run(["python3.8", "./generate_init_result.py"], check=True)
# #         terminate_event = asyncio.Event()
# #         await receive_then_init.receive_then_init(terminate_event)
# #         logging.info("Completed receive_then_init")

# #         if os.path.exists(LATEST_FIOLA_STATE_PATH):
# #             with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
# #                 latest_fiola_state_file = f.read().strip()
# #             if os.path.exists(latest_fiola_state_file):
# #                 for _ in range(FIOLA_POOL_SIZE):
# #                     fio_objects.append(load_fiola_state(latest_fiola_state_file))
# #             else:
# #                 logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
# #                 sys.exit(1)
# #         else:
# #             logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
# #             sys.exit(1)

# #     await corelink.set_server_callback(update, 'update')
# #     await corelink.set_server_callback(stale, 'stale')
# #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# #     await corelink.set_data_callback(callback)
    
# #     receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
# #     logging.info(f"Receiver ID: {receiver_id}")
    
# #     logging.info("Start receive process frames")
# #     await corelink.keep_open()
    
# #     await corelink.set_server_callback(subscriber, 'subscriber')
# #     await corelink.set_server_callback(dropped, 'dropped')

# #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# #     sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")

# #     try:
# #         while True:
# #             await asyncio.sleep(3600)
# #     except KeyboardInterrupt:
# #         logging.info('Receiver terminated.')

# # if __name__ == "__main__":
# #     corelink.run(processing())
# import asyncio
# import os
# import sys
# import struct
# import pickle
# import logging
# from collections import defaultdict
# from time import time
# import tifffile
# import numpy as np
# import warnings
# from fiola.fiola import FIOLA
# import io
# import queue
# from concurrent.futures import ProcessPoolExecutor
# from numba import njit, prange

# warnings.filterwarnings("ignore", message="no queue or thread to delete")

# import corelink
# from corelink.resources.control import subscribe_to_stream
# import receive_then_init

# HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)

# # Configure logging
# logging.basicConfig(
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     level=logging.INFO,
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# # Dictionary to hold the incoming chunks for each frame
# incoming_frames = defaultdict(lambda: {
#     "timestamp": 0,
#     "total_slices": 0,
#     "received_slices": 0,
#     "chunks": [],
#     "start_time": time()
# })

# # Ensure the 'results' directory exists
# os.makedirs('results', exist_ok=True)

# # Process pool executor for concurrent processing
# executor = ProcessPoolExecutor(max_workers=os.cpu_count())  # Adjusted for parallel processing

# FIOLA_POOL_SIZE = 8  # Define the number of FIOLA objects to rotate through
# fio_objects = []
# fio_index = 0

# LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'

# #%%    
# # Loading fiola state from a pickle file 
# def load_fiola_state(filepath):
#     with open(filepath, 'rb') as f:
#         fio_state = pickle.load(f)
#     params = fio_state['params']
#     trace_fiola = np.array(fio_state['trace_fiola'], dtype=np.float32)
#     template = np.array(fio_state['template'], dtype=np.float32)
#     Ab = np.array(fio_state['Ab'], dtype=np.float32)
#     min_mov = fio_state['min_mov']
#     mc_nn_mov = np.array(fio_state['mov_data'], dtype=np.float32)
    
#     fio = FIOLA(params=params)
#     fio.create_pipeline(mc_nn_mov, trace_fiola, template, Ab, min_mov=min_mov)
#     fio.pipeline.saoz.update_q = queue.Queue()
    
#     return fio

# #%%    
# # Reading each tiff frame in buffer, converting to the format that FIOLA wants
# def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
#     image_data = np.zeros(shape, dtype=dtype)
#     for i in range(len(offsets)):
#         buffer.seek(offsets[i])
#         data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
#         image_data[i, ...] = data.reshape(shape[1:])
#     return image_data

# #%%    
# # Using numba to speed up numerical processing
# @njit(parallel=True)
# def process_tiff_data(image_data, offsets, bytecounts, dtype, shape):
#     for i in prange(len(offsets)):
#         data = np.frombuffer(image_data[offsets[i]:offsets[i]+bytecounts[i]], dtype=dtype)
#         image_data[i, ...] = data.reshape(shape[1:])
#     return image_data

# #%%    
# #
# async def memmap_from_buffer(tiff_buffer):
#     buffer = io.BytesIO(tiff_buffer)
#     try:
#         with tifffile.TiffFile(buffer) as tif:
#             tiff_series = tif.series[0]
#             dtype = tiff_series.dtype
#             shape = tiff_series.shape
#             byte_order = tif.byteorder

#             # Accept 1 frame input only
#             shape = (1, *shape)
#             logging.info(f"Shape: {shape}")

#             # Initialize image_data to hold the entire TIFF data
#             image_data = np.zeros(shape, dtype=np.dtype(byte_order + dtype.char))
#             offsets = []
#             bytecounts = []

#             for page in tif.pages:
#                 offsets.extend(page.dataoffsets)
#                 bytecounts.extend(page.databytecounts)

#             image_data = read_tiff_data(buffer, offsets, bytecounts, np.dtype(byte_order + dtype.char), shape)
#             image_data = process_tiff_data(image_data, np.array(offsets), np.array(bytecounts), np.dtype(byte_order + dtype.char), shape)

#         if image_data.size != np.prod(shape):
#             logging.error(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")
#             raise ValueError(f"Data array size {image_data.size} does not match expected size {np.prod(shape)}.")

#         logging.info(f"Successfully memmapped buffer. Shape: {image_data.shape}, dtype: {image_data.dtype}")
#         return image_data
#     except Exception as e:
#         logging.error(f"Error processing TIFF buffer: {e}")
#         raise

# def process_frame_parallel(fio, memmap_image, frame_idx, timestamp, processtimestamp):
#     try:
#         frame_batch = memmap_image.astype(np.float32)
#         fio.fit_online_frame(frame_batch)
#         fio.compute_estimates()
        
#         # np.save(f'./results/fiola_result_ptr_{frame_idx}', fio.estimates)
        
#         end_time = time()
#         total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds

#         message = f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}"
#         return message
#     except Exception as e:
#         logging.error(f"Error processing frame {frame_idx}: {e}")

# async def process_frame_with_buffer(fio, frame_data, frame_idx, timestamp, processtimestamp):
#     try:
#         memmap_image = await memmap_from_buffer(frame_data)
#         loop = asyncio.get_event_loop()
#         message = await loop.run_in_executor(executor, process_frame_parallel, fio, memmap_image, frame_idx, timestamp, processtimestamp)
#         await corelink.send(sender_id, message)
#     except Exception as e:
#         logging.error(f"Failed to process frame with buffer: {e}")


# async def callback(data_bytes, streamID, header):
#     global incoming_frames, fio_index

#     # Extract the header information
#     timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
#     chunk_data = data_bytes[HEADER_SIZE:]

#     frame = incoming_frames[frame_number]
#     frame["timestamp"] = timestamp

#     # Initialize frame entry if receiving the first chunk
#     if frame["received_slices"] == 0:
#         frame["total_slices"] = total_chunks
#         frame["chunks"] = [None] * total_chunks
#         frame["start_time"] = int(time() * 1000)
#     # Store the chunk data in the correct position
#     if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
#         frame["chunks"][chunk_index] = chunk_data
#         frame["received_slices"] += 1
#         logging.info(f"Received slice {chunk_index} for frame {frame_number}")

#         # Check if we have received all chunks for this frame
#         if frame["received_slices"] == total_chunks:
#             # Reconstruct the frame
#             frame_data = b''.join(frame["chunks"])

#             # Process the frame with a rotated FIOLA object concurrently
#             fio = fio_objects[fio_index]
#             fio_index = (fio_index + 1) % FIOLA_POOL_SIZE
#             asyncio.create_task(process_frame_with_buffer(fio, frame_data, frame_number, frame["timestamp"], frame["start_time"]))

#             # Clean up the completed frame entry
#             del incoming_frames[frame_number]
#     else:
#         logging.info(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

# async def update(response, key):
#     logging.info(f'Updating as new sender valid in the workspace: {response}')
#     await subscribe_to_stream(response['receiverID'], response['streamID'])

# async def stale(response, key):
#     logging.info(response)

# async def subscriber(response, key):
#     logging.info(f"subscriber: {response}")

# async def dropped(response, key):    
#     logging.info(f"dropped: {response}")

# async def processing():
#     global fio_objects, sender_id

#     # Read the latest FIOLA state path
#     if os.path.exists(LATEST_FIOLA_STATE_PATH):
#         with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
#             latest_fiola_state_file = f.read().strip()

#         if os.path.exists(latest_fiola_state_file):
#             logging.info(f"Loading FIOLA state from {latest_fiola_state_file}")
#             for _ in range(FIOLA_POOL_SIZE):
#                 fio_objects.append(load_fiola_state(latest_fiola_state_file))
#         else:
#             logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
#             sys.exit(1)
#     else:
#         logging.info("Generating new FIOLA init file")
        
#         #subprocess.run(["python3.8", "./generate_init_result.py"], check=True)
#         terminate_event = asyncio.Event()
#         await receive_then_init.receive_then_init(terminate_event)
#         logging.info("Completed receive_then_init")

#         if os.path.exists(LATEST_FIOLA_STATE_PATH):
#             with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
#                 latest_fiola_state_file = f.read().strip()
#             if os.path.exists(latest_fiola_state_file):
#                 for _ in range(FIOLA_POOL_SIZE):
#                     fio_objects.append(load_fiola_state(latest_fiola_state_file))
#             else:
#                 logging.error(f"Failed to generate the FIOLA state file at {latest_fiola_state_file}")
#                 sys.exit(1)
#         else:
#             logging.error(f"Failed to generate the latest FIOLA state file path at {LATEST_FIOLA_STATE_PATH}")
#             sys.exit(1)

#     await corelink.set_server_callback(update, 'update')
#     await corelink.set_server_callback(stale, 'stale')
#     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#     await corelink.set_data_callback(callback)
    
#     receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
#     logging.info(f"Receiver ID: {receiver_id}")
    
#     logging.info("Start receive process frames")
#     await corelink.keep_open()
    
#     await corelink.set_server_callback(subscriber, 'subscriber')
#     await corelink.set_server_callback(dropped, 'dropped')

#     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#     sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")

#     try:
#         while True:
#             await asyncio.sleep(3600)
#     except KeyboardInterrupt:
#         logging.info('Receiver terminated.')

# if __name__ == "__main__":
#     corelink.run(processing())
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
from fiola.fiola import FIOLA
import io
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from numba import njit, prange

warnings.filterwarnings("ignore", message="no queue or thread to delete")

import corelink
from corelink.resources.control import subscribe_to_stream
import receive_then_init

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

# Process pool executor for CPU-bound tasks
process_executor = ProcessPoolExecutor(max_workers=os.cpu_count())

# Thread pool executor for tasks involving non-pickleable objects
thread_executor = ThreadPoolExecutor(max_workers=12)

FIOLA_POOL_SIZE = 12  # Define the number of FIOLA objects to rotate through
fio_objects = []
fio_index = 0

LATEST_FIOLA_STATE_PATH = '/persistent_storage/latest_fiola_state.pkl'

#%%    
# Loading fiola state from a pickle file 
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

#%%    
# Reading each tiff frame in buffer, converting to the format that FIOLA wants
def read_tiff_data(buffer, offsets, bytecounts, dtype, shape):
    image_data = np.zeros(shape, dtype=dtype)
    for i in range(len(offsets)):
        buffer.seek(offsets[i])
        data = np.frombuffer(buffer.read(bytecounts[i]), dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

#%%    
# Using numba to speed up numerical processing
@njit(parallel=True)
def process_tiff_data(image_data, offsets, bytecounts, dtype, shape):
    for i in prange(len(offsets)):
        data = np.frombuffer(image_data[offsets[i]:offsets[i]+bytecounts[i]], dtype=dtype)
        image_data[i, ...] = data.reshape(shape[1:])
    return image_data

#%%    
#
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
        memmap_image = await memmap_from_buffer(frame_data)
        loop = asyncio.get_event_loop()
        frame_batch = await loop.run_in_executor(process_executor, process_frame_data, memmap_image)
        proc_time = time()
        await loop.run_in_executor(thread_executor, fio.fit_online_frame, frame_batch)
        await loop.run_in_executor(thread_executor, fio.compute_estimates)
        end_time = time()
        total_time = end_time - timestamp / 1000  # Convert timestamp back to seconds
        # print(f'Total time spend on processing frame {frame_idx} is : {end_time-proc_time}')
        message = f"Total time spent on frame {frame_idx} (from capture to finish): {total_time}, total time processing is {end_time-proc_time}"
        print(message)
        await corelink.send(sender_id, message)
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
        logging.info(f"Received slice {chunk_index} for frame {frame_number}")

        # Check if we have received all chunks for this frame
        if frame["received_slices"] == total_chunks:
            # Reconstruct the frame
            frame_data = b''.join(frame["chunks"])

            # Process the frame with a rotated FIOLA object concurrently
            fio = fio_objects[fio_index]
            fio_index = (fio_index + 1) % FIOLA_POOL_SIZE
            asyncio.create_task(process_frame_with_buffer(fio, frame_data, frame_number, frame["timestamp"], frame["start_time"]))

            # Clean up the completed frame entry
            del incoming_frames[frame_number]
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
            for _ in range(FIOLA_POOL_SIZE):
                fio_objects.append(load_fiola_state(latest_fiola_state_file))
        else:
            logging.error(f"Latest FIOLA state file {latest_fiola_state_file} does not exist.")
            sys.exit(1)
    else:
        logging.info("Generating new FIOLA init file")
        
        #subprocess.run(["python3.8", "./generate_init_result.py"], check=True)
        terminate_event = asyncio.Event()
        await receive_then_init.receive_then_init(terminate_event)
        logging.info("Completed receive_then_init")

        if os.path.exists(LATEST_FIOLA_STATE_PATH):
            with open(LATEST_FIOLA_STATE_PATH, 'r') as f:
                latest_fiola_state_file = f.read().strip()
            if os.path.exists(latest_fiola_state_file):
                for _ in range(FIOLA_POOL_SIZE):
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