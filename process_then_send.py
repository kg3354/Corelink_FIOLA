# # # import sys
# # # import os
# # # import numpy as np
# # # import pickle
# # # import logging
# # # from time import time
# # # from fiola.fiola import FIOLA
# # # import warnings
# # # import socket

# # # # Suppress specific warnings
# # # warnings.filterwarnings("ignore", message="no queue or thread to delete")

# # # # Configure logging
# # # logging.basicConfig(
# # #     format="%(asctime)s [%(levelname)s] %(message)s",
# # #     level=logging.INFO,
# # #     handlers=[
# # #         logging.StreamHandler(sys.stdout)
# # #     ]
# # # )

# # # CHUNK_SIZE = 1024  # Define your chunk size

# # # def load_fiola_state(filepath):
# # #     """Load the FIOLA state from a pickle file."""
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
# # #     return fio

# # # def send_data_in_chunks(data, frame_idx, client_socket):
# # #     total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
# # #     for i in range(total_chunks):
# # #         chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
# # #         frame_number_buffer = frame_idx.to_bytes(2, byteorder='big')
# # #         chunk_info_buffer = bytes([i, total_chunks])
# # #         data_to_send = frame_number_buffer + chunk_info_buffer + chunk
# # #         client_socket.sendall(data_to_send)
# # #     client_socket.sendall(f'FINISHED Frame {frame_idx}'.encode())

# # # def process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, client_socket):
# # #     start_time = time()
    
# # #     # Convert frame data to numpy array
# # #     frame_array = np.frombuffer(frame_data, dtype=np.uint8)  # Adjust dtype if necessary
    
# # #     # Calculate the frame shape
# # #     num_pixels = frame_array.size // 1
# # #     frame_shape = (1, num_pixels)
    
# # #     # Reshape the frame array
# # #     frame_array = frame_array.reshape(frame_shape)
    
# # #     for idx in range(num_frames_init, num_frames_total, batch):
# # #         frame_batch = frame_array[idx:idx + batch].astype(np.float32)
# # #         fio.fit_online_frame(frame_batch)
# # #         online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx - batch:idx]
# # #         online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx - batch - fio.params.retrieve['lag']:idx - fio.params.retrieve['lag']]

# # #     fio.compute_estimates()
    
# # #     # Send result to Node.js server in chunks
# # #     result_data = fio.estimates.tobytes()
# # #     send_data_in_chunks(result_data, frame_idx, client_socket)
    
# # #     end_time = time()
# # #     total_time = end_time - start_time
# # #     logging.info(f"Total time spent processing frame {frame_idx}: {total_time:.6f} seconds")

# # # def main():
# # #     fio = load_fiola_state('fiola_state_msCam.pkl')
# # #     frame_idx = 0

# # #     # Initialize trace arrays based on FIOLA parameters
# # #     num_frames_init = 3  # Number of frames used for initialization
# # #     num_frames_total = 3  # Estimated total number of frames for processing
# # #     batch = 1  # Number of frames processed at the same time using GPU
# # #     online_trace = np.zeros((fio.Ab.shape[-1], num_frames_total - num_frames_init), dtype=np.float32)
# # #     online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], num_frames_total - num_frames_init), dtype=np.float32)

# # #     # Connect to the Node.js server
# # #     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# # #     client_socket.connect(('127.0.0.1', 3000))

# # #     while True:
# # #         # Read frame size from stdin
# # #         frame_size_data = sys.stdin.buffer.read(4)
# # #         if not frame_size_data:
# # #             break
# # #         frame_size = int.from_bytes(frame_size_data, byteorder='little')
        
# # #         # Read frame data from stdin
# # #         frame_data = sys.stdin.buffer.read(frame_size)
# # #         if not frame_data or len(frame_data) != frame_size:
# # #             break
        
# # #         frame_idx += 1
# # #         process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, client_socket)

# # #     client_socket.close()

# # # if __name__ == "__main__":
# # #     main()
# # import sys
# # import os
# # import numpy as np
# # import pickle
# # import logging
# # from time import time
# # from fiola.fiola import FIOLA
# # import warnings
# # import subprocess
# # import io

# # # Suppress specific warnings
# # warnings.filterwarnings("ignore", message="no queue or thread to delete")

# # # Configure logging
# # logging.basicConfig(
# #     format="%(asctime)s [%(levelname)s] %(message)s",
# #     level=logging.INFO,
# #     handlers=[
# #         logging.StreamHandler(sys.stdout)
# #     ]
# # )

# # CHUNK_SIZE = 1024  # Define your chunk size

# # def load_fiola_state(filepath):
# #     """Load the FIOLA state from a pickle file."""
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
# #     return fio

# # def send_data_via_subprocess(data, frame_idx, node_process):
# #     total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
# #     for i in range(total_chunks):
# #         chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
# #         frame_number_buffer = frame_idx.to_bytes(2, byteorder='big')
# #         chunk_info_buffer = i.to_bytes(2, byteorder='big') + total_chunks.to_bytes(2, byteorder='big')
# #         data_to_send = frame_number_buffer + chunk_info_buffer + chunk
# #         node_process.stdin.write(data_to_send)
# #     node_process.stdin.flush()

# # def process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, node_process):
# #     start_time = time()
    
# #     # Convert frame data to numpy array
# #     frame_array = np.frombuffer(frame_data, dtype=np.uint8)  # Adjust dtype if necessary
    
# #     # Calculate the frame shape
# #     num_pixels = frame_array.size // 1
# #     frame_shape = (1, num_pixels)
    
# #     # Reshape the frame array
# #     frame_array = frame_array.reshape(frame_shape)
    
# #     for idx in range(num_frames_init, num_frames_total, batch):
# #         frame_batch = frame_array[idx:idx + batch].astype(np.float32)
# #         fio.fit_online_frame(frame_batch)
# #         online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx - batch:idx]
# #         online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx - batch - fio.params.retrieve['lag']:idx - fio.params.retrieve['lag']]

# #     fio.compute_estimates()
    
# #     # Serialize result using pickle
# #     result_data = pickle.dumps(fio.estimates)
    
# #     # Send the serialized data to the Node.js process via stdin
# #     send_data_via_subprocess(result_data, frame_idx, node_process)
    
# #     end_time = time()
# #     total_time = end_time - start_time
# #     logging.info(f"Total time spent processing frame {frame_idx}: {total_time:.6f} seconds")

# # def main():
# #     fio = load_fiola_state('fiola_state_msCam.pkl')
# #     frame_idx = 0

# #     # Initialize trace arrays based on FIOLA parameters
# #     num_frames_init = 3  # Number of frames used for initialization
# #     num_frames_total = 3  # Estimated total number of frames for processing
# #     batch = 1  # Number of frames processed at the same time using GPU
# #     online_trace = np.zeros((fio.Ab.shape[-1], num_frames_total - num_frames_init), dtype=np.float32)
# #     online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], num_frames_total - num_frames_init), dtype=np.float32)

# #     # Start the Node.js process
# #     node_process = subprocess.Popen(['node', 'send_result.js'], stdin=subprocess.PIPE)

# #     while True:
# #         # Read frame size from stdin
# #         frame_size_data = sys.stdin.buffer.read(4)
# #         if not frame_size_data:
# #             break
# #         frame_size = int.from_bytes(frame_size_data, byteorder='little')
        
# #         # Read frame data from stdin
# #         frame_data = sys.stdin.buffer.read(frame_size)
# #         if not frame_data or len(frame_data) != frame_size:
# #             break
        
# #         frame_idx += 1
# #         process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, node_process)

# #     node_process.stdin.close()
# #     node_process.wait()

# # if __name__ == "__main__":
# #     main()
# import sys
# import os
# import numpy as np
# import pickle
# import logging
# from time import time
# from fiola.fiola import FIOLA
# import warnings
# import subprocess
# import io

# # Suppress specific warnings
# warnings.filterwarnings("ignore", message="no queue or thread to delete")

# # Configure logging
# logging.basicConfig(
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     level=logging.INFO,
#     handlers=[
#         logging.StreamHandler(sys.stdout)
#     ]
# )

# CHUNK_SIZE = 1024  # Define your chunk size

# def load_fiola_state(filepath):
#     """Load the FIOLA state from a pickle file."""
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
#     return fio

# def send_data_via_subprocess(data, frame_idx, node_process):
#     total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
#     for i in range(total_chunks):
#         chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
#         frame_number_buffer = frame_idx.to_bytes(2, byteorder='big')
#         chunk_info_buffer = i.to_bytes(2, byteorder='big') + total_chunks.to_bytes(2, byteorder='big')
#         data_to_send = frame_number_buffer + chunk_info_buffer + chunk
#         try:
#             node_process.stdin.write(data_to_send)
#         except BrokenPipeError:
#             logging.error("Broken pipe error occurred when sending data to Node.js process.")
#             break
#     node_process.stdin.flush()

# def process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, node_process):
#     start_time = time()
    
#     # Convert frame data to numpy array
#     frame_array = np.frombuffer(frame_data, dtype=np.uint8)  # Adjust dtype if necessary
    
#     # Calculate the frame shape
#     num_pixels = frame_array.size // 1
#     frame_shape = (1, num_pixels)
    
#     # Reshape the frame array
#     frame_array = frame_array.reshape(frame_shape)
    
#     for idx in range(num_frames_init, num_frames_total, batch):
#         frame_batch = frame_array[idx:idx + batch].astype(np.float32)
#         fio.fit_online_frame(frame_batch)
#         online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx - batch:idx]
#         online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx - batch - fio.params.retrieve['lag']:idx - fio.params.retrieve['lag']]

#     fio.compute_estimates()
    
#     # Serialize result using pickle
#     result_data = pickle.dumps(fio.estimates)
    
#     # Send the serialized data to the Node.js process via stdin
#     send_data_via_subprocess(result_data, frame_idx, node_process)
    
#     end_time = time()
#     total_time = end_time - start_time
#     logging.info(f"Total time spent processing frame {frame_idx}: {total_time:.6f} seconds")

# def main():
#     fio = load_fiola_state('fiola_state_msCam.pkl')
#     frame_idx = 0

#     # Initialize trace arrays based on FIOLA parameters
#     num_frames_init = 3  # Number of frames used for initialization
#     num_frames_total = 3  # Estimated total number of frames for processing
#     batch = 1  # Number of frames processed at the same time using GPU
#     online_trace = np.zeros((fio.Ab.shape[-1], num_frames_total - num_frames_init), dtype=np.float32)
#     online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], num_frames_total - num_frames_init), dtype=np.float32)

#     # Start the Node.js process
#     node_process = subprocess.Popen(['node', 'send_result.js'], stdin=subprocess.PIPE)

#     while True:
#         # Read frame size from stdin
#         frame_size_data = sys.stdin.buffer.read(4)
#         if not frame_size_data:
#             break
#         frame_size = int.from_bytes(frame_size_data, byteorder='little')
        
#         # Read frame data from stdin
#         frame_data = sys.stdin.buffer.read(frame_size)
#         if not frame_data or len(frame_data) != frame_size:
#             break
        
#         frame_idx += 1
#         process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, node_process)

#     node_process.stdin.close()
#     node_process.wait()

# if __name__ == "__main__":
#     main()
import sys
import os
import numpy as np
import pickle
import logging
from time import time
from fiola.fiola import FIOLA
import warnings
import asyncio
import subprocess

# Suppress specific warnings
warnings.filterwarnings("ignore", message="no queue or thread to delete")

# Configure logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

CHUNK_SIZE = 1024  # Define your chunk size

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

async def send_data_via_subprocess(data, frame_idx, node_process):
    total_chunks = (len(data) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(total_chunks):
        chunk = data[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE]
        frame_number_buffer = frame_idx.to_bytes(2, byteorder='big')
        chunk_info_buffer = i.to_bytes(2, byteorder='big') + total_chunks.to_bytes(2, byteorder='big')
        data_to_send = frame_number_buffer + chunk_info_buffer + chunk
        try:
            node_process.stdin.write(data_to_send)
            await node_process.stdin.drain()
        except BrokenPipeError:
            logging.error("Broken pipe error occurred when sending data to Node.js process.")
            break

async def process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, node_process):
    start_time = time()
    
    # Convert frame data to numpy array
    frame_array = np.frombuffer(frame_data, dtype=np.uint8)  # Adjust dtype if necessary
    
    # Calculate the frame shape
    num_pixels = frame_array.size // 1
    frame_shape = (1, num_pixels)
    
    # Reshape the frame array
    frame_array = frame_array.reshape(frame_shape)
    
    for idx in range(num_frames_init, num_frames_total, batch):
        frame_batch = frame_array[idx:idx + batch].astype(np.float32)
        fio.fit_online_frame(frame_batch)
        online_trace[:, idx:idx + batch] = fio.pipeline.saoz.trace[:, idx - batch:idx]
        online_trace_deconvolved[:, idx:idx + batch] = fio.pipeline.saoz.trace_deconvolved[:, idx - batch - fio.params.retrieve['lag']:idx - fio.params.retrieve['lag']]

    fio.compute_estimates()
    
    # Serialize result using pickle
    result_data = pickle.dumps(fio.estimates)
    
    # Send the serialized data to the Node.js process via stdin
    await send_data_via_subprocess(result_data, frame_idx, node_process)
    
    end_time = time()
    total_time = end_time - start_time
    logging.info(f"Total time spent processing frame {frame_idx}: {total_time:.6f} seconds")

async def main():
    fio = load_fiola_state('fiola_state_msCam.pkl')
    frame_idx = 0

    # Initialize trace arrays based on FIOLA parameters
    num_frames_init = 3  # Number of frames used for initialization
    num_frames_total = 3  # Estimated total number of frames for processing
    batch = 1  # Number of frames processed at the same time using GPU
    online_trace = np.zeros((fio.Ab.shape[-1], num_frames_total - num_frames_init), dtype=np.float32)
    online_trace_deconvolved = np.zeros((fio.Ab.shape[-1] - fio.params.hals['nb'], num_frames_total - num_frames_init), dtype=np.float32)

    # Start the Node.js process
    node_process = await asyncio.create_subprocess_exec(
        'node', 'send_result.js',
        stdin=subprocess.PIPE
    )

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
        await process_frame(fio, frame_data, frame_idx, num_frames_init, num_frames_total, batch, online_trace, online_trace_deconvolved, node_process)

    node_process.stdin.close()
    await node_process.wait()

if __name__ == "__main__":
    asyncio.run(main())
