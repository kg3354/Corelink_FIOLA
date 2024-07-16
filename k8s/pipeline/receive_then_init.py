
# import os
# import sys
# import struct
# import asyncio
# import logging
# from collections import defaultdict
# from time import time, strftime, localtime
# import tifffile
# import numpy as np
# import io
# import corelink
# from corelink.resources.control import subscribe_to_stream
# from generate_init_result import caiman_process

# HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
# INACTIVITY_TIMEOUT = 10  # seconds

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
#     "last_update_time": time()
# })

# # List to keep track of sessions
# sessions = []
# current_session_frames = []
# current_session_start_time = None
# last_update_time = time()

# # Ensure the 'data' directory exists
# os.makedirs('data', exist_ok=True)

# async def save_session_tiff(frames, session_start_time):
#     tiff_filename = f'./data/session_{strftime("%Y%m%d_%H%M%S", localtime(session_start_time))}.tif'
#     with tifffile.TiffWriter(tiff_filename, bigtiff=True) as tif_writer:
#         for frame in frames:
#             tif_writer.write(np.array(tifffile.imread(io.BytesIO(frame))), contiguous=True)
#     print(f"Session saved as {tiff_filename}")
#     return tiff_filename

# async def check_inactivity():
#     global current_session_frames, current_session_start_time, last_update_time
#     while True:
#         await asyncio.sleep(1)
#         current_time = time()
#         if current_session_start_time and (current_time - last_update_time > INACTIVITY_TIMEOUT):
#             print(f"Session inactive for {INACTIVITY_TIMEOUT} seconds. Marking session as complete.")
#             tiff_filename = await save_session_tiff(current_session_frames, current_session_start_time)
#             await asyncio.get_event_loop().run_in_executor(None, caiman_process, tiff_filename)
#             await corelink.close()  # Close the corelink session
#             print("Corelink session closed due to inactivity.")
#             sys.exit()  # End the script

# async def callback(data_bytes, streamID, header):
#     global incoming_frames, current_session_frames, current_session_start_time, last_update_time

#     # Extract the header information
#     timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QIII', data_bytes[:HEADER_SIZE])
#     chunk_data = data_bytes[HEADER_SIZE:]

#     frame = incoming_frames[frame_number]
#     frame["timestamp"] = timestamp

#     # Update the last update time
#     last_update_time = time()

#     # Initialize session start time
#     if not current_session_start_time:
#         current_session_start_time = time()

#     # Initialize frame entry if receiving the first chunk
#     if frame["received_slices"] == 0:
#         frame["total_slices"] = total_chunks
#         frame["chunks"] = [None] * total_chunks

#     # Store the chunk data in the correct position
#     if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
#         frame["chunks"][chunk_index] = chunk_data
#         frame["received_slices"] += 1
#         print(f"Received slice {chunk_index} for frame {frame_number}")

#         # Check if we have received all chunks for this frame
#         if frame["received_slices"] == total_chunks:
#             # Reconstruct the frame
#             frame_data = b''.join(frame["chunks"])
#             current_session_frames.append(frame_data)
#             print(f"Frame {frame_number} completed and added to session.")
			
#             # Clean up the completed frame entry
#             del incoming_frames[frame_number]
#     else:
#         print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

# async def update(response, key):
#     print(f'Updating as new sender valid in the workspace: {response}')
#     await subscribe_to_stream(response['receiverID'], response['streamID'])

# async def stale(response, key):
#     print(response)

# async def receive_then_init():
#     async def main():
#         asyncio.create_task(check_inactivity())  # Start inactivity check

#         await corelink.set_data_callback(callback)
#         await corelink.set_server_callback(update, 'update')
#         await corelink.set_server_callback(stale, 'stale')
		
#         await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
		
#         receiver_id = await corelink.create_receiver("FentonInit", "ws", alert=True, echo=True)
		
#         print(f'Receiver ID: {receiver_id}')
#         print("Start receiving")
#         await corelink.keep_open()
		
#         try:
#             while True:
#                 await asyncio.sleep(3600)
#         except KeyboardInterrupt:
#             print('Receiver terminated.')
#             await corelink.close()

#         print('Finished')

#     corelink.run(main)

# # # Start the receive process
# # receive_then_init()
import os
import sys
import struct
import asyncio
import logging
from collections import defaultdict
from time import time, strftime, localtime
import tifffile
import numpy as np
import io
import corelink
from corelink.resources.control import subscribe_to_stream
from generate_init_result import caiman_process

HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
INACTIVITY_TIMEOUT = 10  # seconds

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
	"last_update_time": time()
})

# List to keep track of sessions
sessions = []
current_session_frames = []
current_session_start_time = None
last_update_time = time()

# Ensure the 'data' directory exists
os.makedirs('data', exist_ok=True)

async def save_session_tiff(frames, session_start_time):
	tiff_filename = f'./data/session_{strftime("%Y%m%d_%H%M%S", localtime(session_start_time))}.tif'
	with tifffile.TiffWriter(tiff_filename, bigtiff=True) as tif_writer:
		for frame in frames:
			tif_writer.write(np.array(tifffile.imread(io.BytesIO(frame))), contiguous=True)
	print(f"Session saved as {tiff_filename}")
	return tiff_filename

async def check_inactivity():
	global current_session_frames, current_session_start_time, last_update_time
	while True:
		await asyncio.sleep(1)
		current_time = time()
		if current_session_start_time and (current_time - last_update_time > INACTIVITY_TIMEOUT):
			print(f"Session inactive for {INACTIVITY_TIMEOUT} seconds. Marking session as complete.")
			tiff_filename = await save_session_tiff(current_session_frames, current_session_start_time)
			await asyncio.get_event_loop().run_in_executor(None, caiman_process, tiff_filename)
			await corelink.close()  # Close the corelink session
			print("Corelink session closed due to inactivity.")
			sys.exit()  # End the script

async def callback(data_bytes, streamID, header):
	global incoming_frames, current_session_frames, current_session_start_time, last_update_time

	# Extract the header information
	timestamp, frame_number, chunk_index, total_chunks = struct.unpack('>QHHH', data_bytes[:HEADER_SIZE])
	chunk_data = data_bytes[HEADER_SIZE:]

	frame = incoming_frames[frame_number]
	frame["timestamp"] = timestamp

	# Update the last update time
	last_update_time = time()

	# Initialize session start time
	if not current_session_start_time:
		current_session_start_time = time()

	# Initialize frame entry if receiving the first chunk
	if frame["received_slices"] == 0:
		frame["total_slices"] = total_chunks
		frame["chunks"] = [None] * total_chunks

	# Store the chunk data in the correct position
	if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
		frame["chunks"][chunk_index] = chunk_data
		frame["received_slices"] += 1
		print(f"Received slice {chunk_index} for frame {frame_number}")

		# Check if we have received all chunks for this frame
		if frame["received_slices"] == total_chunks:
			# Reconstruct the frame
			frame_data = b''.join(frame["chunks"])
			current_session_frames.append(frame_data)
			print(f"Frame {frame_number} completed and added to session.")
			
			# Clean up the completed frame entry
			del incoming_frames[frame_number]
	else:
		print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

async def update(response, key):
	print(f'Updating as new sender valid in the workspace: {response}')
	await subscribe_to_stream(response['receiverID'], response['streamID'])

async def stale(response, key):
	print(response)

async def receive_then_init():
	async def main():
		asyncio.create_task(check_inactivity())  # Start inactivity check

		await corelink.set_data_callback(callback)
		await corelink.set_server_callback(update, 'update')
		await corelink.set_server_callback(stale, 'stale')
		
		await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
		
		receiver_id = await corelink.create_receiver("FentonRaw", "ws", alert=True, echo=True)
		
		print(f'Receiver ID: {receiver_id}')
		print("Start receiving initilization frames")
		await corelink.keep_open()
		
		try:
			while True:
				await asyncio.sleep(3600)
		except KeyboardInterrupt:
			print('Receiver terminated.')
			await corelink.close()

		print('Finished')

	corelink.run(main())

