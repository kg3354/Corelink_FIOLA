import sys
import os
import struct
import asyncio
from collections import defaultdict
import cv2
import tifffile
import numpy as np
from time import time

sys.path.append("/Users/guobuzai/Desktop/corelink/corelink-client/python/package/Corelink/src")
import corelink


CHUNK_SIZE = 4 * 1024  # 4 KB chunk size
HEADER_SIZE = 6

# Dictionary to hold the incoming chunks for each frame
incoming_frames = defaultdict(lambda: {
    "total_slices": 0,
    "received_slices": 0,
    "chunks": [],
    "start_time": time()
})

# Ensure the 'avifiles' directory exists
os.makedirs('avifiles', exist_ok=True)

async def callback(data_bytes, streamID, header):
    global incoming_frames

    # Extract the header information
    frame_number, chunk_index, total_chunks = struct.unpack('>HHH', data_bytes[:HEADER_SIZE])
    chunk_data = data_bytes[HEADER_SIZE:]

    frame = incoming_frames[frame_number]

    # Initialize frame entry if receiving the first chunk
    if frame["received_slices"] == 0:
        frame["total_slices"] = total_chunks
        frame["chunks"] = [None] * total_chunks
        frame["start_time"] = time()
    # Store the chunk data in the correct position
    if chunk_index < total_chunks and frame["chunks"][chunk_index] is None:
        frame["chunks"][chunk_index] = chunk_data
        frame["received_slices"] += 1
        print(f"Received slice {chunk_index} for frame {frame_number}")

        # Check if we have received all chunks for this frame
        if frame["received_slices"] == total_chunks:
            # Reconstruct the frame
            frame_data = b''.join(frame["chunks"])

            # Save the reconstructed frame to a file
            frame_path = f'avifiles/frame_{frame_number}.avi'
            with open(frame_path, 'wb') as frame_file:
                frame_file.write(frame_data)

            print(f'Saved frame {frame_number} to {frame_path}')

            print(f"Total time spent on frame {frame_number} is {time() - frame['start_time']}")
            # Clean up the completed frame entry
            del incoming_frames[frame_number]
    else:
        print(f"Invalid or duplicate slice index: {chunk_index} for frame: {frame_number}")

async def update(response, key):
    print(f'Updating as new sender valid in the workspace: {response}')
    # await control.subscribe_to_stream(response['receiverID'], response['streamID'])

async def stale(response, key):

    print(response)


    # Call the function to generate a TIFF file
    generate_tiff_from_avi_files('avifiles', 'output.tif')

def get_avi_files(directory):
    """Get a list of AVI files in the specified directory."""
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.avi')]

def extract_frames_from_avi(avi_file):
    """Extract frames from an AVI file."""
    cap = cv2.VideoCapture(avi_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale (if needed)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()
    return frames

def generate_tiff_from_avi_files(input_directory, output_tif_file):
    """Generate a TIFF file from AVI files in the specified directory."""
    avi_files = get_avi_files(input_directory)
    
    with tifffile.TiffWriter(output_tif_file, bigtiff=True) as tif_writer:
        for avi_file in avi_files:
            frames = extract_frames_from_avi(avi_file)
            for frame in frames:
                tif_writer.write(frame, contiguous=True, metadata={'axes': 'YX'})

    # Make the TIFF file memory-mappable
    memmap_tiff(output_tif_file)
    print(f"Generated memory-mappable TIFF file {output_tif_file} from AVI files in {input_directory}")

def memmap_tiff(tiff_file):
    """Ensure the TIFF file is memory-mappable."""
    with tifffile.TiffWriter(tiff_file, append=True) as tif:
        pass
async def main():
    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    await corelink.set_data_callback(callback)
    await corelink.set_server_callback(update, 'update')
    await corelink.set_server_callback(stale, 'stale')

    receiver_id = await corelink.create_receiver("Holodeck", "ws", alert=True, echo=True)
    print(receiver_id)
    
    print("Start receiving")
    await corelink.keep_open()
    
    await asyncio.sleep(200)

    print('Finished')

corelink.run(main())
