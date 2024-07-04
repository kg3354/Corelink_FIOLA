import asyncio
import os
import math
import struct
import time
from io import BytesIO
import cv2
import tifffile
import aiofiles
import corelink

CHUNK_SIZE = 8* 1024  # 8 KB chunk size
HEADER_SIZE = 20  # Updated to include timestamp (8 bytes) + frame number (4 bytes) + chunk index (4 bytes) + total chunks (4 bytes)
VALIDATION_TIMEOUT = 15  # seconds
RETRY_COUNT = 5  # Number of retries
RETRY_DELAY = 0.01  # Delay in seconds between retries
validConnection = False
frame_counter = 0  # Frame counter for sequential frame numbers

async def callback(data_bytes, streamID, header):
    print(f"Received data with length {len(data_bytes)} : {data_bytes.decode(encoding='UTF-8')}")

async def subscriber(response, key):
    global validConnection
    print("subscriber: ", response)
    validConnection = True

async def dropped(response, key):
    global validConnection
    print("dropped", response)
    validConnection = False

async def check_connection():
    global validConnection
    while True:
        await asyncio.sleep(VALIDATION_TIMEOUT)
        if not validConnection:
            print("Connection not validated, retrying...")

async def send_file(file_data, frame_counter):
    retries = 0
    while retries < RETRY_COUNT:
        try:
            file_size = len(file_data)
            total_chunks = math.ceil(file_size / CHUNK_SIZE)

            # Get the current timestamp
            timestamp = int(time.time() * 1000)  # Convert to milliseconds

            for chunk_index in range(total_chunks):
                chunk = file_data[chunk_index * CHUNK_SIZE:(chunk_index + 1) * CHUNK_SIZE]
                buffer = bytearray(HEADER_SIZE + len(chunk))
                struct.pack_into('>QIII', buffer, 0, timestamp, frame_counter, chunk_index, total_chunks)
                buffer[HEADER_SIZE:] = chunk

                try:
                    await corelink.send(sender_id, buffer)
                except Exception as e:
                    print(f"Failed to send chunk {chunk_index}/{total_chunks} for frame {frame_counter}: {e}")
                    raise

                print(f'Frame {frame_counter}, Chunk {chunk_index}/{total_chunks}, Size {len(buffer)}')

            print(f"File sent successfully.")
            return  # Exit the function on success

        except PermissionError as e:
            retries += 1
            print(f"Failed to send file: {e}. Retrying {retries}/{RETRY_COUNT}...")
            await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Failed to send file due to a WebSocket error: {e}")
            break

    print(f"Failed to send file after {RETRY_COUNT} retries due to permission issues or WebSocket errors.")

async def send_end_message():
    end_message = b'FINISHED'
    try:
        await corelink.send(sender_id, end_message)
        print('End message sent.')
    except Exception as e:
        print(f"Failed to send end message: {e}")

async def convert_avi_to_tiff_in_memory(avi_file):
    """Convert AVI file to TIFF file in memory asynchronously."""
    cap = cv2.VideoCapture(avi_file)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    cap.release()

    tiff_buffer = BytesIO()
    with tifffile.TiffWriter(tiff_buffer, bigtiff=True) as tif_writer:
        for frame in frames:
            tif_writer.write(frame, contiguous=True)
    tiff_buffer.seek(0)
    return tiff_buffer.read()

async def process_and_send_file(file_path):
    global frame_counter
    try:
        if file_path.lower().endswith('.tif'):
            print(f'Processing TIFF file: {file_path}')
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read()
            await send_file(file_data, frame_counter)

        elif file_path.lower().endswith('.avi'):
            print(f'Processing AVI file: {file_path}')
            file_data = await convert_avi_to_tiff_in_memory(file_path)
            await send_file(file_data, frame_counter)
        asyncio.sleep(0.1)
        frame_counter += 1  # Increment the frame counter after processing the file

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

async def main():
    global validConnection, sender_id
    await corelink.set_server_callback(subscriber, 'subscriber')
    await corelink.set_server_callback(dropped, 'dropped')

    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    sender_id = await corelink.create_sender("FentonInit2", "ws", "description1")

    asyncio.create_task(check_connection())  # Start connection validation in the background

    watch_dir = os.getenv('WATCH_DIR', '../../curr')
    
    for file_name in os.listdir(watch_dir):
        file_path = os.path.join(watch_dir, file_name)
        if os.path.isfile(file_path):
            await process_and_send_file(file_path)

    await send_end_message()

corelink.run(main())
