import asyncio
import sys
import os
import math
import struct
import time
from io import BytesIO
import cv2
import tifffile
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import aiofiles

sys.path.append("C:/Users/29712/corelink-client/python/package/Corelink/src")
import corelink

CHUNK_SIZE = 8 * 1024  # 8 KB chunk size
HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
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
                struct.pack_into('>QHHH', buffer, 0, timestamp, frame_counter, chunk_index, total_chunks)
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

class FileHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop

    async def process_file(self, file_path):
        global frame_counter
        if file_path.lower().endswith('.tif'):
            print(f'New TIFF file detected: {file_path}')
            async with aiofiles.open(file_path, 'rb') as f:
                file_data = await f.read()
            await send_file(file_data, frame_counter)
           
        elif file_path.lower().endswith('.avi'):
            print(f'New AVI file detected: {file_path}')
            file_data = await convert_avi_to_tiff_in_memory(file_path)
            await send_file(file_data, frame_counter)

        frame_counter += 1  # Increment the frame counter after processing the file

    def on_created(self, event):
        if event.is_directory:
            return
        self.loop.create_task(self.process_file(event.src_path))

async def main():
    global validConnection, sender_id
    await corelink.set_server_callback(subscriber, 'subscriber')
    await corelink.set_server_callback(dropped, 'dropped')

    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")

    asyncio.create_task(check_connection())  # Start connection validation in the background

    watch_dir = os.getenv('WATCH_DIR', 'C:/Users/29712/Corelink_FIOLA/curr')
    
    loop = asyncio.get_event_loop()
    event_handler = FileHandler(loop)
    observer = Observer()
    observer.schedule(event_handler, watch_dir, recursive=True)
    observer.start()
    print(f'Watching for new .tif or .avi files in {watch_dir}')

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

corelink.run(main())
