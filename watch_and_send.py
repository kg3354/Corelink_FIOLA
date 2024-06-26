# # import asyncio
# # import sys
# # import os
# # import math
# # import struct
# # import time
# # import threading
# # from watchdog.observers import Observer
# # from watchdog.events import FileSystemEventHandler

# # sys.path.append("C:/Users/29712/corelink-client/python/package/Corelink/src")
# # import corelink

# # CHUNK_SIZE = 16 * 1024  # 16 KB chunk size
# # HEADER_SIZE = 6
# # VALIDATION_TIMEOUT = 15  # seconds
# # validConnection = False
# # frame_counter = 0  # Frame counter for sequential frame numbers

# # async def callback(data_bytes, streamID, header):
# #     print(f"Received data with length {len(data_bytes)} : {data_bytes.decode(encoding='UTF-8')}")

# # async def subscriber(response, key):
# #     global validConnection
# #     print("subscriber: ", response)
# #     validConnection = True

# # async def dropped(response, key):
# #     global validConnection
# #     print("dropped", response)
# #     validConnection = False

# # async def check_connection():
# #     global validConnection
# #     while True:
# #         await asyncio.sleep(VALIDATION_TIMEOUT)
# #         if not validConnection:
# #             print("Connection not validated, retrying...")
# #             await corelink.disconnect()
# #             await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# #             validConnection = True
# #             print("Reconnected to Corelink")

# # async def send_file(file_path):
# #     global frame_counter
# #     try:
# #         with open(file_path, 'rb') as file:
# #             file_size = os.path.getsize(file_path)
# #             total_chunks = math.ceil(file_size / CHUNK_SIZE)
# #             frame_number = frame_counter  # Sequential frame number
# #             frame_counter += 1

# #             for chunk_index in range(total_chunks):
# #                 chunk = file.read(CHUNK_SIZE)
# #                 buffer = bytearray(HEADER_SIZE + len(chunk))
# #                 struct.pack_into('>HHH', buffer, 0, frame_number, chunk_index, total_chunks)
# #                 buffer[HEADER_SIZE:] = chunk

# #                 await corelink.send(sender_id, buffer)
# #                 # await asyncio.sleep(0.01)
# #                 print(f'Frame {frame_number}, Chunk {chunk_index}/{total_chunks}, Size {len(buffer)}')

# #             print(f"File {file_path} sent successfully.")

# #     except Exception as e:
# #         print(f"Failed to send file: {e}")

# # async def send_end_message():
# #     end_message = b'FINISHED'
# #     await corelink.send(sender_id, end_message)
# #     print('End message sent.')

# # class FileHandler(FileSystemEventHandler):
# #     def on_created(self, event):
# #         if event.is_directory:
# #             return
# #         if event.src_path.lower().endswith('.tif'):
# #             print(f'New video file detected: {event.src_path}')
# #             # if self.inactive_timeout:
# #             #     self.inactive_timeout.cancel()
# #             asyncio.run(send_file(event.src_path))
# #             # self.inactive_timeout = threading.Timer(1.0, lambda: asyncio.run(send_end_message()))
# #             # self.inactive_timeout.start()

# # async def main():
# #     global validConnection, sender_id
# #     await corelink.set_server_callback(subscriber, 'subscriber')
# #     await corelink.set_server_callback(dropped, 'dropped')

# #     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
# #     sender_id = await corelink.create_sender("Holodeck", "ws", "description1")

# #     # await processing.connect_sender(sender_id)
# #     asyncio.create_task(check_connection())  # Start connection validation in the background

# #     watch_dir = os.getenv('WATCH_DIR', 'C:/Users/29712/Corelink_FIOLA/sample')
    
# #     event_handler = FileHandler()
# #     observer = Observer()
# #     observer.schedule(event_handler, watch_dir, recursive=True)
# #     observer.start()
# #     print(f'Watching for new .tif files in {watch_dir}')

# #     try:
# #         while True:
# #             await asyncio.sleep(1)
# #     except KeyboardInterrupt:
# #         observer.stop()
# #     observer.join()

# # corelink.run(main())
# import asyncio
# import sys
# import os
# import math
# import struct
# import time
# import threading
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# sys.path.append("C:/Users/29712/corelink-client/python/package/Corelink/src")
# import corelink

# CHUNK_SIZE = 16 * 1024  # 16 KB chunk size
# HEADER_SIZE = 6
# VALIDATION_TIMEOUT = 15  # seconds
# RETRY_COUNT = 5  # Number of retries
# RETRY_DELAY = 0.05  # Delay in seconds between retries
# validConnection = False
# frame_counter = 0  # Frame counter for sequential frame numbers

# async def callback(data_bytes, streamID, header):
#     print(f"Received data with length {len(data_bytes)} : {data_bytes.decode(encoding='UTF-8')}")

# async def subscriber(response, key):
#     global validConnection
#     print("subscriber: ", response)
#     validConnection = True

# async def dropped(response, key):
#     global validConnection
#     print("dropped", response)
#     validConnection = False

# async def check_connection():
#     global validConnection
#     while True:
#         await asyncio.sleep(VALIDATION_TIMEOUT)
#         if not validConnection:
#             print("Connection not validated, retrying...")
#             await corelink.disconnect()
#             await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#             validConnection = True
#             print("Reconnected to Corelink")

# async def send_file(file_path):
#     global frame_counter
#     retries = 0
#     while retries < RETRY_COUNT:
#         try:
#             with open(file_path, 'rb') as file:
#                 file_size = os.path.getsize(file_path)
#                 total_chunks = math.ceil(file_size / CHUNK_SIZE)
#                 frame_number = frame_counter  # Sequential frame number
#                 frame_counter += 1

#                 for chunk_index in range(total_chunks):
#                     chunk = file.read(CHUNK_SIZE)
#                     buffer = bytearray(HEADER_SIZE + len(chunk))
#                     struct.pack_into('>HHH', buffer, 0, frame_number, chunk_index, total_chunks)
#                     buffer[HEADER_SIZE:] = chunk

#                     await corelink.send(sender_id, buffer)
#                     await asyncio.sleep(0.01)
#                     print(f'Frame {frame_number}, Chunk {chunk_index}/{total_chunks}, Size {len(buffer)}')

#                 print(f"File {file_path} sent successfully.")
#                 return  # Exit the function on success

#         except PermissionError as e:
#             retries += 1
#             print(f"Failed to send file: {e}. Retrying {retries}/{RETRY_COUNT}...")
#             time.sleep(RETRY_DELAY)

#     print(f"Failed to send file: {file_path} after {RETRY_COUNT} retries due to permission issues.")

# async def send_end_message():
#     end_message = b'FINISHED'
#     await corelink.send(sender_id, end_message)
#     print('End message sent.')

# class FileHandler(FileSystemEventHandler):
     
#     def on_created(self, event):
#         if event.is_directory:
#             return
#         if event.src_path.lower().endswith('.tif'):
#             print(f'New video file detected: {event.src_path}')
          
#             asyncio.run(send_file(event.src_path))
             
# async def main():
#     global validConnection, sender_id
#     await corelink.set_server_callback(subscriber, 'subscriber')
#     await corelink.set_server_callback(dropped, 'dropped')

#     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#     sender_id = await corelink.create_sender("Holodeck", "ws", "description1")

#     asyncio.create_task(check_connection())  # Start connection validation in the background

#     watch_dir = os.getenv('WATCH_DIR', 'C:/Users/29712/Corelink_FIOLA/sample')
    
#     event_handler = FileHandler()
#     observer = Observer()
#     observer.schedule(event_handler, watch_dir, recursive=True)
#     observer.start()
#     print(f'Watching for new .tif files in {watch_dir}')

#     try:
#         while True:
#             await asyncio.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# corelink.run(main())
# import asyncio
# import sys
# import os
# import math
# import struct
# import time
# import threading
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# sys.path.append("C:/Users/29712/corelink-client/python/package/Corelink/src")
# import corelink

# CHUNK_SIZE = 8 * 1024  # 16 KB chunk size
# HEADER_SIZE = 6
# VALIDATION_TIMEOUT = 15  # seconds
# RETRY_COUNT = 5  # Number of retries
# RETRY_DELAY = 0.01  # Delay in seconds between retries
# validConnection = False
# frame_counter = 0  # Frame counter for sequential frame numbers

# async def callback(data_bytes, streamID, header):
#     print(f"Received data with length {len(data_bytes)} : {data_bytes.decode(encoding='UTF-8')}")

# async def subscriber(response, key):
#     global validConnection
#     print("subscriber: ", response)
#     validConnection = True

# async def dropped(response, key):
#     global validConnection
#     print("dropped", response)
#     validConnection = False

# async def check_connection():
#     global validConnection
#     while True:
#         await asyncio.sleep(VALIDATION_TIMEOUT)
#         if not validConnection:
#             print("Connection not validated, retrying...")
#             # await corelink.disconnect()
#             # await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#             # validConnection = True
#             # print("Reconnected to Corelink")

# async def send_file(file_path):
#     global frame_counter
#     retries = 0
#     while retries < RETRY_COUNT:
#         try:
#             with open(file_path, 'rb') as file:
#                 file_size = os.path.getsize(file_path)
#                 total_chunks = math.ceil(file_size / CHUNK_SIZE)
#                 frame_number = frame_counter  # Sequential frame number
#                 frame_counter += 1

#                 for chunk_index in range(total_chunks):
#                     chunk = file.read(CHUNK_SIZE)
#                     buffer = bytearray(HEADER_SIZE + len(chunk))
#                     struct.pack_into('>HHH', buffer, 0, frame_number, chunk_index, total_chunks)
#                     buffer[HEADER_SIZE:] = chunk

#                     try:
#                         await corelink.send(sender_id, buffer)
#                         await asyncio.sleep(0.01)
#                     except Exception as e:
#                         print(f"Failed to send chunk {chunk_index}/{total_chunks} for frame {frame_number}: {e}")
#                         raise

#                     print(f'Frame {frame_number}, Chunk {chunk_index}/{total_chunks}, Size {len(buffer)}')

#                 print(f"File {file_path} sent successfully.")
#                 return  # Exit the function on success

#         except PermissionError as e:
#             retries += 1
#             print(f"Failed to send file: {e}. Retrying {retries}/{RETRY_COUNT}...")
#             time.sleep(RETRY_DELAY)
#         except Exception as e:
#             print(f"Failed to send file due to a WebSocket error: {e}")
#             break

#     print(f"Failed to send file: {file_path} after {RETRY_COUNT} retries due to permission issues or WebSocket errors.")

# async def send_end_message():
#     end_message = b'FINISHED'
#     try:
#         await corelink.send(sender_id, end_message)
#         print('End message sent.')
#     except Exception as e:
#         print(f"Failed to send end message: {e}")

# class FileHandler(FileSystemEventHandler):
 
#     def on_created(self, event):
#         if event.is_directory:
#             return
#         if event.src_path.lower().endswith('.tif'):
#             print(f'New video file detected: {event.src_path}')
             
#             asyncio.run(send_file(event.src_path))
             

# async def main():
#     global validConnection, sender_id
#     await corelink.set_server_callback(subscriber, 'subscriber')
#     await corelink.set_server_callback(dropped, 'dropped')

#     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#     sender_id = await corelink.create_sender("Holodeck", "ws", "description1")

#     asyncio.create_task(check_connection())  # Start connection validation in the background

#     watch_dir = os.getenv('WATCH_DIR', 'C:/Users/29712/Corelink_FIOLA/sample')
    
#     event_handler = FileHandler()
#     observer = Observer()
#     observer.schedule(event_handler, watch_dir, recursive=True)
#     observer.start()
#     print(f'Watching for new .tif files in {watch_dir}')

#     try:
#         while True:
#             await asyncio.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# corelink.run(main())
# import asyncio
# import sys
# import os
# import math
# import struct
# import time
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler

# sys.path.append("C:/Users/29712/corelink-client/python/package/Corelink/src")
# import corelink

# CHUNK_SIZE = 8 * 1024  # 8 KB chunk size
# HEADER_SIZE = 14  # Updated to include timestamp (8 bytes) + frame number (2 bytes) + chunk index (2 bytes) + total chunks (2 bytes)
# VALIDATION_TIMEOUT = 15  # seconds
# RETRY_COUNT = 5  # Number of retries
# RETRY_DELAY = 0.01  # Delay in seconds between retries
# validConnection = False
# frame_counter = 0  # Frame counter for sequential frame numbers

# async def callback(data_bytes, streamID, header):
#     print(f"Received data with length {len(data_bytes)} : {data_bytes.decode(encoding='UTF-8')}")

# async def subscriber(response, key):
#     global validConnection
#     print("subscriber: ", response)
#     validConnection = True

# async def dropped(response, key):
#     global validConnection
#     print("dropped", response)
#     validConnection = False

# async def check_connection():
#     global validConnection
#     while True:
#         await asyncio.sleep(VALIDATION_TIMEOUT)
#         if not validConnection:
#             print("Connection not validated, retrying...")

# async def send_file(file_path):
#     global frame_counter
#     retries = 0
#     while retries < RETRY_COUNT:
#         try:
#             with open(file_path, 'rb') as file:
#                 file_size = os.path.getsize(file_path)
#                 total_chunks = math.ceil(file_size / CHUNK_SIZE)
#                 frame_number = frame_counter  # Sequential frame number
#                 frame_counter += 1

#                 # Get the current timestamp
#                 timestamp = int(time.time() * 1000)  # Convert to milliseconds

#                 for chunk_index in range(total_chunks):
#                     chunk = file.read(CHUNK_SIZE)
#                     buffer = bytearray(HEADER_SIZE + len(chunk))
#                     struct.pack_into('>QHHH', buffer, 0, timestamp, frame_number, chunk_index, total_chunks)
#                     buffer[HEADER_SIZE:] = chunk

#                     try:
#                         await corelink.send(sender_id, buffer)
#                         # await asyncio.sleep(0.01)
#                     except Exception as e:
#                         print(f"Failed to send chunk {chunk_index}/{total_chunks} for frame {frame_number}: {e}")
#                         raise

#                     print(f'Frame {frame_number}, Chunk {chunk_index}/{total_chunks}, Size {len(buffer)}')

#                 print(f"File {file_path} sent successfully.")
#                 return  # Exit the function on success

#         except PermissionError as e:
#             retries += 1
#             print(f"Failed to send file: {e}. Retrying {retries}/{RETRY_COUNT}...")
#             time.sleep(RETRY_DELAY)
#         except Exception as e:
#             print(f"Failed to send file due to a WebSocket error: {e}")
#             break

#     print(f"Failed to send file: {file_path} after {RETRY_COUNT} retries due to permission issues or WebSocket errors.")

# async def send_end_message():
#     end_message = b'FINISHED'
#     try:
#         await corelink.send(sender_id, end_message)
#         print('End message sent.')
#     except Exception as e:
#         print(f"Failed to send end message: {e}")

# class FileHandler(FileSystemEventHandler):
#     def on_created(self, event):
#         if event.is_directory:
#             return
#         if event.src_path.lower().endswith('.tif'):
#             print(f'New video file detected: {event.src_path}')
#             asyncio.run(send_file(event.src_path))

# async def main():
#     global validConnection, sender_id
#     await corelink.set_server_callback(subscriber, 'subscriber')
#     await corelink.set_server_callback(dropped, 'dropped')

#     await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
#     sender_id = await corelink.create_sender("FentonCtl", "ws", "description1")

#     asyncio.create_task(check_connection())  # Start connection validation in the background

#     watch_dir = os.getenv('WATCH_DIR', 'C:/Users/29712/Corelink_FIOLA/sample')
    
#     event_handler = FileHandler()
#     observer = Observer()
#     observer.schedule(event_handler, watch_dir, recursive=True)
#     observer.start()
#     print(f'Watching for new .tif files in {watch_dir}')

#     try:
#         while True:
#             await asyncio.sleep(1)
#     except KeyboardInterrupt:
#         observer.stop()
#     observer.join()

# corelink.run(main())
import asyncio
import sys
import os
import math
import struct
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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

async def send_file(file_path):
    global frame_counter
    retries = 0
    while retries < RETRY_COUNT:
        try:
            with open(file_path, 'rb') as file:
                file_size = os.path.getsize(file_path)
                total_chunks = math.ceil(file_size / CHUNK_SIZE)
                frame_number = frame_counter  # Sequential frame number
                frame_counter += 1

                # Get the current timestamp
                timestamp = int(time.time() * 1000)  # Convert to milliseconds

                for chunk_index in range(total_chunks):
                    chunk = file.read(CHUNK_SIZE)
                    buffer = bytearray(HEADER_SIZE + len(chunk))
                    struct.pack_into('>QHHH', buffer, 0, timestamp, frame_number, chunk_index, total_chunks)
                    buffer[HEADER_SIZE:] = chunk

                    try:
                        await corelink.send(sender_id, buffer)
                        # await asyncio.sleep(0.01)
                    except Exception as e:
                        print(f"Failed to send chunk {chunk_index}/{total_chunks} for frame {frame_number}: {e}")
                        raise

                    print(f'Frame {frame_number}, Chunk {chunk_index}/{total_chunks}, Size {len(buffer)}')

                print(f"File {file_path} sent successfully.")
                return  # Exit the function on success

        except PermissionError as e:
            retries += 1
            print(f"Failed to send file: {e}. Retrying {retries}/{RETRY_COUNT}...")
            time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"Failed to send file due to a WebSocket error: {e}")
            break

    print(f"Failed to send file: {file_path} after {RETRY_COUNT} retries due to permission issues or WebSocket errors.")

async def send_end_message():
    end_message = b'FINISHED'
    try:
        await corelink.send(sender_id, end_message)
        print('End message sent.')
    except Exception as e:
        print(f"Failed to send end message: {e}")

class FileHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop

    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.lower().endswith('.tif'):
            print(f'New video file detected: {event.src_path}')
            self.loop.create_task(send_file(event.src_path))

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
    print(f'Watching for new .tif files in {watch_dir}')

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

corelink.run(main())
