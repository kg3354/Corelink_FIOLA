import asyncio
import sys
import os
import math
import struct
import time
sys.path.append("/Users/guobuzai/Desktop/corelink/corelink-client/python/package/Corelink/src")
import corelink
from corelink import *
from corelink import processing
from time import time

CHUNK_SIZE = 4 * 1024  # 4 KB chunk size
HEADER_SIZE = 6
VALIDATION_TIMEOUT = 15  # seconds
validConnection = False

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
    await asyncio.sleep(VALIDATION_TIMEOUT)
    if not validConnection:
        print("Connection not validated, retrying...")

async def main():
    await corelink.set_server_callback(subscriber, 'subscriber')
    await corelink.set_server_callback(dropped, 'dropped')

    await corelink.connect("Testuser", "Testpassword", "corelink.hpc.nyu.edu", 20012)
    sender_id = await corelink.create_sender("Holodeck", "ws", "description1")

    # res = await corelink.list_functions()
    # print(res)
    # res = await corelink.list_server_functions()
    # print(res)

    ranger = 123    
    timeout = 0.01
    chunkCounter = 0
    await processing.connect_sender(sender_id)

    asyncio.create_task(check_connection())  # Start connection validation in the background

    while True:
        if validConnection:
            startTime = time()
            for frame_number in range(ranger):
                filePath = f'/Users/guobuzai/Projects/kafkaresult/frame_{frame_number}.avi'
                print(filePath)
                with open(filePath, 'rb') as file:
                    fileSize = os.path.getsize(filePath)
                    totalChunks = math.ceil(fileSize / CHUNK_SIZE)

                    for chunkIndex in range(totalChunks):
                        chunk = file.read(CHUNK_SIZE)
                        buffer = bytearray(HEADER_SIZE + len(chunk))
                        struct.pack_into('>HHH', buffer, 0, frame_number, chunkIndex, totalChunks)
                        buffer[HEADER_SIZE:] = chunk

                        await corelink.send(sender_id, buffer)
                        await asyncio.sleep(timeout)
                        print(f'Frame {frame_number}, Chunk {chunkIndex}/{totalChunks}, Size {len(buffer)}')
                        chunkCounter += 1
            print("Total chunks sent:", chunkCounter)
            print(f'Total time spent is: {time() - startTime}')
            break
        else:
            print("Waiting for valid connection...")
            await asyncio.sleep(5)

corelink.run(main())
