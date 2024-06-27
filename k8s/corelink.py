"""
Python client for Corelink.\n
corelink.hsrn.nyu.edu
"""

import websockets
import ssl
import asyncio
import json
from .resources import reqs, processing, variables
from .resources.control import *
from .resources.variables import log
import logging

def run(function):
    """Runs the user function in an event loop, and catches keyboard interrupt."""
    try:
        variables.loop = asyncio.get_event_loop()
        variables.loop.run_until_complete(function)
        print("Closing down Corelink...")
    except KeyboardInterrupt:
        print("Closing from keyboard interrupt.")
    finally:
        if variables.is_open:
            variables.loop.run_until_complete(_exit())

async def keep_open():
    """Called by user to ensure that the program stays open while awaiting data."""
    asyncio.create_task(processing.maintain())

async def close():
    """Called by user to end a receiver session (used after calling `keep_open()`)."""
    variables.shut_down = True

async def connect(username: str, password: str, host: str, port: str):
    """Connects to server and authenticates.
    :param username: username registered with Corelink server
    :param password: password associated with username
    :param host: host address to connect to
    :param port: host port to connect to
    :return: token
    """
    variables.user = username
    variables.password = password
    variables.host = host
    variables.port = port
    # variables.protocol = protocol
    variables.connection = await websockets.connect(f"wss://{variables.host}:{variables.port}", ssl=ssl.SSLContext(ssl.PROTOCOL_TLS))
    variables.receiver_task = asyncio.create_task(processing.ws_control_receiver(variables.connection))
    await auth()

async def set_data_callback(callback):
    """User should pass their callback function into this.
    The function is expected to take:
        param1 message: bytes,
        param2 streamID: int,
        param3 header: dict (sometimes empty)"""
    variables.user_cb = callback

async def set_server_callback(callback, key: str):
    """Sets a callback function for server messages of the given key:
        options: 'update', 'subscriber', 'stale', 'dropped'
    callback should expect dict message with the server message (details in docs),
                        and str key listing what the message type is."""
    variables.server_callbacks[key] = callback

async def active_streams() -> list:
    """Returns a list of current streamIDs."""
    return list(variables.streams) + list(variables.receiver)

async def create_sender(workspace, protocol: str = "tcp", streamID="", data_type='', metadata='', sender="", ip="", port="") -> int:
    """Requests a sender from the server and opens the connection
    :return: streamID used to send
    """
    protocol = protocol.lower()
    if protocol not in variables.valid_proto:
        raise ValueError("protocol: protocol must be ws, udp, or tcp.")
    request = {
        "function": "sender",
        "workspace": workspace,
        "senderID": streamID,
        "proto": protocol,
        "IP": ip,
        "port": port,
        "alert": False,
        "type": data_type,
        "meta": metadata,
        "from": sender,
        "token": variables.token
    }
    sender = reqs.retrieve(await reqs.request_func(request), ret=True)
    variables.streams[sender['streamID']] = sender
    variables.streams[sender['streamID']]['protocol'] = request['proto']

    await processing.connect_sender(sender['streamID'])
    return sender['streamID']

async def create_receiver(workspace, protocol, data_type="", metadata="", alert: bool = False, echo: bool = False, receiver_id="", stream_ids=[], ip=None, port=0, subscribe=True) -> int:
    """Requests a receiver from the server and opens the connection.
    :return: streamID used to receive
    """
    protocol = protocol.lower()
    if protocol not in variables.valid_proto:
        raise ValueError("protocol: protocol must be ws, udp, or tcp.")
    if ip is None:
        ip = variables.user_ip
    request = {
        "function": "receiver",
        "workspace": workspace,
        "receiverID": receiver_id,
        "streamIDs": stream_ids,
        "proto": protocol,
        "type": data_type,
        "alert": alert,
        "echo": echo,
        "IP": ip,
        "port": port,
        "meta": metadata,
        "subscribe": subscribe,
        "token": variables.token
    }
    receiver = reqs.retrieve(await reqs.request_func(request), ret=True)
    log("receiver: " + str(receiver))
    variables.receiver[receiver['streamID']] = receiver
    await processing.connect_receiver(receiver['streamID'])
    return receiver['streamID']

async def send(streamID, data, user_header: dict = None):
    """Sends data to streamID's stream (user should first call connect_sender(streamID)).
    data should be either str or bytes.
    """
    stream = variables.streams[streamID] # for convenience
    user_h = json.dumps(user_header) if user_header else ""
    header = [0, 0, 0, 0, 0, 0, 0, 0]
    header = bytearray(header)
    head = memoryview(header)
    head[0:2] = int.to_bytes(len(user_h), 2, 'little')
    head[2:4] = int.to_bytes(len(data), 2, 'little')
    head[4:6] = int.to_bytes(int(streamID), 2, 'little')
    log(header)
    try:
        if isinstance(data, str):
            message = bytes(header) + user_h.encode() + data.encode()
        else:
            message = bytes(header) + user_h.encode() + data
    except Exception as e:
        logging.error(f"An error occurred during Corelink.send: {e}")
    log(message)
    if stream['protocol'] == 'ws':
        asyncio.create_task(stream['connection'].send(message))
        log("[ws] data sent")
    else:
        stream['connection'].send(message)

async def disconnect_senders(streamIDs: list):
    """Disconnects given list of streamIDs from server and removes streams."""
    for ID in streamIDs:
        log("disconnecting " + str(ID))
        if variables.streams[ID]['protocol'] == 'ws':
            await variables.streams[ID]['connection'].close()
        else:
            variables.streams[ID]['connection'].close()
    await disconnect_streams(stream_ids=streamIDs)

async def disconnect_receivers(streamIDs: list):
    """Disconnects given list of streamIDs from server and removes streams."""
    for ID in streamIDs:
        log("disconnecting receiver " + str(ID))
        receiver = variables.receiver[ID]
        if receiver['proto'] == 'ws':
            await receiver['connection'].close()
            variables.receiver_task.cancel()
        else:
            receiver['connection'][0].close()
    await disconnect_streams(stream_ids=streamIDs)

async def _exit():
    """Disconnects all open streams and closes connection.
    Automatically called by run()."""
    if variables.streams:
        await disconnect_senders(list(variables.streams))
    if variables.receiver:
        await disconnect_receivers(list(variables.receiver))
    # expire()
    await variables.connection.close()  # closes websockets control stream
    variables.is_open = False
    print("Closed.")
