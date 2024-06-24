
const fs = require('fs');
const chokidar = require('chokidar');
const path = require('path');
const corelink = require('corelink-client');

// Initializing the configurations required to use Corelink service.
const config = {
  ControlPort: 20012,
  ControlIP: 'corelink.hpc.nyu.edu',
  autoReconnect: true,
};

// Configurations for the Corelink sender.
const username = 'Testuser';
const password = 'Testpassword';
const controlWorkspace = 'Control';
const resultWorkspace = 'Result2';
const protocol = 'ws';
const datatype = 'video';
const CHUNK_SIZE = 16 * 1024; // 16KB chunk size

// Should be set to false for production.
corelink.debug = true;

// Initialization parameters that will be used.
let receiverActive = true;
let sender;
let resultReceiver;
let currentFrameNumber = 0;
let inactiveTimeout;
const frames = {};
let timer;

/**
 * Async function to send a file in chunks. Each chunk starts with a 2-byte frame number,
 * followed by a 2-byte chunk index and 2-byte total chunk number,
 * then followed by the chunk data itself.
 * @param {String} filePath The file path of the file to be sent via Corelink.
 */
async function sendFile(filePath) {
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, (err, fileBuffer) => {
      if (err) return reject(err);

      const totalChunks = Math.ceil(fileBuffer.length / CHUNK_SIZE);
      const frameNumber = currentFrameNumber++; // Ensure unique frame number for each file

      for (let i = 0; i < totalChunks; i++) {
        const chunk = fileBuffer.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
        const header = Buffer.alloc(6);
        header.writeUInt16BE(frameNumber, 0);
        header.writeUInt16BE(i, 2);
        header.writeUInt16BE(totalChunks, 4);

        const dataToSend = Buffer.concat([header, chunk]);

        if (receiverActive) {
          corelink.send(sender, dataToSend);
          console.log(`Chunk ${i} of frame ${frameNumber} sent.`);
        }
      }
      resolve();
    });
  });
}

/**
 * Sends an end message indicating no more frames will be sent.
 */
async function sendEndMessage() {
  if (receiverActive) {
    const endMessage = Buffer.from('FINISHED');
    await corelink.send(sender, endMessage);
    console.log('End message sent.');
  }
}

/**
 * Initializes Corelink sender and chokidar watcher, when new video files are written to disk,
 * the chokidar watcher will retrieve that file path and calls sendFile function to send.
 */
const run = async () => {
  try {
    await corelink.connect({ username, password }, config);

    sender = await corelink.createSender({
      workspace: controlWorkspace,
      protocol,
      type: datatype,
      metadata: { name: 'AVI File Sender' },
      echo: true,
      alert: true,
    });

    resultReceiver = await corelink.createReceiver({
      workspace: resultWorkspace,
      protocol,
      type: datatype,
      echo: true,
      alert: true,
    });

    corelink.on('stale', (data) => {
      console.log('Stream went stale:', data);
      receiverActive = false;
    });

    corelink.on('dropped', (data) => {
      console.log('Stream dropped:', data);
      receiverActive = false;
    });

    corelink.on('close', () => {
      console.log('Control connection closed');
      receiverActive = false;
    });

    corelink.on('receiver', async (data) => {
      if (data.workspace === resultWorkspace) {
        const options = { streamIDs: [data.streamID] };
        await corelink.subscribe(options);
      }
    });

    corelink.on('data', async (streamID, data, header) => {
      await processData(data);
    });

    const watchDir = process.env.WATCH_DIR || 'C:/Users/Research/Desktop/Corelink-FIOLA';

    const watcher = chokidar.watch(watchDir, {
      persistent: true,
      ignoreInitial: true,
      followSymlinks: false,
      depth: 5,
      awaitWriteFinish: {
        stabilityThreshold: 2000,
        pollInterval: 100,
      },
      usePolling: true,
    });

    watcher.on('add', async (filePath) => {
      if (path.extname(filePath).toLowerCase() === '.tif') {
        console.log(`New video file detected: ${filePath}`);
        clearTimeout(inactiveTimeout);
        timer = Date.now()
        try {
          await sendFile(filePath);
        } catch (err) {
          console.error('Failed to send file:', err);
        }

        inactiveTimeout = setTimeout(async () => {
          await sendEndMessage();
        }, 1000);
      }
    });

    console.log(`Watching for new .tif files in ${watchDir}`);
  } catch (err) {
    console.error('Error:', err);
  }
};

const processData = async (data) => {
  try {
    const frameNumber = data.readUInt16BE(0);
    const sliceIndex = data.readUInt16BE(2);
    const totalSlices = data.readUInt16BE(4);
    const content = data.slice(6);

    if (!frames[frameNumber]) {
      frames[frameNumber] = {
        totalSlices,
        receivedSlices: 0,
        chunks: new Array(totalSlices).fill(null),
        startTime: performance.now()
      };
    }

    const frame = frames[frameNumber];
    if (sliceIndex < totalSlices && frame.chunks[sliceIndex] === null) {
      frame.chunks[sliceIndex] = content;
      frame.receivedSlices++;
      console.log(`Received slice ${sliceIndex} for frame ${frameNumber}`);

      if (frame.receivedSlices === totalSlices) {
        const endTime = performance.now();
        const timeTaken = endTime - frame.startTime;
        console.log(`Frame ${frameNumber} reassembled in ${timeTaken.toFixed(2)} ms.`);

        const fullFile = Buffer.concat(frame.chunks);
        const timestamp = Date.now();
        const resultFilePath = path.join(__dirname, 'results', `fiola_result_${timestamp}.pkl`);
        await fs.promises.writeFile(resultFilePath, fullFile);
        console.log(`Result saved to ${resultFilePath}`);

        // Clean up the frame data
        delete frames[frameNumber];
        console.log(`Spent ${(Date.now()-timer).toFixed(2)} ms in total on ${frameNumber}`);
      }
    } else {
      console.error(`Invalid or duplicate slice index: ${sliceIndex} for frame: ${frameNumber}`);
    }
  } catch (error) {
    console.error(`Error processing data:`, error);
  }
};

run();
