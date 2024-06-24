// // //               //                     _oo0oo_
// // //               //                    088888880
// // //               //                    88" . "88
// // //               //                    (| -_- |)
// // //               //                     0\ = /0
// // //               //                  ___/'---'\___
// // //               //                .' \\\\|     |// '.
// // //               //               / \\\\|||  :  |||// \\
// // //               //              /_ ||||| -:- |||||- \\
// // //               //             |   | \\\\\\  -  /// |   |
// // //               //             | \_|  ''\---/''  |_/ |
// // //               //             \  .-\__  '-'  __/-.  /
// // //               //           ___'. .'  /--.--\  '. .'___
// // //               //        ."" '<  '.___\_<|>_/___.' >'  "".
// // //               //       | | : '-  \'.;'\ _ /';.'/ - ' : | |
// // //               //       \  \ '_.   \_ __\ /__ _/   .-' /  /
// // //               //   ====='-.____'.___ \_____/___.-'____.-'=====
// // //               //                     '=---='
  
  
// // //               // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// // //               //           佛祖保佑    iii    永不死机
// // //               //           心外无法   (   )   法外无心

const corelink = require('corelink-client');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');
const { Worker } = require('worker_threads');

// Initializing the configurations required to use Corelink service.
const config = {
  ControlPort: 20012,
  ControlIP: 'corelink.hpc.nyu.edu',
  autoReconnect: true,
};

// Configurations for the Corelink receiver.
const username = 'Testuser';
const password = 'Testpassword';
const controlWorkspace = 'Control';
const resultWorkspace = 'Result2';
const protocol = 'ws';
const datatype = 'video';
const CHUNK_SIZE = 8 * 1024; // 8KB chunk size

// Initialization parameters that will be used.
const frames = {};
let currentFrameNumber = 0;

// Ensure the results directory exists
const ensureResultsDirectoryExists = () => {
  const resultsDir = path.join(__dirname, 'results');
  if (!fs.existsSync(resultsDir)) {
    fs.mkdirSync(resultsDir);
  }
};

ensureResultsDirectoryExists();

// Start the Python process once when the script starts
const pythonProcess = spawn('python', ['process_then_return.py']);

pythonProcess.stdout.on('data', (data) => {
  if (data.toString().includes('--PICKLE-START--')) {
    sendProcessedData(Buffer.from(data));
  } else {
    console.log(`Python output: ${data.toString()}`);
  }
});

pythonProcess.stderr.on('data', (data) => {
  console.error(`Python error: ${data.toString()}`);
});

pythonProcess.on('close', (code) => {
  console.log(`Python script finished with code ${code}`);
});

const run = async () => {
  try {
    await corelink.connect({ username, password }, config);
    const receiver = await corelink.createReceiver({
      workspace: controlWorkspace,
      protocol,
      type: datatype,
      echo: true,
      alert: true,
    });

    resultSender = await corelink.createSender({
      workspace: resultWorkspace,
      protocol,
      type: datatype,
      metadata: { name: 'Processed Data' },
    });

    corelink.on('receiver', async (data) => {
      const options = { streamIDs: [data.streamID] };
      await corelink.subscribe(options);
      console.log('Receiver and sender connected, subscribing to data.');
    });

    corelink.on('data', async (streamID, data) => {
      if (data.toString() !== 'FINISHED') {
        await processData(data);
      } else {
        console.log('Received FINISHED message.');
      }
    });

    corelink.on('sender', (data) => {
      console.log('Sender update:', data);
    });
  } catch (err) {
    console.log('Error connecting to Corelink:', err);
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
        sendFileToPython(fullFile);

        // Clean up the frame data
        delete frames[frameNumber];
      }
    } else {
      console.error(`Invalid or duplicate slice index: ${sliceIndex} for frame: ${frameNumber}`);
    }
  } catch (error) {
    console.error(`Error processing data:`, error);
  }
};

const sendFileToPython = async (fileBuffer) => {
  // Write file size to stdin
  const fileSize = Buffer.alloc(4);
  fileSize.writeUInt32LE(fileBuffer.length, 0);
  pythonProcess.stdin.write(fileSize);

  // Write file data to stdin
  pythonProcess.stdin.write(fileBuffer);
  console.log(`Multi-frame TIFF file sent to Python for processing.`);
};

const sendProcessedData = async (fileBuffer) => {
  const totalChunks = Math.ceil(fileBuffer.length / CHUNK_SIZE);
  const frameNumber = currentFrameNumber++; // Ensure unique frame number for each file

  for (let i = 0; i < totalChunks; i++) {
    const chunk = fileBuffer.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE);
    const header = Buffer.alloc(6);
    header.writeUInt16BE(frameNumber, 0);
    header.writeUInt16BE(i, 2);
    header.writeUInt16BE(totalChunks, 4);

    const dataToSend = Buffer.concat([header, chunk]);

    await corelink.send(resultSender, dataToSend);
    console.log(`Chunk ${i} of frame ${frameNumber} sent.`);
  }
};

run();
