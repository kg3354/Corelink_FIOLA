// const fs = require('fs');
// const path = require('path');
// const config = {
//   ControlPort: 20012,
//   ControlIP: 'corelink.hpc.nyu.edu',
//   autoReconnect: false,
// };
// const username = 'Testuser';
// const password = 'Testpassword';
// const corelink = require('corelink-client');

// const workspace = 'fiola-result';
// const protocol = 'ws';
// const datatype = 'distance';

// const run = async () => {
//   if (await corelink.connect({ username, password }, config).catch((err) => { console.log(err); })) {
//     await corelink.createReceiver({
//       workspace, protocol, type: datatype, echo: true, alert: true,
//     }).catch((err) => { console.log(err); });

//     corelink.on('receiver', async (data) => {
//       const options = { streamIDs: [data.streamID] };
//       await corelink.subscribe(options);
//     });

//     let frameBuffer = Buffer.alloc(0);
//     let currentFrame = 0;
    
//     // Ensure the result directory exists
//     const resultDir = path.join(__dirname, 'result');
//     if (!fs.existsSync(resultDir)) {
//       fs.mkdirSync(resultDir);
//     }

//     corelink.on('data', (streamID, data, header) => {
//       const dataStr = data.toString();
//       if (dataStr.startsWith('FINISHED Frame ')) {
//         // Save the frame buffer to a file in the result directory
//         const framePath = path.join(resultDir, `frame_${currentFrame}.npy`);
//         fs.writeFileSync(framePath, frameBuffer);
//         console.log(`Frame ${currentFrame} saved to ${framePath}`);

//         // Reset for the next frame
//         frameBuffer = Buffer.alloc(0);
//         currentFrame++;
//       } else {
//         // Append chunk data to the frame buffer
//         frameBuffer = Buffer.concat([frameBuffer, data]);
//       }
//     });
//   }
// };

// run();
const fs = require('fs');
const path = require('path');
const config = {
  ControlPort: 20012,
  ControlIP: 'corelink.hpc.nyu.edu',
  autoReconnect: true,
};
const username = 'Testuser';
const password = 'Testpassword';
const corelink = require('corelink-client');

const workspace = 'fiola-result';
const protocol = 'ws';
const datatype = 'distance';

const run = async () => {
  if (await corelink.connect({ username, password }, config).catch((err) => { console.log(err); })) {
    await corelink.createReceiver({
      workspace, protocol, type: datatype, echo: true, alert: true,
    }).catch((err) => { console.log(err); });

    corelink.on('receiver', async (data) => {
      const options = { streamIDs: [data.streamID] };
      await corelink.subscribe(options);
    });

    const resultDir = path.join(__dirname, 'result');
    if (!fs.existsSync(resultDir)) {
      fs.mkdirSync(resultDir);
    }

    let fileParts = {};
    let frameBuffers = {};

    corelink.on('data', (streamID, data) => {
      const frameIndex = (data.readUInt8(0) << 8) | data.readUInt8(1);
      const chunkIndex = data.readUInt16BE(2);
      const totalChunks = data.readUInt16BE(4);
      const content = data.slice(6);

      if (!fileParts[frameIndex]) {
        fileParts[frameIndex] = new Array(totalChunks).fill(null);
      }
      fileParts[frameIndex][chunkIndex] = content;

      if (fileParts[frameIndex].every(part => part !== null)) {
        const fullFile = Buffer.concat(fileParts[frameIndex]);
        const framePath = path.join(resultDir, `frame_${frameIndex}.pkl`);
        fs.writeFileSync(framePath, fullFile);
        console.log(`Frame ${frameIndex} saved to ${framePath}`);
        delete fileParts[frameIndex];
      }
    });
  }
};

run();
