// // // // // const { spawn } = require('child_process');
// // // // // const fs = require('fs');
// // // // // const path = require('path');

// // // // // const pythonProcess = spawn('python', ['process_then_return.py']);

// // // // // const tiffFolder = './CaImAn/example_movies/frame_sample';
// // // // // const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// // // // // let frameCounter = 0;
// // // // // let buffer = Buffer.alloc(0);

// // // // // pythonProcess.stdout.on('data', (data) => {
// // // // //   buffer = Buffer.concat([buffer, data]);

// // // // //   while (buffer.length >= 4) {
// // // // //     const dataSize = buffer.readUInt32LE(0);
// // // // //     if (buffer.length >= 4 + dataSize) {
// // // // //       const frameData = buffer.slice(4, 4 + dataSize);
// // // // //       saveProcessedData(frameCounter, frameData);

// // // // //       buffer = buffer.slice(4 + dataSize);
// // // // //       frameCounter++;
// // // // //     } else {
// // // // //       break;
// // // // //     }
// // // // //   }
// // // // // });

// // // // // pythonProcess.stderr.on('data', (data) => {
// // // // //   console.error(`Python error: ${data.toString()}`);
// // // // // });

// // // // // pythonProcess.on('close', (code) => {
// // // // //   console.log(`Python script finished with code ${code}`);
// // // // // });

// // // // // const processTiffFiles = () => {
// // // // //   for (const tiffFile of tiffFiles) {
// // // // //     const filePath = path.join(tiffFolder, tiffFile);
// // // // //     const frameBuffer = fs.readFileSync(filePath);
// // // // //     sendFrameToPython(frameBuffer);
// // // // //   }
// // // // // };

// // // // // const sendFrameToPython = (frameBuffer) => {
// // // // //   const frameSize = Buffer.alloc(4);
// // // // //   frameSize.writeUInt32LE(frameBuffer.length, 0);
// // // // //   pythonProcess.stdin.write(frameSize);
// // // // //   pythonProcess.stdin.write(frameBuffer);

// // // // //   console.log(`Frame sent to Python for processing.`);
// // // // // };

// // // // // const saveProcessedData = (frameCounter, frameData) => {
// // // // //   const resultFolder = './processed_frames';
// // // // //   if (!fs.existsSync(resultFolder)) {
// // // // //     fs.mkdirSync(resultFolder);
// // // // //   }

// // // // //   const resultFilePath = path.join(resultFolder, `fiola_result_${frameCounter}.npy`);
// // // // //   fs.writeFileSync(resultFilePath, frameData);
// // // // //   console.log(`Processed data for frame ${frameCounter} saved to ${resultFilePath}.`);
// // // // // };

// // // // // processTiffFiles();
// // // // const { spawn } = require('child_process');
// // // // const fs = require('fs');
// // // // const path = require('path');

// // // // const pythonProcess = spawn('python', ['process_frame.py']);

// // // // const tiffFolder = './trash';
// // // // const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// // // // let frameCounter = 0;
// // // // let buffer = Buffer.alloc(0);

// // // // pythonProcess.stdout.on('data', (data) => {
// // // //   buffer = Buffer.concat([buffer, data]);

// // // //   while (buffer.length >= 4) {
// // // //     const dataSize = buffer.readUInt32LE(0);
// // // //     if (buffer.length >= 4 + dataSize) {
// // // //       const frameData = buffer.slice(4, 4 + dataSize);
// // // //       saveProcessedData(frameCounter, frameData);

// // // //       buffer = buffer.slice(4 + dataSize);
// // // //       frameCounter++;
// // // //     } else {
// // // //       break;
// // // //     }
// // // //   }
// // // // });

// // // // pythonProcess.stderr.on('data', (data) => {
// // // //   console.error(`Python error: ${data.toString()}`);
// // // // });

// // // // pythonProcess.on('close', (code) => {
// // // //   console.log(`Python script finished with code ${code}`);
// // // // });

// // // // const processTiffFiles = () => {
// // // //   for (const tiffFile of tiffFiles) {
// // // //     const filePath = path.join(tiffFolder, tiffFile);
// // // //     const frameBuffer = fs.readFileSync(filePath);
// // // //     sendFrameToPython(frameBuffer);
// // // //   }
// // // // };

// // // // const sendFrameToPython = (frameBuffer) => {
// // // //   const frameSize = Buffer.alloc(4);
// // // //   frameSize.writeUInt32LE(frameBuffer.length, 0);
// // // //   pythonProcess.stdin.write(frameSize);
// // // //   pythonProcess.stdin.write(frameBuffer);

// // // //   console.log(`Frame sent to Python for processing.`);
// // // // };

// // // // const saveProcessedData = (frameCounter, frameData) => {
// // // //   const resultFolder = './processed_frames';
// // // //   if (!fs.existsSync(resultFolder)) {
// // // //     fs.mkdirSync(resultFolder);
// // // //   }

// // // //   const resultFilePath = path.join(resultFolder, `fiola_result_${frameCounter}.pkl`);
// // // //   fs.writeFileSync(resultFilePath, frameData);
// // // //   console.log(`Processed data for frame ${frameCounter} saved to ${resultFilePath}.`);
// // // // };

// // // // processTiffFiles();
// // // const { spawn } = require('child_process');
// // // const fs = require('fs');
// // // const path = require('path');

// // // // Start the Python process once when the script starts
// // // const pythonProcess = spawn('python', ['process_then_return.py']);

// // // // Configuration for the sequence of TIFF files
// // // const tiffFolder = './trash';
// // // const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// // // let frameCounter = 0;

// // // // Event listeners for Python process output and errors
// // // pythonProcess.stdout.on('data', (data) => {
// // //   if (data.length < 4) return; // Ignore incomplete data

// // //   const estimateSize = data.readUInt32LE(0);
// // //   const estimateData = data.slice(4, 4 + estimateSize);

// // //   const estimateFilePath = path.join(__dirname, `estimate_frame_${frameCounter}.pkl`);
// // //   fs.writeFileSync(estimateFilePath, estimateData);
// // //   console.log(`Frame ${frameCounter} estimates saved to ${estimateFilePath}`);
// // // });

// // // pythonProcess.stderr.on('data', (data) => {
// // //   console.error(`Python error: ${data.toString()}`);
// // // });

// // // pythonProcess.on('close', (code) => {
// // //   console.log(`Python script finished with code ${code}`);
// // // });

// // // // Function to process each TIFF file
// // // const processTiffFiles = () => {
// // //   for (const tiffFile of tiffFiles) {
// // //     const filePath = path.join(tiffFolder, tiffFile);

// // //     const stats = fs.statSync(filePath);
// // //     const fileSizeInBytes = stats.size;
// // //     console.log(`File: ${tiffFile}, Size: ${fileSizeInBytes} bytes`);

// // //     const frameBuffer = fs.readFileSync(filePath);
// // //     frameCounter++;
// // //     sendFrameToPython(frameBuffer, frameCounter, fileSizeInBytes);
// // //   }
// // // };

// // // const sendFrameToPython = (frameBuffer, frameCounter, fileSizeInBytes) => {
// // //   const frameSize = Buffer.alloc(4);
// // //   frameSize.writeUInt32LE(frameBuffer.length, 0);
// // //   pythonProcess.stdin.write(frameSize);

// // //   pythonProcess.stdin.write(frameBuffer);

// // //   console.log(`Frame ${frameCounter} sent to Python for processing. Buffer size: ${frameBuffer.length} bytes, File size: ${fileSizeInBytes} bytes.`);
// // // };

// // // // Start processing the TIFF files
// // // processTiffFiles();
// // const { spawn } = require('child_process');
// // const fs = require('fs');
// // const path = require('path');

// // // Start the Python process once when the script starts
// // const pythonProcess = spawn('python', ['process_frame.py']);

// // // Configuration for the sequence of TIFF files
// // const tiffFolder = './trash'; // Update with the path to your TIFF files
// // const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// // let frameCounter = 0;

// // // Event listeners for Python process output and errors
// // pythonProcess.stdout.on('data', (data) => {
// //   console.log(`Python output received`);
  
// //   // Save the received data to a file
// //   const resultFilePath = path.join(tiffFolder, `fiola_result_${frameCounter}.pkl`);
// //   fs.writeFileSync(resultFilePath, data);
// //   console.log(`Result saved to ${resultFilePath}`);
// // });

// // pythonProcess.stderr.on('data', (data) => {
// //   console.error(`Python error: ${data.toString()}`);
// // });

// // pythonProcess.on('close', (code) => {
// //   console.log(`Python script finished with code ${code}`);
// // });

// // // Function to process each TIFF file
// // const processTiffFiles = () => {
// //   for (const tiffFile of tiffFiles) {
// //     const filePath = path.join(tiffFolder, tiffFile);

// //     // Get the file size
// //     const stats = fs.statSync(filePath);
// //     const fileSizeInBytes = stats.size;
// //     console.log(`File: ${tiffFile}, Size: ${fileSizeInBytes} bytes`);

// //     // Read the file into a buffer
// //     const frameBuffer = fs.readFileSync(filePath);
// //     frameCounter++;
// //     sendFrameToPython(frameBuffer, frameCounter, fileSizeInBytes);
// //   }
// // };

// // const sendFrameToPython = (frameBuffer, frameCounter, fileSizeInBytes) => {
// //   // Write frame size to stdin
// //   const frameSize = Buffer.alloc(4);
// //   frameSize.writeUInt32LE(frameBuffer.length, 0);
// //   pythonProcess.stdin.write(frameSize);

// //   // Write frame data to stdin
// //   pythonProcess.stdin.write(frameBuffer);

// //   console.log(`Frame ${frameCounter} sent to Python for processing. Buffer size: ${frameBuffer.length} bytes, File size: ${fileSizeInBytes} bytes.`);
// // };

// // // Start processing the TIFF files
// // processTiffFiles();
// const { spawn } = require('child_process');
// const fs = require('fs');
// const path = require('path');

// // Start the Python process once when the script starts
// const pythonProcess = spawn('python', ['process_then_return.py']);

// // Configuration for the sequence of TIFF files
// const tiffFolder = './trash'; // Update with the path to your TIFF files
// const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// let frameCounter = 0;
// let pickleBuffer = Buffer.alloc(0);
// let isReceivingPickle = false;

// // Event listeners for Python process output and errors
// pythonProcess.stdout.on('data', (data) => {
//   const dataString = data.toString();
  
//   if (isReceivingPickle) {
//     const endIndex = dataString.indexOf('--PICKLE-END--');
//     if (endIndex !== -1) {
//       pickleBuffer = Buffer.concat([pickleBuffer, data.slice(0, endIndex)]);
//       const resultFilePath = path.join(tiffFolder, `fiola_result_${frameCounter}.pkl`);
//       fs.writeFileSync(resultFilePath, pickleBuffer);
//       console.log(`Result saved to ${resultFilePath}`);
//       pickleBuffer = Buffer.alloc(0);
//       isReceivingPickle = false;
//     } else {
//       pickleBuffer = Buffer.concat([pickleBuffer, data]);
//     }
//   } else {
//     const startIndex = dataString.indexOf('--PICKLE-START--');
//     if (startIndex !== -1) {
//       isReceivingPickle = true;
//       pickleBuffer = Buffer.concat([pickleBuffer, data.slice(startIndex + 15)]);
//     } else {
//       console.log(`Python output: ${dataString}`);
//     }
//   }
// });

// pythonProcess.stderr.on('data', (data) => {
//   console.error(`Python error: ${data.toString()}`);
// });

// pythonProcess.on('close', (code) => {
//   console.log(`Python script finished with code ${code}`);
// });

// // Function to process each TIFF file
// const processTiffFiles = () => {
//   for (const tiffFile of tiffFiles) {
//     const filePath = path.join(tiffFolder, tiffFile);

//     // Get the file size
//     const stats = fs.statSync(filePath);
//     const fileSizeInBytes = stats.size;
//     console.log(`File: ${tiffFile}, Size: ${fileSizeInBytes} bytes`);

//     // Read the file into a buffer
//     const frameBuffer = fs.readFileSync(filePath);
//     frameCounter++;
//     sendFrameToPython(frameBuffer, frameCounter, fileSizeInBytes);
//   }
// };

// const sendFrameToPython = (frameBuffer, frameCounter, fileSizeInBytes) => {
//   // Write frame size to stdin
//   const frameSize = Buffer.alloc(4);
//   frameSize.writeUInt32LE(frameBuffer.length, 0);
//   pythonProcess.stdin.write(frameSize);

//   // Write frame data to stdin
//   pythonProcess.stdin.write(frameBuffer);

//   console.log(`Frame ${frameCounter} sent to Python for processing. Buffer size: ${frameBuffer.length} bytes, File size: ${fileSizeInBytes} bytes.`);
// };

// // Start processing the TIFF files
// processTiffFiles();
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Start the Python process once when the script starts
const pythonProcess = spawn('python', ['process_then_return.py']);

// Configuration for the sequence of TIFF files
const tiffFolder = './trash'; // Update with the path to your TIFF files
const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

let frameCounter = 0;
let pickleBuffer = Buffer.alloc(0);
let isReceivingPickle = false;

// Event listeners for Python process output and errors
pythonProcess.stdout.on('data', (data) => {
  const dataString = data.toString();
  
  if (isReceivingPickle) {
    const endIndex = dataString.indexOf('--PICKLE-END--');
    if (endIndex !== -1) {
      pickleBuffer = Buffer.concat([pickleBuffer, data.slice(0, endIndex)]);
      const timestamp = Date.now();
      const resultFilePath = path.join(tiffFolder, `fiola_result_${frameCounter}_${timestamp}.pkl`);
      fs.writeFileSync(resultFilePath, pickleBuffer);
      console.log(`Result saved to ${resultFilePath}`);
      pickleBuffer = Buffer.alloc(0);
      isReceivingPickle = false;
      frameCounter++;
    } else {
      pickleBuffer = Buffer.concat([pickleBuffer, data]);
    }
  } else {
    const startIndex = dataString.indexOf('--PICKLE-START--');
    if (startIndex !== -1) {
      isReceivingPickle = true;
      pickleBuffer = Buffer.concat([pickleBuffer, data.slice(startIndex + 15)]);
    } else {
      console.log(`Python output: ${dataString}`);
    }
  }
});

pythonProcess.stderr.on('data', (data) => {
  console.error(`Python error: ${data.toString()}`);
});

pythonProcess.on('close', (code) => {
  console.log(`Python script finished with code ${code}`);
});

// Function to process each TIFF file
const processTiffFiles = () => {
  for (const tiffFile of tiffFiles) {
    const filePath = path.join(tiffFolder, tiffFile);

    // Get the file size
    const stats = fs.statSync(filePath);
    const fileSizeInBytes = stats.size;
    console.log(`File: ${tiffFile}, Size: ${fileSizeInBytes} bytes`);

    // Read the file into a buffer
    const frameBuffer = fs.readFileSync(filePath);
    sendFrameToPython(frameBuffer, frameCounter, fileSizeInBytes);
  }
};

const sendFrameToPython = (frameBuffer, frameCounter, fileSizeInBytes) => {
  // Write frame size to stdin
  const frameSize = Buffer.alloc(4);
  frameSize.writeUInt32LE(frameBuffer.length, 0);
  pythonProcess.stdin.write(frameSize);

  // Write frame data to stdin
  pythonProcess.stdin.write(frameBuffer);

  console.log(`Frame ${frameCounter} sent to Python for processing. Buffer size: ${frameBuffer.length} bytes, File size: ${fileSizeInBytes} bytes.`);
};

// Start processing the TIFF files
processTiffFiles();
