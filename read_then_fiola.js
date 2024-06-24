 
// // const { spawn } = require('child_process');
// // const fs = require('fs');
// // const path = require('path');


// // // Start the Python process once when the script starts
// // const pythonProcess = spawn('python', ['process_frame.py']);
// // // const pythonProcess = spawn('python', ['process_then_send.py']);

// // // Configuration for the sequence of TIFF files

// // const tiffFolder = './trash'; // Update with the path to your TIFF files
// // const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// // let frameCounter = 0;

// // // Event listeners for Python process output and errors
// // pythonProcess.stdout.on('data', (data) => {
// //   console.log(`Python output: ${data.toString()}`);
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
// //     const frameBuffer = fs.readFileSync(filePath);
// //     frameCounter++;
// //     sendFrameToPython(frameBuffer, frameCounter);
// //   }
// // };

// // const sendFrameToPython = (frameBuffer, frameCounter) => {
// //   // Write frame size to stdin
// //   const frameSize = Buffer.alloc(4);
// //   frameSize.writeUInt32LE(frameBuffer.length, 0);
// //   pythonProcess.stdin.write(frameSize);

// //   // Write frame data to stdin
// //   pythonProcess.stdin.write(frameBuffer);

// //   console.log(`Frame ${frameCounter} sent to Python for processing.`);
// // };

// // // Start processing the TIFF files
// // processTiffFiles();
// const { spawn } = require('child_process');
// const fs = require('fs');
// const path = require('path');

// // Start the Python process once when the script starts
// const pythonProcess = spawn('python', ['process_frame.py']);

// // Configuration for the sequence of TIFF files
// const tiffFolder = './trash'; // Update with the path to your TIFF files
// const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

// let frameCounter = 0;

// // Event listeners for Python process output and errors
// pythonProcess.stdout.on('data', (data) => {
//   console.log(`Python output: ${data.toString()}`);
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
//     const frameBuffer = fs.readFileSync(filePath);
//     frameCounter++;
//     sendFrameToPython(frameBuffer, frameCounter);
//   }
// };

// const sendFrameToPython = (frameBuffer, frameCounter) => {
//   // Write frame size to stdin
//   const frameSize = Buffer.alloc(4);
//   frameSize.writeUInt32LE(frameBuffer.length, 0);
//   pythonProcess.stdin.write(frameSize);

//   // Write frame data to stdin
//   pythonProcess.stdin.write(frameBuffer);

//   console.log(`Frame ${frameCounter} sent to Python for processing.`);
// };

// // Start processing the TIFF files
// processTiffFiles();
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// Start the Python process once when the script starts
const pythonProcess = spawn('python', ['process_frame.py']);

// Configuration for the sequence of TIFF files
const tiffFolder = './trash'; // Update with the path to your TIFF files
const tiffFiles = fs.readdirSync(tiffFolder).filter(file => file.endsWith('.tiff') || file.endsWith('.tif'));

let frameCounter = 0;

// Event listeners for Python process output and errors
pythonProcess.stdout.on('data', (data) => {
  console.log(`Python output: ${data.toString()}`);
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
    frameCounter++;
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
