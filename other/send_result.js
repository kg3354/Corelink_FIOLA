// // const fs = require('fs');
// // const corelink = require('corelink-client');
// // const path = require('path');

// // const config = {
// //   ControlPort: 20012,
// //   ControlIP: 'corelink.hpc.nyu.edu',
// //   autoReconnect: false,
// // };

// // const username = 'Testuser';
// // const password = 'Testpassword';
// // const workspace = 'fiola-result';
// // const protocol = 'ws';
// // const datatype = 'distance';

// // async function run() {
// //   let sender;

// //   if (await corelink.connect({ username, password }, config).catch((err) => { console.log(err); })) {
// //     sender = await corelink.createSender({
// //       workspace,
// //       protocol,
// //       type: datatype,
// //       metadata: { name: 'Python Data' },
// //     }).catch((err) => { console.log(err); });

// //     // Ensure the result directory exists
// //     const resultDir = path.join(__dirname, 'result');
// //     if (!fs.existsSync(resultDir)) {
// //       fs.mkdirSync(resultDir);
// //     }

// //     process.stdin.on('data', (data) => {
// //       const frameIndex = data.readUInt16BE(0);
// //       const resultData = data.slice(2);

// //       // Save the received data to a file
// //       const outputPath = path.join(resultDir, `frame_${frameIndex}.bin`);
// //       fs.writeFileSync(outputPath, resultData);
// //       console.log(`Frame ${frameIndex} saved to ${outputPath}`);

// //       // Send the data to CoreLink
// //       corelink.send(sender, resultData);
// //       console.log(`Frame ${frameIndex} sent to CoreLink`);
// //     });

// //     process.stdin.on('end', () => {
// //       console.log('Finished receiving data');
// //     });
// //   }
// // }

// // run();
// const corelink = require('corelink-client');

// const config = {
//   ControlPort: 20012,
//   ControlIP: 'corelink.hpc.nyu.edu',
//   autoReconnect: true,
// };

// const username = 'Testuser';
// const password = 'Testpassword';
// const workspace = 'fiola-result';
// const protocol = 'ws';
// const datatype = 'distance';

// const run = async () => {
//   let sender;

//   if (await corelink.connect({ username, password }, config).catch((err) => { console.log(err); })) {
//     sender = await corelink.createSender({
//       workspace,
//       protocol,
//       type: datatype,
//       metadata: { name: 'Python Data' },
//     }).catch((err) => { console.log(err); });

//     process.stdin.on('data', (data) => {
//       corelink.send(sender, data);
//     });

//     process.stdin.on('end', () => {
//       console.log('Finished sending data');
//     });
//   }
// }

// run();
const corelink = require('corelink-client');

const config = {
  ControlPort: 20012,
  ControlIP: 'corelink.hpc.nyu.edu',
  autoReconnect: true,
};

const username = 'Testuser';
const password = 'Testpassword';
const workspace = 'fiola-result';
const protocol = 'ws';
const datatype = 'distance';

async function run() {
  let sender;

  if (await corelink.connect({ username, password }, config).catch((err) => { console.log(err); })) {
    sender = await corelink.createSender({
      workspace,
      protocol,
      type: datatype,
      metadata: { name: 'Python Data' },
    }).catch((err) => { console.log(err); });

    process.stdin.on('data', (data) => {
      corelink.send(sender, data);
    });

    process.stdin.on('end', () => {
      console.log('Finished sending data');
    });
  }
}

run();
