const { spawn } = require('child_process');
const path = require('path');

// Get the current directory
const currentDir = __dirname;

// Run npm start
const npm = spawn('npm', ['start'], {
  cwd: currentDir,
  stdio: 'inherit',
  shell: true
});

npm.on('error', (err) => {
  console.error('Failed to start npm:', err);
});

npm.on('close', (code) => {
  console.log(`npm process exited with code ${code}`);
});
