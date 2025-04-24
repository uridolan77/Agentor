/**
 * Script to toggle between strict and non-strict TypeScript modes
 * 
 * Usage:
 * - To use strict mode: node toggle-typescript-mode.js strict
 * - To use non-strict mode: node toggle-typescript-mode.js nostrict
 * - To check current mode: node toggle-typescript-mode.js status
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const TSCONFIG_PATH = path.join(__dirname, 'tsconfig.json');
const TSCONFIG_STRICT_PATH = path.join(__dirname, 'tsconfig.strict.json');
const TSCONFIG_NOSTRICT_PATH = path.join(__dirname, 'tsconfig.nostrict.json');

// Ensure we have a backup of the original strict config
function backupStrictConfig() {
  if (!fs.existsSync(TSCONFIG_STRICT_PATH)) {
    console.log('Creating backup of original strict tsconfig.json...');
    fs.copyFileSync(TSCONFIG_PATH, TSCONFIG_STRICT_PATH);
  }
}

// Check if the current config is strict or non-strict
function getCurrentMode() {
  try {
    const config = JSON.parse(fs.readFileSync(TSCONFIG_PATH, 'utf8'));
    if (config.compilerOptions && config.compilerOptions.strict === false) {
      return 'nostrict';
    }
    return 'strict';
  } catch (error) {
    console.error('Error reading tsconfig.json:', error.message);
    return 'unknown';
  }
}

// Switch to strict mode
function enableStrictMode() {
  backupStrictConfig();
  console.log('Switching to strict TypeScript mode...');
  fs.copyFileSync(TSCONFIG_STRICT_PATH, TSCONFIG_PATH);
  console.log('Done! TypeScript is now in strict mode.');
}

// Switch to non-strict mode
function enableNonStrictMode() {
  backupStrictConfig();
  console.log('Switching to non-strict TypeScript mode...');
  fs.copyFileSync(TSCONFIG_NOSTRICT_PATH, TSCONFIG_PATH);
  console.log('Done! TypeScript is now in non-strict mode.');
}

// Run TypeScript compiler to check for errors
function checkTypeScriptErrors() {
  console.log('Checking for TypeScript errors...');
  try {
    execSync('npx tsc --noEmit', { stdio: 'inherit' });
    console.log('No TypeScript errors found!');
    return true;
  } catch (error) {
    console.log('TypeScript errors found. See above for details.');
    return false;
  }
}

// Main function
function main() {
  const args = process.argv.slice(2);
  const command = args[0] || 'status';

  switch (command) {
    case 'strict':
      enableStrictMode();
      checkTypeScriptErrors();
      break;
    case 'nostrict':
      enableNonStrictMode();
      checkTypeScriptErrors();
      break;
    case 'status':
      const currentMode = getCurrentMode();
      console.log(`Current TypeScript mode: ${currentMode}`);
      checkTypeScriptErrors();
      break;
    default:
      console.log('Unknown command. Usage:');
      console.log('- To use strict mode: node toggle-typescript-mode.js strict');
      console.log('- To use non-strict mode: node toggle-typescript-mode.js nostrict');
      console.log('- To check current mode: node toggle-typescript-mode.js status');
      break;
  }
}

main();
