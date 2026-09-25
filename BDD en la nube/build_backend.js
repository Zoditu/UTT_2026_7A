#!/usr/bin/env node

/**
 * Usage:
 *   node create-project-from-json.js files.json
 *   node create-project-from-json.js files.json ./my-project
 *   node create-project-from-json.js files.json --output ./my-project
 *
 * JSON format:
 * [
 *   {
 *     "filename": "app.js",
 *     "location": "src",
 *     "content": "console.log('hello');"
 *   }
 * ]
 */

const fs = require('fs/promises');
const path = require('path');

function printHelp() {
  console.log(`
Create a project from a JSON file.

Usage:
  node create-project-from-json.js <files.json> [outputDirectory]
  node create-project-from-json.js <files.json> --output ./my-project

JSON format:
[
  {
    "filename": "server.js",
    "location": "src",
    "content": "console.log('hello');"
  }
]
`);
}

function parseArgs(argv) {
  const args = argv.slice(2);
  const positionals = [];
  let outputDir;

  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];

    if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }

    if (arg === '--output' || arg === '-o') {
      if (i + 1 >= args.length) {
        throw new Error('Missing value for --output.');
      }
      outputDir = args[i + 1];
      i += 1;
      continue;
    }

    if (arg.startsWith('--output=')) {
      outputDir = arg.slice('--output='.length);
      continue;
    }

    if (arg.startsWith('-')) {
      throw new Error(`Unknown option: ${arg}. Use --help for usage.`);
    }

    positionals.push(arg);
  }

  if (positionals.length === 0) {
    printHelp();
    process.exit(1);
  }

  return {
    inputFile: positionals[0],
    outputDir: outputDir || positionals[1] || '.'
  };
}

async function readJsonFile(inputFile) {
  const absoluteInput = path.resolve(inputFile);
  let raw;

  try {
    raw = await fs.readFile(absoluteInput, 'utf8');
  } catch (error) {
    throw new Error(`Could not read JSON file at ${absoluteInput}: ${error.message}`);
  }

  if (raw.charCodeAt(0) === 0xfeff) {
    raw = raw.slice(1);
  }

  try {
    return JSON.parse(raw);
  } catch (error) {
    throw new Error(`Invalid JSON in ${absoluteInput}: ${error.message}`);
  }
}

function getEntries(json) {
  if (Array.isArray(json)) {
    return json;
  }

  if (json && Array.isArray(json.files)) {
    return json.files;
  }

  throw new Error('JSON must be an array of file objects or an object with a "files" array.');
}

function normalizeEntry(entry, index) {
  if (!entry || typeof entry !== 'object' || Array.isArray(entry)) {
    throw new Error(`Entry at index ${index} must be an object.`);
  }

  const filename = entry.filename;
  const location = entry.location ?? '';
  let content = entry.content ?? '';

  if (typeof filename !== 'string' || filename.trim().length === 0) {
    throw new Error(`Entry at index ${index} must include a non-empty "filename".`);
  }

  if (typeof location !== 'string') {
    throw new Error(`Entry at index ${index} has an invalid "location". It must be a string.`);
  }

  if (filename === '.' || filename === '..') {
    throw new Error(`Entry at index ${index} has an invalid "filename".`);
  }

  if (filename.includes('/') || filename.includes('\\') || filename.includes('\0')) {
    throw new Error(`Entry at index ${index} has a "filename" containing a path separator.`);
  }

  if (path.isAbsolute(location)) {
    throw new Error(`Entry at index ${index} has an absolute "location".`);
  }

  const locationParts = location
    .split(/[\/\\]+/)
    .filter((part) => part !== '' && part !== '.');

  for (const part of locationParts) {
    if (part === '..' || part.includes('\0')) {
      throw new Error(`Entry at index ${index} has an unsafe "location".`);
    }
  }

  if (typeof content !== 'string') {
    content = JSON.stringify(content, null, 2);
  }

  return {
    filename,
    location: locationParts.join('/'),
    content
  };
}

function resolveSafePath(rootDir, location, filename) {
  const root = path.resolve(rootDir);
  const target = path.resolve(root, location, filename);
  const relative = path.relative(root, target);

  if (relative === '') {
    throw new Error('Resolved file path is the project root.');
  }

  if (path.isAbsolute(relative) || relative.split(path.sep).some((part) => part === '..')) {
    throw new Error(`Unsafe file path detected: ${target}`);
  }

  return target;
}

async function createProject(inputFile, outputDir) {
  const json = await readJsonFile(inputFile);
  const entries = getEntries(json);
  const rootDir = path.resolve(outputDir);

  await fs.mkdir(rootDir, { recursive: true });

  const createdFiles = [];

  for (let index = 0; index < entries.length; index += 1) {
    const entry = normalizeEntry(entries[index], index);
    const filePath = resolveSafePath(rootDir, entry.location, entry.filename);

    await fs.mkdir(path.dirname(filePath), { recursive: true });
    await fs.writeFile(filePath, entry.content, 'utf8');
    createdFiles.push(path.relative(rootDir, filePath));
  }

  return {
    rootDir,
    createdFiles
  };
}

async function main() {
  try {
    const { inputFile, outputDir } = { inputFile: "code.json", outputDir: "project" };
    const { rootDir, createdFiles } = await createProject(inputFile, outputDir);

    console.log(`Created ${createdFiles.length} file(s) in ${rootDir}`);

    for (const file of createdFiles) {
      console.log(`- ${file}`);
    }
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

main();