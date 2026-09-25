const fs = require('fs');
const path = require('path');

// Import the JSON structure
let codeData;
try {
  // require() resolves relative to the script's location
  codeData = require('./code_frontend.json');
} catch (err) {
  console.error('❌ Error: Could not find or parse "code.json".');
  console.error('Make sure code.json exists in the same folder as this script.');
  process.exit(1);
}

// Define output directory (defaults to current working directory, or pass as CLI argument)
const outputDir = process.argv[2] ? path.resolve(process.argv[2]) : 'front_end_project';

console.log(`📂 Generating project in: ${outputDir}\n`);

codeData.forEach(({ filename, location, content }) => {
  // Resolve the target directory and file path
  const dirPath = path.join(outputDir, location);
  const filePath = path.join(dirPath, filename);

  // Create nested directories recursively if they don't exist
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
  }

  // Write the file content
  fs.writeFileSync(filePath, content, 'utf8');
  
  // Log success
  const relativePath = location ? path.join(location, filename) : filename;
  console.log(`✅ Created: ${relativePath}`);
});

console.log('\n🎉 Project generation complete!');
console.log('👉 Next steps:');
const cdPath = outputDir === process.cwd() ? '.' : path.relative(process.cwd(), outputDir);
console.log(`   1. cd ${cdPath}`);
console.log('   2. npm install');
console.log('   3. npm run dev');