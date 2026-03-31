/**
 * Responsibility: Copy Python modules from `src/python/qnm/` into `docs/public/qnm/`
 * so VitePress and Pyodide tooling can serve them. Runs before `vitepress dev` / `build`
 * on any OS (replaces Unix-only `mkdir -p` and `cp` in npm scripts).
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectoryPath = path.dirname(fileURLToPath(import.meta.url));
const docsRootPath = path.resolve(scriptDirectoryPath, "..");
const repositoryRootPath = path.resolve(docsRootPath, "..");
const qnmSourceDirectoryPath = path.join(repositoryRootPath, "src", "python", "qnm");
const qnmDestinationDirectoryPath = path.join(docsRootPath, "public", "qnm");

if (!fs.existsSync(qnmSourceDirectoryPath)) {
  console.error(`prepare-qnm-public: missing source directory: ${qnmSourceDirectoryPath}`);
  process.exit(1);
}

fs.mkdirSync(qnmDestinationDirectoryPath, { recursive: true });

const pyFileNameList = fs
  .readdirSync(qnmSourceDirectoryPath)
  .filter((fileName) => fileName.endsWith(".py"));

if (pyFileNameList.length === 0) {
  console.warn("prepare-qnm-public: no .py files found to copy.");
}

for (const pyFileName of pyFileNameList) {
  const fromPath = path.join(qnmSourceDirectoryPath, pyFileName);
  const toPath = path.join(qnmDestinationDirectoryPath, pyFileName);
  fs.copyFileSync(fromPath, toPath);
}

console.log(`prepare-qnm-public: copied ${pyFileNameList.length} file(s) to public/qnm.`);
