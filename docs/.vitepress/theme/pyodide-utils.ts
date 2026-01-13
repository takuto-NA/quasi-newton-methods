// Pyodide loader and manager for VitePress

let pyodideInstance: any = null;
let pyodidePromise: Promise<any> | null = null;

const PYODIDE_CDN_URL = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js";

async function loadScript(url: string): Promise<void> {
  return new Promise((resolve, reject) => {
    if (typeof window === "undefined") return resolve();
    if (document.querySelector(`script[src="${url}"]`)) return resolve();
    const script = document.createElement("script");
    script.src = url;
    script.onload = () => resolve();
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

export async function getPyodide() {
  if (pyodideInstance) return pyodideInstance;
  if (pyodidePromise) return pyodidePromise;

  pyodidePromise = (async () => {
    await loadScript(PYODIDE_CDN_URL);
    // @ts-ignore
    pyodideInstance = await loadPyodide();
    await pyodideInstance.loadPackage(["numpy"]);
    return pyodideInstance;
  })();

  return pyodidePromise;
}

/**
 * Syncs local Python source files to Pyodide's virtual file system.
 */
export async function syncPythonSource(pyodide: any, files: string[]) {
  const baseUrl = "/quasi-newton-methods/qnm/";
  
  // Create directory if not exists
  pyodide.FS.mkdirTree("/home/pyodide/qnm");

  for (const file of files) {
    const response = await fetch(`${baseUrl}${file}`);
    if (!response.ok) {
      console.error(`Failed to fetch ${file}: ${response.statusText}`);
      continue;
    }
    const content = await response.text();
    pyodide.FS.writeFile(`/home/pyodide/qnm/${file}`, content);
  }

  // Add to sys.path
  pyodide.runPython(`
import sys
if "/home/pyodide" not in sys.path:
    sys.path.append("/home/pyodide")
`);
}
