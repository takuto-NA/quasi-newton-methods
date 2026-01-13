<script setup lang="ts">
import { ref, onMounted, computed } from 'vue';
import { getPyodide, syncPythonSource } from '../pyodide-utils';

const props = defineProps<{
  algorithm: 'bfgs' | 'lbfgs';
  problemType: 'quadratic' | 'rosenbrock';
  dim?: number;
}>();

const status = ref('Initializing...');
const isRunning = ref(false);
const isLoaded = ref(false);
const results = ref<any>(null);
const currentIteration = ref(0);
const history = ref<any[]>([]);
const output = ref('');
const dimension = computed(() => props.dim ?? 2);
const selectedIndex = ref<number>(-1);
const autoFollowLatest = ref(true);

const pythonFiles = [
  '__init__.py',
  'bfgs.py',
  'lbfgs.py',
  'lbfgsb.py',
  'line_search.py',
  'problems.py',
  'utils.py'
];

function appendOutput(chunk: unknown) {
  const s = String(chunk ?? '');
  output.value += s;
  if (output.value.length > 200_000) {
    output.value = output.value.slice(output.value.length - 200_000);
  }
}

async function init() {
  try {
    status.value = 'Loading Pyodide...';
    const pyodide = await getPyodide();
    
    status.value = 'Syncing source files...';
    await syncPythonSource(pyodide, pythonFiles);
    
    status.value = 'Ready';
    isLoaded.value = true;
  } catch (e: any) {
    status.value = `Error: ${e.message}`;
    console.error(e);
  }
}

async function runOptimization() {
  if (isRunning.value) return;
  isRunning.value = true;
  output.value = '';
  history.value = [];
  currentIteration.value = 0;
  selectedIndex.value = -1;
  autoFollowLatest.value = true;
  
  try {
    const pyodide = await getPyodide();
    
    // Define the problem and callback in Python
    pyodide.globals.set('on_step_js', (res: any) => {
      const data = res.toJs({ dict_converter: Object.fromEntries });
      history.value.push(data);
      currentIteration.value = data.n_iter;
      if (autoFollowLatest.value) {
        selectedIndex.value = history.value.length - 1;
      }
    });
    pyodide.globals.set('on_log_js', (chunk: any) => {
      appendOutput(chunk);
    });

    const pythonCode = `
import numpy as np
import sys
from qnm.${props.algorithm} import ${props.algorithm}
from qnm.problems import ${props.problemType}_problem

# Setup problem
prob = ${props.problemType}_problem(dim=${dimension.value})

class _JSWriter:
    def __init__(self, fn):
        self._fn = fn
    def write(self, s):
        # Some libs write non-str objects; coerce to string
        self._fn(str(s))
        return len(str(s))
    def flush(self):
        return None

sys.stdout = _JSWriter(on_log_js)
sys.stderr = _JSWriter(on_log_js)

def callback(res):
    # Convert result to a dict that JS can handle easily
    step_data = {
        "n_iter": int(res.n_iter),
        "fun": float(res.fun),
        "x": res.x.tolist(),
        "grad_norm": float(np.linalg.norm(res.grad)),
        "success": bool(res.success),
        "message": str(res.message)
    }
    
    # We'll inject H or history into the callback
    if hasattr(res, 'extra_info'):
        if 'H' in res.extra_info:
            step_data['H'] = res.extra_info['H'].tolist()
        if 'alpha' in res.extra_info:
            step_data['alpha'] = float(res.extra_info['alpha'])
        if 'ys' in res.extra_info:
            step_data['ys'] = float(res.extra_info['ys'])
        if 'step_norm' in res.extra_info:
            step_data['step_norm'] = float(res.extra_info['step_norm'])
        if 's_history' in res.extra_info:
            step_data['s_history'] = res.extra_info['s_history']
        if 'y_history' in res.extra_info:
            step_data['y_history'] = res.extra_info['y_history']
    
    # Human-readable convergence log
    alpha = step_data.get("alpha", None)
    if alpha is None:
        print(f"[iter {step_data['n_iter']:3d}] f={step_data['fun']:.6e}  ||g||={step_data['grad_norm']:.3e}")
    else:
        sn = step_data.get("step_norm", float('nan'))
        print(f"[iter {step_data['n_iter']:3d}] f={step_data['fun']:.6e}  ||g||={step_data['grad_norm']:.3e}  alpha={alpha:.3e}  ||s||={sn:.3e}")

    on_step_js(step_data)

# Run optimization
print(f"=== Run {\"${props.algorithm}\".upper()} on {\"${props.problemType}\"} (dim=${dimension.value}) ===")
res = ${props.algorithm}(
    prob.fun, 
    prob.grad, 
    prob.x0, 
    callback=callback,
    max_iter=50
)
print(f"=== Done: success={res.success}, n_iter={res.n_iter}, f={res.fun:.6e} ===\\n")
res
`;
    status.value = 'Running...';
    await pyodide.runPythonAsync(pythonCode);
    status.value = 'Completed';
  } catch (e: any) {
    status.value = `Error: ${e.message}`;
    output.value += `\nError: ${e.message}`;
  } finally {
    isRunning.value = false;
  }
}

const selectedState = computed(() => {
  if (history.value.length === 0) return null;
  const idx = selectedIndex.value < 0 ? history.value.length - 1 : selectedIndex.value;
  return history.value[Math.min(Math.max(idx, 0), history.value.length - 1)];
});

const selectedH = computed(() => {
  return selectedState.value?.H ?? null;
});

function selectIteration(idx: number) {
  selectedIndex.value = idx;
  autoFollowLatest.value = false;
}

onMounted(() => {
  // We don't auto-init to save bandwidth unless needed
});
</script>

<template>
  <div class="optimizer-visualizer">
    <div class="controls">
      <button v-if="!isLoaded" @click="init" class="btn-primary">
        デモをロード（Pyodide + NumPy）
      </button>
      <button v-else @click="runOptimization" :disabled="isRunning" class="btn-run">
        {{ isRunning ? '実行中...' : '最適化実行' }}
      </button>
      <span class="status-badge">{{ status }}</span>
    </div>

    <div v-if="history.length > 0" class="viz-container">
      <div class="metrics">
        <div class="metric-card">
          <label>Iteration</label>
          <span>{{ selectedState.n_iter }}</span>
        </div>
        <div class="metric-card">
          <label>Function Value</label>
          <span>{{ selectedState.fun.toFixed(6) }}</span>
        </div>
        <div class="metric-card">
          <label>Grad Norm</label>
          <span>{{ selectedState.grad_norm.toExponential(3) }}</span>
        </div>
      </div>

      <div class="log-panel">
        <div class="log-header">
          <h3>収束ログ（反復履歴）</h3>
          <div class="log-controls">
            <label class="checkbox">
              <input type="checkbox" v-model="autoFollowLatest" />
              最新に追従
            </label>
            <span class="small">（クリックで任意の反復を選択）</span>
          </div>
        </div>

        <div class="history-table-wrap">
          <table class="history-table">
            <thead>
              <tr>
                <th>iter</th>
                <th>f(x)</th>
                <th>||g||</th>
                <th v-if="algorithm === 'bfgs'">alpha</th>
                <th v-if="algorithm === 'bfgs'">||s||</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="(row, i) in history"
                :key="i"
                :class="{ selected: i === selectedIndex }"
                @click="selectIteration(i)"
              >
                <td class="mono">{{ row.n_iter }}</td>
                <td class="mono">{{ Number(row.fun).toExponential(6) }}</td>
                <td class="mono">{{ Number(row.grad_norm).toExponential(3) }}</td>
                <td v-if="algorithm === 'bfgs'" class="mono">
                  {{ row.alpha != null ? Number(row.alpha).toExponential(3) : '-' }}
                </td>
                <td v-if="algorithm === 'bfgs'" class="mono">
                  {{ row.step_norm != null ? Number(row.step_norm).toExponential(3) : '-' }}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div v-if="selectedH" class="matrix-viz">
        <h3>近似逆ヘッセ行列 $H$</h3>
        <div class="matrix-grid" :style="{ gridTemplateColumns: `repeat(${dimension}, 1fr)` }">
          <div v-for="(row, i) in selectedH" :key="i" class="matrix-row">
            <div v-for="(val, j) in row" :key="j" class="matrix-cell" 
                 :style="{ backgroundColor: `rgba(64, 128, 255, ${Math.min(Math.abs(val), 1)})` }">
              {{ val.toFixed(4) }}
            </div>
          </div>
        </div>
      </div>
      
      <div v-if="algorithm === 'lbfgs'" class="lbfgs-info">
        <h3>L-BFGS メモリバッファ ($s_k, y_k$)</h3>
        <p>直近 {{ selectedState.s_history?.length || 0 }} 組のベクトルを保持しています。</p>
        <div class="history-grid">
          <div v-for="(s, i) in selectedState.s_history" :key="i" class="history-item">
            <label>s[{{ i }}]</label>
            <div class="vector-row">
              <span v-for="(val, j) in s" :key="j">{{ val.toFixed(4) }}</span>
            </div>
          </div>
        </div>
        <div class="history-grid">
          <div v-for="(y, i) in selectedState.y_history" :key="i" class="history-item">
            <label>y[{{ i }}]</label>
            <div class="vector-row">
              <span v-for="(val, j) in y" :key="j">{{ val.toFixed(4) }}</span>
            </div>
          </div>
        </div>
        <p class="description">L-BFGS では完全な行列 $H$ を保持せず、これらの直近 $m$ 回の $s_k, y_k$ ペアから two-loop recursion により探索方向を計算します。</p>
      </div>
    </div>

    <pre v-if="output" class="console-output">{{ output }}</pre>
  </div>
</template>

<style scoped>
.optimizer-visualizer {
  margin: 20px 0;
  padding: 20px;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  background-color: var(--vp-c-bg-soft);
}

.controls {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 20px;
}

.btn-primary, .btn-run {
  padding: 8px 16px;
  border-radius: 4px;
  background-color: var(--vp-c-brand);
  color: white;
  border: none;
  cursor: pointer;
  font-weight: bold;
}

.btn-run:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.status-badge {
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.viz-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.metrics {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
}

.metric-card {
  padding: 10px;
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
  text-align: center;
}

.metric-card label {
  display: block;
  font-size: 0.8em;
  color: var(--vp-c-text-2);
}

.metric-card span {
  font-family: var(--vp-font-family-mono);
  font-weight: bold;
}

.matrix-viz h3 {
  margin-top: 0;
  font-size: 1.1em;
}

.matrix-grid {
  display: grid;
  gap: 2px;
  background: var(--vp-c-divider);
  padding: 2px;
  border-radius: 4px;
  overflow: auto;
}

.matrix-cell {
  background: var(--vp-c-bg);
  padding: 8px;
  text-align: center;
  font-family: var(--vp-font-family-mono);
  font-size: 0.85em;
  min-width: 80px;
}

.log-panel {
  border: 1px solid var(--vp-c-divider);
  background: var(--vp-c-bg);
  border-radius: 6px;
  padding: 12px;
}

.log-header {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 10px;
}

.log-header h3 {
  margin: 0;
  font-size: 1.1em;
}

.log-controls {
  display: flex;
  align-items: center;
  gap: 10px;
}

.checkbox {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.small {
  font-size: 0.85em;
  color: var(--vp-c-text-2);
}

.history-table-wrap {
  max-height: 240px;
  overflow: auto;
  border: 1px solid var(--vp-c-divider);
  border-radius: 4px;
}

.history-table {
  width: 100%;
  border-collapse: collapse;
}

.history-table th,
.history-table td {
  padding: 6px 8px;
  border-bottom: 1px solid var(--vp-c-divider);
  text-align: right;
}

.history-table th {
  position: sticky;
  top: 0;
  background: var(--vp-c-bg);
  z-index: 1;
  font-size: 0.85em;
  color: var(--vp-c-text-2);
}

.history-table tbody tr {
  cursor: pointer;
}

.history-table tbody tr:hover {
  background: var(--vp-c-bg-soft);
}

.history-table tbody tr.selected {
  background: rgba(64, 128, 255, 0.12);
}

.mono {
  font-family: var(--vp-font-family-mono);
}

.history-grid {
  display: flex;
  flex-direction: column;
  gap: 10px;
  margin-bottom: 10px;
}

.history-item {
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--vp-c-bg);
  padding: 8px;
  border-radius: 4px;
  border: 1px solid var(--vp-c-divider);
}

.history-item label {
  font-weight: bold;
  font-size: 0.8em;
  min-width: 40px;
}

.vector-row {
  display: flex;
  gap: 8px;
  font-family: var(--vp-font-family-mono);
  font-size: 0.85em;
}

.description {
  font-size: 0.9em;
  color: var(--vp-c-text-2);
}

.console-output {
  margin-top: 20px;
  padding: 10px;
  background: #1e1e1e;
  color: #d4d4d4;
  border-radius: 4px;
  font-size: 0.85em;
  max-height: 200px;
  overflow-y: auto;
}
</style>
