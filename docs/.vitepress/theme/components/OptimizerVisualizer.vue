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

const pythonFiles = [
  '__init__.py',
  'bfgs.py',
  'lbfgs.py',
  'lbfgsb.py',
  'line_search.py',
  'problems.py',
  'utils.py'
];

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
  
  try {
    const pyodide = await getPyodide();
    
    // Define the problem and callback in Python
    pyodide.globals.set('on_step_js', (res: any) => {
      const data = res.toJs({ dict_converter: Object.fromEntries });
      history.value.push(data);
      currentIteration.value = data.nit;
    });

    const pythonCode = `
import numpy as np
from qnm.${props.algorithm} import ${props.algorithm}
from qnm.problems import ${props.problemType}_problem

# Setup problem
prob = ${props.problemType}_problem(dim=${props.dim || 2})

def callback(res):
    # Convert result to a dict that JS can handle easily
    # H is only available in BFGS
    h_matrix = None
    if hasattr(res, 'H'):
        h_matrix = res.H.tolist()
    elif "${props.algorithm}" == "bfgs":
        # In our bfgs.py, H is not in OptimizeResult yet, we might need to modify it 
        # or capture it here if we were inside the function.
        # For now, let's assume we want to see H.
        pass

    step_data = {
        "nit": int(res.nit),
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
        if 's_history' in res.extra_info:
            step_data['s_history'] = res.extra_info['s_history']
        if 'y_history' in res.extra_info:
            step_data['y_history'] = res.extra_info['y_history']
    
    import on_step_js
    on_step_js(step_data)

# Run optimization
res = ${props.algorithm}(
    prob.fun, 
    prob.grad, 
    prob.x0, 
    callback=callback,
    max_iter=50
)
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

const latestH = computed(() => {
  if (history.value.length === 0) return null;
  return history.value[history.value.length - 1].H || null;
});

const latestState = computed(() => {
  if (history.value.length === 0) return null;
  return history.value[history.value.length - 1];
});

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
          <span>{{ latestState.nit }}</span>
        </div>
        <div class="metric-card">
          <label>Function Value</label>
          <span>{{ latestState.fun.toFixed(6) }}</span>
        </div>
        <div class="metric-card">
          <label>Grad Norm</label>
          <span>{{ latestState.grad_norm.toExponential(3) }}</span>
        </div>
      </div>

      <div v-if="latestH" class="matrix-viz">
        <h3>近似逆ヘッセ行列 $H$</h3>
        <div class="matrix-grid" :style="{ gridTemplateColumns: `repeat(${dim || 2}, 1fr)` }">
          <div v-for="(row, i) in latestH" :key="i" class="matrix-row">
            <div v-for="(val, j) in row" :key="j" class="matrix-cell" 
                 :style="{ backgroundColor: `rgba(64, 128, 255, ${Math.min(Math.abs(val), 1)})` }">
              {{ val.toFixed(4) }}
            </div>
          </div>
        </div>
      </div>
      
      <div v-if="algorithm === 'lbfgs'" class="lbfgs-info">
        <h3>L-BFGS メモリバッファ ($s_k, y_k$)</h3>
        <p>直近 {{ latestState.s_history?.length || 0 }} 組のベクトルを保持しています。</p>
        <div class="history-grid">
          <div v-for="(s, i) in latestState.s_history" :key="i" class="history-item">
            <label>s[{{ i }}]</label>
            <div class="vector-row">
              <span v-for="(val, j) in s" :key="j">{{ val.toFixed(4) }}</span>
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
