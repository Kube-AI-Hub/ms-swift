// SWIFT Web-UI Restful – main frontend logic
(function () {
  'use strict';

  // ── API base (relative to current page path, works under any prefix) ──
  const apiBase = window.location.pathname.replace(/\/[^/]*$/, '').replace(/\/$/, '');

  // Build an absolute TensorBoard URL by appending the backend-provided
  // relative path (e.g. "/tensorboard/") to the current page URL.
  // Example: page "https://host/entrypoint/task1/" + "/tensorboard/"
  //          → "https://host/entrypoint/task1/tensorboard/".
  function buildTensorBoardUrl(relPath) {
    const rel = String(relPath || '/tensorboard/').replace(/^\/+/, '');
    const loc = window.location;
    let path = loc.pathname || '/';
    if (!path.endsWith('/')) {
      path = path.substring(0, path.lastIndexOf('/') + 1);
    }
    return loc.origin + path + rel;
  }

  // ── Language ──
  const urlParams = new URLSearchParams(window.location.search);
  let pageLang = urlParams.get('lang') === 'en' ? 'en' : 'zh';
  const langSelect = document.getElementById('lang-select');
  langSelect.value = pageLang;
  langSelect.addEventListener('change', () => {
    pageLang = langSelect.value;
    window.applyI18n(pageLang);
  });

  const DEFAULT_GRPO_SYSTEM =
    'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. '
    + 'The assistant first thinks about the reasoning process in the mind and then provides the user '
    + 'with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> '
    + '</answer> tags, respectively, i.e., <think> reasoning process here </think>'
    + '<answer> answer here </answer>';

  // ── Tab config: maps tab name → [cmd, selectId, statusId] ──
  const tabCmds = {
    train:  [['sft', 'pt'], 'train-running-tasks', 'train-status'],
    rlhf:   ['rlhf',   'rlhf-running-tasks',   'rlhf-status'],
    grpo:   ['rlhf',   'grpo-running-tasks',   'grpo-status'],
    infer:  ['deploy', 'infer-running-tasks',  'infer-status'],
    export: ['export', 'export-running-tasks', 'export-status'],
    eval:   ['eval',   'eval-running-tasks',   'eval-status'],
    sample: ['sample', 'sample-running-tasks', 'sample-status'],
  };

  // ── Tab switching ──
  document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.getAttribute('data-target');
      document.querySelectorAll('.section').forEach(s => {
        s.classList.toggle('active', s.id === `section-${target}`);
      });
      // When switching to any tab, refresh its running tasks
      if (target === 'rlhf') {
        refreshRlhfTasks();
      } else if (target === 'grpo') {
        refreshGrpoTasks();
      } else if (tabCmds[target]) {
        refreshTasks(...tabCmds[target]).then(() => {
          if (target === 'infer' || target === 'export' || target === 'eval' || target === 'sample') {
            updateTaskButtons(target);
          }
        });
      }
    });
  });

  // ── Accordion toggle ──
  document.querySelectorAll('.accordion-header').forEach(header => {
    header.addEventListener('click', (e) => {
      // Don't toggle if a button inside the header was clicked
      if (e.target.tagName === 'BUTTON') return;
      header.closest('.accordion').classList.toggle('open');
    });
  });

  // ── Advanced accordion inner tabs ──
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const container = btn.closest('.accordion-body') || btn.parentElement.parentElement;
      container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.getAttribute('data-tab');
      container.querySelectorAll('.tab-panel').forEach(p => {
        p.classList.toggle('active', p.id === target);
      });
    });
  });

  // ── Media tabs (infer) ──
  document.querySelectorAll('.media-tab').forEach(tab => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.media-tab').forEach(t => t.classList.remove('active'));
      tab.classList.add('active');
      const target = tab.getAttribute('data-media');
      document.querySelectorAll('.media-panel').forEach(p => {
        p.classList.toggle('active', p.id === `media-${target}`);
      });
    });
  });

  // ── Helper: API fetch ──
  async function apiFetch(path, options = {}) {
    const url = `${apiBase}${path}`;
    const res = await fetch(url, options);
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(body.detail || res.statusText);
    }
    return res.json();
  }

  // ── Helper: set status text ──
  function setStatus(id, text, color) {
    const el = document.getElementById(id);
    if (el) {
      el.textContent = text;
      el.style.color = color || 'var(--gray-700)';
    }
    if (color === 'var(--danger)') showToast(text, 'error');
  }

  // ── Toast notifications ──
  function showToast(msg, type = 'error', duration = 5000) {
    const container = document.getElementById('toast-container');
    if (!container || !msg) return;
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const span = document.createElement('span');
    span.className = 'toast-msg';
    span.textContent = msg;
    const btn = document.createElement('button');
    btn.className = 'toast-close';
    btn.textContent = '×';
    btn.onclick = () => toast.remove();
    toast.appendChild(span);
    toast.appendChild(btn);
    container.appendChild(toast);
    if (duration > 0) setTimeout(() => toast.remove(), duration);
  }

  // ── Helper: show/clear validation error (now delegates to toast) ──
  function showTrainError(msg) { showToast(msg, 'error'); }
  function clearTrainError() { /* no-op — toasts are self-dismissing */ }

  // ── Helper: toggle button loading spinner ──
  function setBtnLoading(btn, loading) {
    if (!btn) return;
    if (loading) { btn.classList.add('btn-loading'); btn.disabled = true; }
    else          { btn.classList.remove('btn-loading'); btn.disabled = false; }
  }

  // ── Helper: sync train kill-button enabled state ──
  function updateTrainKillBtn() {
    const sel = document.getElementById('train-running-tasks');
    const btn = document.getElementById('train-btn-kill');
    const refreshBtn = document.getElementById('train-btn-refresh');
    if (btn && sel) btn.disabled = !sel.value;
    if (refreshBtn) refreshBtn.disabled = !sel || !sel.value;
  }

  // ── Helper: collect non-empty string or null ──
  function val(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    const v = el.type === 'checkbox' ? el.checked : el.value.trim();
    if (v === '' || v === false) return null;
    return v;
  }

  function numVal(id) {
    const v = val(id);
    return v !== null ? Number(v) : null;
  }

  function gpuList(id) {
    const el = document.getElementById(id);
    if (!el) return null;
    if (el.tagName === 'SELECT') {
      const selected = Array.from(el.selectedOptions).map(o => o.value).filter(Boolean);
      return selected.length ? selected : null;
    }
    const v = el.value;
    if (!v) return null;
    return v.split(/[\s,]+/).filter(Boolean);
  }

  function datasetList(id) {
    const v = val(id);
    if (!v) return null;
    return v.split(/\s+/).filter(Boolean);
  }

  function multiSelectList(id) {
    const el = document.getElementById(id);
    if (!el || !el.multiple) return null;
    const values = Array.from(el.selectedOptions).map(o => o.value).filter(Boolean);
    return values.length ? values : null;
  }

  // ── SSE log streaming ──
  const logSources = {};   // key -> EventSource
  const logStopFlags = {}; // key -> bool

  let trainCurrentLoggingDir = '';
  let rlhfCurrentLoggingDir = '';
  let grpoCurrentLoggingDir = '';
  let trainRestoring = false;
  const RLHF_PLOT_KEYS = {
    dpo:   ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/logps/chosen', 'train/logps/rejected'],
    cpo:   ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/logps/chosen', 'train/logps/rejected'],
    simpo: ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/logps/chosen', 'train/logps/rejected'],
    kto:   ['kl', 'rewards/chosen_sum', 'logps/chosen_sum', 'rewards/rejected_sum', 'logps/rejected_sum'],
    orpo:  ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/rewards/chosen', 'train/log_odds_ratio'],
    rm:    ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/logps/chosen', 'train/logps/rejected'],
    ppo:   ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/logps/chosen', 'train/logps/rejected'],
    gkd:   ['train/loss', 'train/rewards/accuracies', 'train/rewards/margins', 'train/logps/chosen', 'train/logps/rejected'],
  };
  const GRPO_KEYS = ['train/loss', 'train/reward', 'train/learning_rate', 'train/completions/mean_length', 'train/kl'];

  const METRICS_CFG = {
    train: { keys: ['train/loss', 'train/acc', 'train/learning_rate', 'eval/loss', 'eval/acc'], endpoint: '/api/v1/train/tensorboard-metrics', slotBased: false },
    rlhf:  { keys: RLHF_PLOT_KEYS.dpo, endpoint: '/api/v1/rlhf/tensorboard-metrics', slotBased: true },
    grpo:  { keys: GRPO_KEYS, endpoint: '/api/v1/grpo/tensorboard-metrics', slotBased: true },
  };
  const metricsTimers = { train: null, rlhf: null, grpo: null };

  function syncRlhfMetricsKeys() {
    const rt = val('rlhf-type') || 'dpo';
    const keys = RLHF_PLOT_KEYS[rt] || RLHF_PLOT_KEYS.dpo;
    METRICS_CFG.rlhf.keys = keys;
    keys.forEach((k, i) => {
      const t = document.getElementById('rlhf-chart-title-' + i);
      if (t) t.textContent = k;
    });
  }

  function chartCanvasId(prefix, key, index) {
    const cfg = METRICS_CFG[prefix];
    if (cfg && cfg.slotBased) return `${prefix}-chart-${index}`;
    return `${prefix}-chart-` + key.replace(/[\/_]/g, '-');
  }

  // Resolve an absolute logging_dir given a potentially-relative value and
  // the absolute log_file path (used as the CWD anchor).
  function _absoluteLoggingDir(loggingDir, logFile) {
    if (!loggingDir) return '';
    // Already absolute
    if (loggingDir.startsWith('/') || /^[A-Za-z]:[\\/]/.test(loggingDir)) return loggingDir;
    // Relative — anchor to the log_file's directory
    if (!logFile) return loggingDir;
    const base = logFile.lastIndexOf('/');
    if (base <= 0) return loggingDir;
    const logDir = logFile.slice(0, base);
    // e.g. logDir = /abs/path/output/run-xxx, loggingDir = output/run-xxx
    // Walk up from logDir until we find a common anchor that makes sense.
    // Simplest heuristic: return the dir derived from log_file (they share the same dir).
    return logDir;
  }

  function resolveLoggingDir(prefix) {
    const map = { train: trainCurrentLoggingDir, rlhf: rlhfCurrentLoggingDir, grpo: grpoCurrentLoggingDir };
    if (map[prefix]) return map[prefix];
    const sel = document.getElementById(prefix + '-running-tasks');
    if (!sel || !sel.value) return '';
    const lf = sel.value;  // absolute log_file path
    const t = _taskCache[prefix + '-running-tasks'] && _taskCache[prefix + '-running-tasks'][lf];
    if (t && t.args && t.args.logging_dir) {
      return _absoluteLoggingDir(t.args.logging_dir, lf);
    }
    const idx = lf.lastIndexOf('/');
    if (idx <= 0) return '';
    return lf.slice(0, idx);
  }

  function updateMetricCharts(prefix, series) {
    if (typeof Chart === 'undefined') return;
    const cfg = METRICS_CFG[prefix];
    if (!cfg) return;
    const panel = document.getElementById(prefix + '-metrics-panel');
    if (panel) panel.style.display = '';
    const rawLabel = (window.i18n[pageLang] && window.i18n[pageLang].metricLegendRaw) || 'raw';
    const smLabel = (window.i18n[pageLang] && window.i18n[pageLang].metricLegendSmoothed) || 'smoothed';
    cfg.keys.forEach((key, idx) => {
      const canvas = document.getElementById(chartCanvasId(prefix, key, idx));
      if (!canvas) return;
      if (cfg.slotBased) {
        const titleEl = document.getElementById(`${prefix}-chart-title-${idx}`);
        if (titleEl) titleEl.textContent = key;
      }
      const s = series[key];
      let chart = Chart.getChart(canvas);
      if (!s || !s.step || s.step.length === 0) {
        if (chart) chart.destroy();
        return;
      }
      const pts = s.step.map((st, i) => ({ x: st, y: s.value[i] }));
      const datasets = [{
        label: rawLabel,
        data: pts,
        borderColor: 'rgba(148, 163, 184, 0.9)',
        backgroundColor: 'transparent',
        pointRadius: pts.length === 1 ? 4 : 0,
        tension: 0,
        borderWidth: 1.5,
      }];
      if (s.value_smoothed && s.value_smoothed.length === s.step.length) {
        datasets.push({
          label: smLabel,
          data: s.step.map((st, i) => ({ x: st, y: s.value_smoothed[i] })),
          borderColor: 'rgba(249, 115, 22, 0.95)',
          backgroundColor: 'transparent',
          pointRadius: 0,
          tension: 0,
          borderWidth: 1.5,
        });
      }
      if (!chart) {
        chart = new Chart(canvas, {
          type: 'line',
          data: { datasets },
          options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: 'index', intersect: false },
            plugins: {
              legend: {
                display: datasets.length > 1,
                labels: { color: '#cfe4f2', boxWidth: 10, font: { size: 10 } },
              },
              tooltip: { enabled: true },
            },
            scales: {
              x: {
                type: 'linear',
                ticks: { color: '#8ea0c8', maxTicksLimit: 6 },
                grid: { color: 'rgba(255,255,255,0.06)' },
              },
              y: {
                ticks: { color: '#8ea0c8', maxTicksLimit: 6 },
                grid: { color: 'rgba(255,255,255,0.06)' },
              },
            },
          },
        });
      } else {
        chart.data.datasets = datasets;
        chart.options.plugins.legend.display = datasets.length > 1;
        chart.update('none');
      }
    });
  }

  async function fetchMetrics(prefix) {
    const cfg = METRICS_CFG[prefix];
    if (!cfg) return;
    const ld = resolveLoggingDir(prefix);
    if (!ld || typeof Chart === 'undefined') return;
    try {
      let url = `${cfg.endpoint}?logging_dir=${encodeURIComponent(ld)}`;
      if (prefix === 'rlhf') {
        url += '&rlhf_type=' + encodeURIComponent(val('rlhf-type') || 'dpo');
      }
      const data = await apiFetch(url);
      updateMetricCharts(prefix, data.series || {});
    } catch (_) {}
  }

  function startMetricsPoll(prefix) {
    const cfg = METRICS_CFG[prefix];
    if (!cfg) return;
    stopMetricsPoll(prefix);
    const panel = document.getElementById(prefix + '-metrics-panel');
    if (panel) panel.style.display = '';
    fetchMetrics(prefix);
    metricsTimers[prefix] = setInterval(() => fetchMetrics(prefix), 2000);
  }

  function stopMetricsPoll(prefix) {
    const cfg = METRICS_CFG[prefix];
    if (!cfg) return;
    if (metricsTimers[prefix]) {
      clearInterval(metricsTimers[prefix]);
      metricsTimers[prefix] = null;
    }
    const panel = document.getElementById(prefix + '-metrics-panel');
    if (panel) panel.style.display = 'none';
    if (typeof Chart === 'undefined') return;
    cfg.keys.forEach((key, idx) => {
      const canvas = document.getElementById(chartCanvasId(prefix, key, idx));
      if (!canvas) return;
      const ch = Chart.getChart(canvas);
      if (ch) ch.destroy();
    });
  }

  function resolveTrainLoggingDir() { return resolveLoggingDir('train'); }

  function startLogStream(key, logFile, textareaId) {
    stopLogStream(key);
    logStopFlags[key] = false;
    const ta = document.getElementById(textareaId);
    if (ta) { ta.style.display = ''; ta.value = ''; }
    if (key === 'train') parseTrainProgress('');   // reset progress on new stream
    if (key === 'rlhf' || key === 'grpo') parseTaskProgress(key, '');
    if (key === 'export' || key === 'eval' || key === 'sample') parseTaskProgress(key, '');
    const url = `${apiBase}/api/v1/log?path=${encodeURIComponent(logFile)}`;
    const es = new EventSource(url);
    logSources[key] = es;
    es.onmessage = (e) => {
      if (logStopFlags[key]) { es.close(); return; }
      try {
        const data = JSON.parse(e.data);
        if (ta) {
          ta.value = data.text || '';
          ta.scrollTop = ta.scrollHeight;
        }
        if (key === 'train') parseTrainProgress(data.text || '');
        if (key === 'rlhf' || key === 'grpo') parseTaskProgress(key, data.text || '');
        if (key === 'export' || key === 'eval' || key === 'sample') parseTaskProgress(key, data.text || '');
      } catch (_) {}
    };
    es.onerror = () => { es.close(); };
    if (key === 'train' || key === 'rlhf' || key === 'grpo') startMetricsPoll(key);
  }

  // ── Parse tqdm progress line from training log ──
  // Only match Swift/Transformers training phases: Train, Eval, Predict, Sanity
  const _TRAIN_PHASE_RE = /^(?:Train|Eval(?:uation)?|Predict|Sanity)$/i;
  // elapsed is always HH:MM once started; remaining may be "?" at step 0
  const _TQDM_RE = /([\w/]+):\s+(\d+)%\|[^|]*\|\s+(\d+)\/(\d+)\s+\[(\d+:\d+)<([^,\]]+),\s*([^\]]+)\]/g;

  function _localizeSpeed(speed, i18n) {
    const units = (i18n && i18n.speedUnits) || {};
    // speed looks like "61.18s/it" or "1.23it/s" — replace only the trailing unit
    return speed.replace(/(s\/it|it\/s|ms\/it|min\/it|it\/min)/, u => units[u] || u);
  }

  function _progressTimeText(elapsed, remain, speed, i18n) {
    const elapsedLabel = (i18n && i18n.progressElapsed) || '已用时间';
    const etaLabel     = (i18n && i18n.progressEta)     || '预计剩余时间';
    const speedLabel   = (i18n && i18n.progressSpeed)   || '速度';
    return `⏱ ${elapsedLabel} ${elapsed}  ·  ${etaLabel} ${remain}  ·  ${speedLabel} ${_localizeSpeed(speed, i18n)}`;
  }

  function parseTrainProgress(logText) {
    const el = document.getElementById('train-progress');
    if (!el) return;
    if (!logText) { el.style.display = 'none'; return; }
    // Match tqdm line: Phase:  XX%|bar| cur/total [elapsed<remaining, speed]
    _TQDM_RE.lastIndex = 0;
    let m, last;
    while ((m = _TQDM_RE.exec(logText)) !== null) {
      if (_TRAIN_PHASE_RE.test(m[1])) last = m;
    }
    if (!last) { el.style.display = 'none'; return; }
    const [, phase, pct, cur, total, elapsed, remain, speed] = last;
    el.style.display = '';
    const phaseEl = document.getElementById('train-progress-phase');
    const stepsEl = document.getElementById('train-progress-steps');
    const pctEl   = document.getElementById('train-progress-pct');
    const fillEl  = document.getElementById('train-progress-fill');
    const timeEl  = document.getElementById('train-progress-time');
    if (phaseEl) phaseEl.textContent = phase;
    if (stepsEl) stepsEl.textContent = `${cur} / ${total}`;
    if (pctEl)   pctEl.textContent   = pct + '%';
    if (fillEl)  fillEl.style.width  = pct + '%';
    if (timeEl) {
      const i18n = (window.i18n && window.i18n[pageLang]) || {};
      timeEl.textContent = _progressTimeText(elapsed, remain, speed, i18n);
    }
  }

  function parseTaskProgress(prefix, logText) {
    const el = document.getElementById(prefix + '-progress');
    if (!el) return;
    if (!logText) { el.style.display = 'none'; return; }
    _TQDM_RE.lastIndex = 0;
    let m, last;
    while ((m = _TQDM_RE.exec(logText)) !== null) {
      if (_TRAIN_PHASE_RE.test(m[1])) last = m;
    }
    if (!last) { el.style.display = 'none'; return; }
    const [, phase, pct, cur, total, elapsed, remain, speed] = last;
    el.style.display = '';
    const phaseEl = document.getElementById(prefix + '-progress-phase');
    const stepsEl = document.getElementById(prefix + '-progress-steps');
    const pctEl = document.getElementById(prefix + '-progress-pct');
    const fillEl = document.getElementById(prefix + '-progress-fill');
    const timeEl = document.getElementById(prefix + '-progress-time');
    if (phaseEl) phaseEl.textContent = phase;
    if (stepsEl) stepsEl.textContent = `${cur} / ${total}`;
    if (pctEl) pctEl.textContent = pct + '%';
    if (fillEl) fillEl.style.width = pct + '%';
    if (timeEl) {
      const i18n = (window.i18n && window.i18n[pageLang]) || {};
      timeEl.textContent = _progressTimeText(elapsed, remain, speed, i18n);
    }
  }

  function stopLogStream(key) {
    logStopFlags[key] = true;
    if (logSources[key]) {
      logSources[key].close();
      delete logSources[key];
    }
    if (key === 'train' || key === 'rlhf' || key === 'grpo') stopMetricsPoll(key);
    if (key === 'rlhf' || key === 'grpo') parseTaskProgress(key, '');
    if (key === 'export' || key === 'eval' || key === 'sample') parseTaskProgress(key, '');
  }

  // When a saved training record is selected, surface its log_file (if any)
  // in the same training-log textarea used for live tasks. Works for both
  // running and already-finished tasks since the SSE endpoint tails the file.
  function maybeShowRecordLog(key, textareaId, params) {
    const logFile = params && params.log_file;
    if (!logFile) return;
    stopLogStream(key);
    const logEl = document.getElementById(textareaId);
    if (logEl) {
      logEl.style.display = '';
      logEl.value = (window.i18n[pageLang] && window.i18n[pageLang].loadingLog) || '正在加载日志...';
    }
    // Seed the per-tab logging_dir so startMetricsPoll (invoked at the end of
    // startLogStream) can fetch TensorBoard scalars for this historical run.
    const ld = _absoluteLoggingDir(params && params.logging_dir, logFile);
    if (key === 'train') trainCurrentLoggingDir = ld || '';
    else if (key === 'rlhf') rlhfCurrentLoggingDir = ld || '';
    else if (key === 'grpo') grpoCurrentLoggingDir = ld || '';
    startLogStream(key, logFile, textareaId);
  }

  // ── Helper: populate running tasks select ──
  const _taskCache = {};  // selectId → { log_file: taskObj }

  async function refreshTasks(cmd, selectId, statusId, extraParams = null) {
    try {
      const cmdQuery = Array.isArray(cmd) ? cmd.join(',') : cmd;
      const params = new URLSearchParams();
      params.set('cmd', cmdQuery);
      if (extraParams) {
        Object.entries(extraParams).forEach(([k, v]) => {
          if (v != null && v !== '') params.set(k, v);
        });
      }
      const data = await apiFetch(`/api/v1/tasks?${params.toString()}`);
      const sel = document.getElementById(selectId);
      if (!sel) return;
      const prevVal = sel.value;
      sel.innerHTML = '';
      _taskCache[selectId] = {};
      if (!data.tasks || data.tasks.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = window.i18n[pageLang].noTasks;
        sel.appendChild(opt);
      } else {
        data.tasks.forEach(t => {
          const opt = document.createElement('option');
          opt.value = t.log_file || '';
          opt.textContent = `pid:${t.pid} | ${t.cmd} | ${t.create_time} | ${t.running}`;
          if (opt.value && opt.value === prevVal) opt.selected = true;
          if (t.log_file) _taskCache[selectId][t.log_file] = t;
          sel.appendChild(opt);
        });
      }
    } catch (e) {
      if (statusId) setStatus(statusId, e.message, 'var(--danger)');
    }
  }

  // ── Helper: parse --key value pairs from a shell cmdline string ──
  function parseCmdlineArgs(cmdline) {
    const multiValueKeys = new Set(['dataset', 'reward_funcs']);
    const args = {}, parts = (cmdline || '').split(/\s+/);
    for (let i = 0; i < parts.length; i++) {
      if (parts[i].startsWith('--') && i + 1 < parts.length && !parts[i + 1].startsWith('--')) {
        const key = parts[i].slice(2);
        if (multiValueKeys.has(key)) {
          const values = [];
          let j = i + 1;
          while (j < parts.length && !parts[j].startsWith('--')) {
            values.push(parts[j]);
            j++;
          }
          args[key] = values;
          i = j - 1;
        } else {
          args[key] = parts[i + 1];
          i++;
        }
      }
    }
    return args;
  }

  // ── Helper: set <select multiple> gpu-ids from a comma-separated string ──
  function setGpuIds(prefix, gpuStr) {
    const gpuIds = gpuStr.split(',');
    const sel = document.getElementById(prefix + '-gpu-ids');
    if (sel) Array.from(sel.options).forEach(o => { o.selected = gpuIds.includes(o.value); });
  }

  // ── Helper: restore tag-input dataset field from a space-separated string ──
  function setTagInputValues(prefix, valueStr) {
    const box = document.getElementById(prefix + '-dataset-box');
    if (!box) return;
    box.querySelectorAll('.tag').forEach(t => t.remove());
    const hidden = document.getElementById(prefix + '-dataset');
    if (hidden) hidden.value = '';
    const textIn = box.querySelector('.tag-text-input');
    if (!textIn) return;
    const values = Array.isArray(valueStr) ? valueStr : (valueStr || '').split(/\s+/).filter(Boolean);
    values.forEach(v => {
      textIn.value = v;
      textIn.dispatchEvent(new Event('change'));
    });
  }

  // ── Helper: restore infer form fields from a running deploy task ──
  function restoreInferFromTask(task) {
    // Prefer pre-parsed args from backend (avoids join/split round-trip issues).
    // Falls back to JS parsing if args is missing/empty.
    const args = (task.args && Object.keys(task.args).length > 0)
      ? task.args
      : parseCmdlineArgs(task.cmdline);
    if (args.model) {
      document.getElementById('infer-model').value = args.model;
      onModelChange('infer');
    }
    if (args.model_type)    document.getElementById('infer-model-type').value  = args.model_type;
    if (args.template)      document.getElementById('infer-template').value    = args.template;
    if (args.adapters)      document.getElementById('infer-adapters').value    = args.adapters;
    if (args.infer_backend) document.getElementById('infer-backend').value     = args.infer_backend;
    if (args.port)          document.getElementById('infer-port').value        = args.port;
    if (args.gpu_ids) setGpuIds('infer', args.gpu_ids);
  }

  // ── Helper: restore any tab's form fields from a running task ──
  const _FIELD_MAP = {
    train: {
      model: 'train-model', model_type: 'train-model-type', template: 'train-template',
      train_stage: 'train-stage', tuner_type: 'train-tuner-type', seed: 'train-seed',
      torch_dtype: 'train-torch-dtype', deepspeed: 'train-deepspeed',
      sequence_parallel_size: 'train-seq-parallel', learning_rate: 'train-lr',
      per_device_train_batch_size: 'train-batch-size', num_train_epochs: 'train-epochs',
      output_dir: 'train-output-dir', system: 'train-system', envs: 'train-envs',
      ddp_num: 'train-ddp-num',
      task_type: 'train-task-type', loss_type: 'train-loss-type', num_labels: 'train-num-labels',
    },
    rlhf: {
      model: 'rlhf-model', model_type: 'rlhf-model-type', template: 'rlhf-template',
      rlhf_type: 'rlhf-type', ref_model: 'rlhf-ref-model', ref_model_type: 'rlhf-ref-model-type',
      reward_model: 'rlhf-reward-model', reward_model_type: 'rlhf-reward-model-type',
      teacher_model: 'rlhf-teacher-model', teacher_model_type: 'rlhf-teacher-model-type',
      beta: 'rlhf-beta', max_completion_length: 'rlhf-max-completion-length',
      loss_scale: 'rlhf-loss-scale', lmbda: 'rlhf-lmbda', cpo_alpha: 'rlhf-cpo-alpha',
      rpo_alpha: 'rlhf-rpo-alpha', simpo_gamma: 'rlhf-simpo-gamma',
      desirable_weight: 'rlhf-desirable-weight', undesirable_weight: 'rlhf-undesirable-weight',
      tuner_type: 'rlhf-tuner-type', torch_dtype: 'rlhf-torch-dtype',
      seed: 'rlhf-seed', learning_rate: 'rlhf-lr',
      per_device_train_batch_size: 'rlhf-batch-size', num_train_epochs: 'rlhf-epochs',
      per_device_eval_batch_size: 'rlhf-eval-batch-size', eval_steps: 'rlhf-eval-steps',
      save_steps: 'rlhf-save-steps', gradient_accumulation_steps: 'rlhf-grad-accum',
      output_dir: 'rlhf-output-dir', logging_dir: 'rlhf-logging-dir', deepspeed: 'rlhf-deepspeed',
      sequence_parallel_size: 'rlhf-seq-parallel', ddp_num: 'rlhf-ddp-num',
      split_dataset_ratio: 'rlhf-split-ratio', max_length: 'rlhf-max-length',
      tuner_backend: 'rlhf-tuner-backend', weight_decay: 'rlhf-weight-decay',
      logging_steps: 'rlhf-logging-steps', lr_scheduler_type: 'rlhf-lr-scheduler',
      warmup_ratio: 'rlhf-warmup-ratio', max_steps: 'rlhf-max-steps',
      max_grad_norm: 'rlhf-max-grad-norm', attn_impl: 'rlhf-attn-impl',
      lora_rank: 'rlhf-lora-rank', lora_alpha: 'rlhf-lora-alpha',
      lora_dropout: 'rlhf-lora-dropout', lora_dtype: 'rlhf-lora-dtype',
      target_modules: 'rlhf-target-modules',
      envs: 'rlhf-envs', report_to: 'rlhf-report-to', swanlab_token: 'rlhf-swanlab-token',
      swanlab_project: 'rlhf-swanlab-project', swanlab_workspace: 'rlhf-swanlab-workspace',
      swanlab_exp_name: 'rlhf-swanlab-exp-name', swanlab_mode: 'rlhf-swanlab-mode',
    },
    grpo: {
      model: 'grpo-model', model_type: 'grpo-model-type', template: 'grpo-template',
      vllm_mode: 'grpo-vllm-mode', num_generations: 'grpo-num-generations',
      max_completion_length: 'grpo-max-completion-length', ref_model: 'grpo-ref-model',
      reward_funcs: 'grpo-reward-funcs', reward_weights: 'grpo-reward-weights',
      temperature: 'grpo-temperature', top_k: 'grpo-top-k', top_p: 'grpo-top-p',
      repetition_penalty: 'grpo-repetition-penalty',
      vllm_gpu_memory_utilization: 'grpo-vllm-gpu-memory-utilization',
      vllm_tensor_parallel_size: 'grpo-vllm-tensor-parallel-size',
      vllm_max_model_len: 'grpo-vllm-max-model-len',
      vllm_server_host: 'grpo-vllm-server-host',
      vllm_server_port: 'grpo-vllm-server-port',
      vllm_server_timeout: 'grpo-vllm-server-timeout',
      loss_type: 'grpo-loss-type', beta: 'grpo-beta', epsilon: 'grpo-epsilon',
      epsilon_high: 'grpo-epsilon-high', num_iterations: 'grpo-num-iterations',
      tuner_type: 'grpo-tuner-type', torch_dtype: 'grpo-torch-dtype', seed: 'grpo-seed',
      learning_rate: 'grpo-lr', per_device_train_batch_size: 'grpo-batch-size',
      per_device_eval_batch_size: 'grpo-eval-batch-size', eval_steps: 'grpo-eval-steps',
      save_steps: 'grpo-save-steps', gradient_accumulation_steps: 'grpo-grad-accum',
      num_train_epochs: 'grpo-epochs', output_dir: 'grpo-output-dir', logging_dir: 'grpo-logging-dir',
      deepspeed: 'grpo-deepspeed', sequence_parallel_size: 'grpo-seq-parallel',
      split_dataset_ratio: 'grpo-split-ratio', max_length: 'grpo-max-length',
      tuner_backend: 'grpo-tuner-backend', weight_decay: 'grpo-weight-decay',
      logging_steps: 'grpo-logging-steps', lr_scheduler_type: 'grpo-lr-scheduler',
      warmup_ratio: 'grpo-warmup-ratio', max_steps: 'grpo-max-steps',
      max_grad_norm: 'grpo-max-grad-norm', attn_impl: 'grpo-attn-impl',
      ddp_num: 'grpo-ddp-num', lora_rank: 'grpo-lora-rank', lora_alpha: 'grpo-lora-alpha',
      lora_dropout: 'grpo-lora-dropout', lora_dtype: 'grpo-lora-dtype',
      target_modules: 'grpo-target-modules',
      envs: 'grpo-envs', report_to: 'grpo-report-to', swanlab_token: 'grpo-swanlab-token',
      swanlab_project: 'grpo-swanlab-project', swanlab_workspace: 'grpo-swanlab-workspace',
      swanlab_exp_name: 'grpo-swanlab-exp-name', swanlab_mode: 'grpo-swanlab-mode',
    },
    export: {
      model: 'export-model', model_type: 'export-model-type', template: 'export-template',
      quant_bits: 'export-quant-bits', quant_method: 'export-quant-method',
      quant_n_samples: 'export-quant-n-samples', max_length: 'export-max-length',
      output_dir: 'export-output-dir', device_map: 'export-device-map',
    },
    eval: {
      model: 'eval-model', model_type: 'eval-model-type', template: 'eval-template',
      eval_backend: 'eval-backend', eval_dataset: 'eval-dataset', eval_limit: 'eval-limit',
      infer_backend: 'eval-infer-backend', eval_output_dir: 'eval-output-dir',
      custom_eval_config: 'eval-custom-config', eval_url: 'eval-url', api_key: 'eval-api-key',
    },
    sample: {
      model: 'sample-model', model_type: 'sample-model-type', template: 'sample-template',
      sampler_type: 'sample-sampler-type', sampler_engine: 'sample-sampler-engine',
      num_return_sequences: 'sample-num-return-seq',
      num_sampling_batch_size: 'sample-batch-size', num_sampling_batches: 'sample-batches',
      max_new_tokens: 'sample-max-tokens', output_dir: 'sample-output-dir',
    },
  };

  // Checkboxes keyed by prefix → {argKey: elementId}
  const _CHECKBOX_MAP = {
    train:  { use_ddp: 'train-use-ddp', use_liger_kernel: 'train-use-liger', use_chat_template: 'train-use-chat-template' },
    rlhf: { use_ddp: 'rlhf-use-ddp', use_liger_kernel: 'rlhf-use-liger', use_rslora: 'rlhf-use-rslora', use_dora: 'rlhf-use-dora', padding_free: 'rlhf-padding-free' },
    grpo: { use_ddp: 'grpo-use-ddp', use_liger_kernel: 'grpo-use-liger', use_rslora: 'grpo-use-rslora', use_dora: 'grpo-use-dora', padding_free: 'grpo-padding-free' },
    export: { merge_lora: 'export-merge-lora' },
  };

  // Tag-input dataset prefixes
  const _TAG_INPUT_PREFIXES = new Set(['train', 'rlhf', 'grpo', 'sample', 'export', 'eval']);

  function restoreFormFromTask(prefix, task, options = {}) {
    const skipModelMeta = Boolean(options.skipModelMeta);
    const args = (task.args && Object.keys(task.args).length > 0)
      ? task.args
      : parseCmdlineArgs(task.cmdline);

    // Text / number / single-select fields
    const map = _FIELD_MAP[prefix] || {};
    Object.entries(map).forEach(([argKey, elId]) => {
      if (args[argKey] != null && args[argKey] !== '') {
        const el = document.getElementById(elId);
        if (el) {
          if (el.multiple) {
            const values = Array.isArray(args[argKey]) ? args[argKey] : String(args[argKey]).split(/[,\s]+/).filter(Boolean);
            Array.from(el.options).forEach(o => { o.selected = values.includes(o.value); });
          } else {
            el.value = args[argKey];
          }
        }
      }
    });

    // Checkboxes
    const cbMap = _CHECKBOX_MAP[prefix] || {};
    Object.entries(cbMap).forEach(([argKey, elId]) => {
      if (args[argKey] != null) {
        const el = document.getElementById(elId);
        if (el) el.checked = args[argKey] === 'true' || args[argKey] === true;
      }
    });

    // GPU multi-select
    if (args.gpu_ids) setGpuIds(prefix, args.gpu_ids);

    // Tag-input dataset (train, rlhf, grpo, sample)
    if (_TAG_INPUT_PREFIXES.has(prefix) && args.dataset) {
      setTagInputValues(prefix, args.dataset);
    }

    // Model linkage (template / model_type / system auto-fill)
    if (args.model && !skipModelMeta) onModelChange(prefix);

    if (prefix === 'train' && task && task.cmd && (task.cmd === 'sft' || task.cmd === 'pt')) {
      const st = document.getElementById('train-stage');
      if (st) st.value = task.cmd;
    }
    if (prefix === 'train' && args.logging_dir) {
      trainCurrentLoggingDir = _absoluteLoggingDir(args.logging_dir, task.log_file);
    }
    if (prefix === 'rlhf' && args.logging_dir) {
      rlhfCurrentLoggingDir = _absoluteLoggingDir(args.logging_dir, task.log_file);
    }
    if (prefix === 'grpo' && args.logging_dir) {
      grpoCurrentLoggingDir = _absoluteLoggingDir(args.logging_dir, task.log_file);
    }
    if (prefix === 'train') updateTrainTaskTypeUI(false);
    if (prefix === 'rlhf') updateRlhfTypeUI(false);
    if (prefix === 'grpo') updateGrpoVllmModeUI(false);
    if (prefix === 'rlhf') scheduleRlhfCommandPreview();
    if (prefix === 'grpo') scheduleGrpoCommandPreview();
  }

  function updateTrainTaskTypeUI(clearIrrelevant) {
    const sel = document.getElementById('train-task-type');
    const tt = sel ? sel.value : '';
    const embed = document.getElementById('train-task-embed-fields');
    const seqcls = document.getElementById('train-task-seqcls-fields');
    if (embed) embed.style.display = tt === 'embedding' ? '' : 'none';
    if (seqcls) seqcls.style.display = tt === 'seq_cls' ? '' : 'none';
    if (clearIrrelevant) {
      const lossIn = document.getElementById('train-loss-type');
      const numIn = document.getElementById('train-num-labels');
      const chatCb = document.getElementById('train-use-chat-template');
      if (tt !== 'embedding' && lossIn) lossIn.value = '';
      if (tt !== 'seq_cls') {
        if (numIn) numIn.value = '';
        if (chatCb) chatCb.checked = false;
      }
    }
  }

  // ── Helper: kill task from select ──
  async function killSelectedTask(selectId, cmd, statusId, extraParams = null) {
    const sel = document.getElementById(selectId);
    if (!sel || !sel.value) return;
    try {
      await apiFetch(`/api/v1/tasks?log_file=${encodeURIComponent(sel.value)}`, { method: 'DELETE' });
      showToast(window.i18n[pageLang].successKillTask, 'success');
      await refreshTasks(cmd, selectId, statusId, extraParams);
      if (selectId === 'train-running-tasks') stopLogStream('train');
      if (selectId === 'rlhf-running-tasks') stopLogStream('rlhf');
      if (selectId === 'grpo-running-tasks') stopLogStream('grpo');
      if (selectId === 'infer-running-tasks') stopLogStream('infer');
      if (selectId === 'export-running-tasks') stopLogStream('export');
      if (selectId === 'eval-running-tasks') stopLogStream('eval');
      if (selectId === 'sample-running-tasks') stopLogStream('sample');
      if (statusId) setStatus(statusId, window.i18n[pageLang].statusIdle || '', 'var(--gray-700)');
    } catch (e) {
      if (statusId) setStatus(statusId, e.message, 'var(--danger)');
    }
  }

  function buildTrainStartBody(dryRun) {
    return {
      model:                       val('train-model') || '',
      model_type:                  val('train-model-type'),
      template:                    val('train-template'),
      dataset:                     datasetList('train-dataset'),
      train_stage:                 val('train-stage') || 'sft',
      tuner_type:                  val('train-tuner-type'),
      seed:                        numVal('train-seed'),
      torch_dtype:                 val('train-torch-dtype'),
      use_liger_kernel:            document.getElementById('train-use-liger').checked,
      gpu_ids:                     gpuList('train-gpu-ids'),
      use_ddp:                     document.getElementById('train-use-ddp').checked,
      ddp_num:                     numVal('train-ddp-num') || 1,
      deepspeed:                   val('train-deepspeed'),
      sequence_parallel_size:      numVal('train-seq-parallel'),
      learning_rate:               val('train-lr') ? parseFloat(val('train-lr')) : null,
      per_device_train_batch_size: numVal('train-batch-size'),
      per_device_eval_batch_size:  numVal('train-eval-batch-size'),
      num_train_epochs:            numVal('train-epochs'),
      eval_steps:                  numVal('train-eval-steps'),
      save_steps:                  numVal('train-save-steps'),
      gradient_accumulation_steps: numVal('train-grad-accum'),
      attn_impl:                   val('train-attn-impl'),
      neftune_noise_alpha:         numVal('train-neftune-alpha'),
      output_dir:                  val('train-output-dir'),
      logging_dir:                 val('train-logging-dir'),
      system:                      val('train-system'),
      envs:                        val('train-envs'),
      dry_run:                     dryRun,
      more_params:                 val('train-more-params'),
      split_dataset_ratio:         numVal('train-split-ratio'),
      max_length:                  numVal('train-max-length'),
      padding_free:                document.getElementById('train-padding-free').checked,
      tuner_backend:               val('train-tuner-backend'),
      weight_decay:                numVal('train-weight-decay'),
      logging_steps:               numVal('train-logging-steps'),
      lr_scheduler_type:           val('train-lr-scheduler'),
      warmup_ratio:                numVal('train-warmup-ratio'),
      truncation_strategy:         val('train-truncation'),
      max_steps:                   numVal('train-max-steps'),
      max_grad_norm:               numVal('train-max-grad-norm'),
      lora_rank:                   numVal('train-lora-rank'),
      lora_alpha:                  numVal('train-lora-alpha'),
      lora_dropout:                numVal('train-lora-dropout'),
      lora_dtype:                  val('train-lora-dtype'),
      use_rslora:                  document.getElementById('train-use-rslora').checked,
      use_dora:                    document.getElementById('train-use-dora').checked,
      target_modules:              val('train-target-modules'),
      task_type:                   val('train-task-type'),
      loss_type:                   (() => {
        const tt = val('train-task-type');
        return tt === 'embedding' ? val('train-loss-type') : null;
      })(),
      num_labels:                  (() => {
        const tt = val('train-task-type');
        return tt === 'seq_cls' ? numVal('train-num-labels') : null;
      })(),
      use_chat_template:           (() => {
        const tt = val('train-task-type');
        if (tt !== 'seq_cls') return null;
        const cb = document.getElementById('train-use-chat-template');
        return cb ? cb.checked : null;
      })(),
      model_name:                  val('train-model-name'),
      model_author:                val('train-model-author'),
      push_to_hub:                 document.getElementById('train-push-to-hub').checked,
      hub_model_id:                val('train-hub-model-id'),
      hub_private_repo:            document.getElementById('train-hub-private').checked,
      hub_strategy:                val('train-hub-strategy'),
      hub_token:                   val('train-hub-token'),
      report_to:                   val('train-report-to'),
      swanlab_token:               val('train-swanlab-token'),
      swanlab_project:             val('train-swanlab-project'),
      swanlab_workspace:           val('train-swanlab-workspace'),
      swanlab_exp_name:            val('train-swanlab-exp-name'),
      swanlab_mode:                val('train-swanlab-mode'),
    };
  }

  let trainCmdPreviewGen = 0;
  let trainCmdPreviewTimer = null;
  function scheduleTrainCommandPreview() {
    if (trainCmdPreviewTimer) clearTimeout(trainCmdPreviewTimer);
    trainCmdPreviewTimer = setTimeout(() => {
      trainCmdPreviewTimer = null;
      runTrainCommandPreview();
    }, 450);
  }
  async function runTrainCommandPreview() {
    const gen = ++trainCmdPreviewGen;
    try {
      const result = await apiFetch('/api/v1/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildTrainStartBody(true)),
      });
      if (gen !== trainCmdPreviewGen) return;
      const cmdEl = document.getElementById('train-cmd');
      if (cmdEl) cmdEl.value = result.command || '';
    } catch (_) {
      if (gen !== trainCmdPreviewGen) return;
    }
  }

  // ══════════════════════════════════════════════════════════════
  //  TAB: TRAIN
  // ══════════════════════════════════════════════════════════════
  document.getElementById('train-btn-start').addEventListener('click', async () => {
    const status = 'train-status';
    // Pre-run validation
    clearTrainError();
    const trainModel = val('train-model');
    if (!trainModel || !trainModel.trim()) {
      showTrainError(window.i18n[pageLang].errMissingModel);
      return;
    }
    const trainDatasets = datasetList('train-dataset');
    if (!trainDatasets || trainDatasets.length === 0) {
      showTrainError(window.i18n[pageLang].errMissingDataset);
      return;
    }
    trainCmdPreviewGen++;
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    const startBtn = document.getElementById('train-btn-start');
    setBtnLoading(startBtn, true);
    const body = buildTrainStartBody(false);
    try {
      const result = await apiFetch('/api/v1/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const cmdEl = document.getElementById('train-cmd');
      if (cmdEl) cmdEl.value = result.command || '';
      clearTrainError();
      trainCurrentLoggingDir = _absoluteLoggingDir(result.logging_dir, result.log_file) || '';
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(async () => {
        await refreshTasks(['sft', 'pt'], 'train-running-tasks', status);
        const sel = document.getElementById('train-running-tasks');
        if (sel && sel.options.length > 0 && sel.options[0].value && !sel.value) {
          sel.selectedIndex = 0;
        }
        updateTrainKillBtn();
      }, 1500);
      if (result.log_file) startLogStream('train', result.log_file, 'train-log');
      loadTrainRecords(val('train-model'));
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(startBtn, false);
    }
  });

  (function wireTrainCommandPreview() {
    const section = document.getElementById('section-train');
    if (!section) return;
    function shouldIgnoreTarget(t) {
      if (!t || !t.closest) return true;
      if (t.closest('#train-log') || t.closest('#train-cmd')) return true;
      return false;
    }
    section.addEventListener('input', (e) => {
      if (shouldIgnoreTarget(e.target)) return;
      scheduleTrainCommandPreview();
    });
    section.addEventListener('change', (e) => {
      if (shouldIgnoreTarget(e.target)) return;
      scheduleTrainCommandPreview();
    });
    scheduleTrainCommandPreview();
  })();

  document.getElementById('train-task-type').addEventListener('change', () => {
    updateTrainTaskTypeUI(true);
    scheduleTrainCommandPreview();
  });

  (function wireAccTrainOtherTabs() {
    const bar = document.getElementById('acc-train-other-tabs');
    if (!bar) return;
    bar.addEventListener('click', (e) => {
      const btn = e.target.closest('.tab-btn');
      if (!btn || !bar.contains(btn)) return;
      bar.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const panelId = btn.getAttribute('data-tab');
      document.querySelectorAll('#acc-train-other .tab-panel').forEach(p => {
        p.classList.toggle('active', p.id === panelId);
      });
    });
  })();

  document.getElementById('train-btn-refresh').addEventListener('click',
    async () => { await refreshTasks(['sft', 'pt'], 'train-running-tasks', 'train-status'); updateTrainKillBtn(); });
  document.getElementById('train-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('train-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('train-running-tasks', ['sft', 'pt'], 'train-status');
    setBtnLoading(btn, false);
    updateTrainKillBtn();
  });

  let trainTbOpen = false;
  document.getElementById('train-btn-tb').addEventListener('click', async () => {
    const status = 'train-status';
    const btn = document.getElementById('train-btn-tb');
    const i18n = window.i18n[pageLang];
    if (!trainTbOpen) {
      const ld = resolveTrainLoggingDir();
      if (!ld) {
        setStatus(status, i18n.tensorBoardNeedLoggingDir, 'var(--danger)');
        return;
      }
      setBtnLoading(btn, true);
      try {
        const r = await apiFetch('/api/v1/tensorboard/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ logging_dir: ld }),
        });
        const abs = buildTensorBoardUrl(r.url);
        window.open(abs, '_blank', 'noopener');
        trainTbOpen = true;
        const urlEl = document.getElementById('train-tb-url');
        if (urlEl) {
          urlEl.href = abs;
          urlEl.textContent = abs;
          urlEl.style.display = '';
        }
        btn.setAttribute('data-i18n', 'btnTensorBoardClose');
        btn.textContent = i18n.btnTensorBoardClose || '关闭 TensorBoard';
      } catch (e) {
        setStatus(status, e.message, 'var(--danger)');
      } finally {
        setBtnLoading(btn, false);
      }
    } else {
      setBtnLoading(btn, true);
      try {
        await apiFetch('/api/v1/tensorboard/stop', { method: 'POST' });
        trainTbOpen = false;
        const urlEl = document.getElementById('train-tb-url');
        if (urlEl) {
          urlEl.removeAttribute('href');
          urlEl.textContent = '';
          urlEl.style.display = 'none';
        }
        btn.setAttribute('data-i18n', 'btnTensorBoardOpen');
        btn.textContent = i18n.btnTensorBoardOpen || '打开 TensorBoard';
      } catch (e) {
        setStatus(status, e.message, 'var(--danger)');
      } finally {
        setBtnLoading(btn, false);
      }
    }
  });

  const tbOpenState = { rlhf: false, grpo: false };
  function wireTbToggle(prefix, statusId) {
    const btn = document.getElementById(prefix + '-btn-tb');
    if (!btn) return;
    btn.addEventListener('click', async () => {
      const i18n = window.i18n[pageLang];
      const isOpen = tbOpenState[prefix];
      if (!isOpen) {
        const ld = resolveLoggingDir(prefix);
        if (!ld) {
          setStatus(statusId, i18n.tensorBoardNeedLoggingDir, 'var(--danger)');
          return;
        }
        setBtnLoading(btn, true);
        try {
          const r = await apiFetch('/api/v1/tensorboard/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ logging_dir: ld }),
          });
          const abs = buildTensorBoardUrl(r.url);
          window.open(abs, '_blank', 'noopener');
          tbOpenState[prefix] = true;
          const urlEl = document.getElementById(prefix + '-tb-url');
          if (urlEl) {
            urlEl.href = abs;
            urlEl.textContent = abs;
            urlEl.style.display = '';
          }
          btn.setAttribute('data-i18n', 'btnTensorBoardClose');
          btn.textContent = i18n.btnTensorBoardClose || '关闭 TensorBoard';
        } catch (e) {
          setStatus(statusId, e.message, 'var(--danger)');
        } finally {
          setBtnLoading(btn, false);
        }
      } else {
        setBtnLoading(btn, true);
        try {
          await apiFetch('/api/v1/tensorboard/stop', { method: 'POST' });
          tbOpenState[prefix] = false;
          const urlEl = document.getElementById(prefix + '-tb-url');
          if (urlEl) {
            urlEl.removeAttribute('href');
            urlEl.textContent = '';
            urlEl.style.display = 'none';
          }
          btn.setAttribute('data-i18n', 'btnTensorBoardOpen');
          btn.textContent = i18n.btnTensorBoardOpen || '打开 TensorBoard';
        } catch (e) {
          setStatus(statusId, e.message, 'var(--danger)');
        } finally {
          setBtnLoading(btn, false);
        }
      }
    });
  }

  function wireSimpleTabBar(barId, scopeSelector) {
    const bar = document.getElementById(barId);
    if (!bar) return;
    bar.addEventListener('click', (e) => {
      const btn = e.target.closest('.tab-btn');
      if (!btn || !bar.contains(btn)) return;
      const panelId = btn.getAttribute('data-tab');
      bar.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
      document.querySelectorAll(`${scopeSelector} .tab-panel`).forEach(panel => {
        panel.classList.toggle('active', panel.id === panelId);
      });
    });
  }

  function updateRlhfTaskButtons() {
    const sel = document.getElementById('rlhf-running-tasks');
    const refreshBtn = document.getElementById('rlhf-btn-refresh');
    const killBtn = document.getElementById('rlhf-btn-kill');
    const enabled = !!(sel && sel.value);
    if (refreshBtn) refreshBtn.disabled = !enabled;
    if (killBtn) killBtn.disabled = !enabled;
  }

  function updateGrpoTaskButtons() {
    const sel = document.getElementById('grpo-running-tasks');
    const refreshBtn = document.getElementById('grpo-btn-refresh');
    const killBtn = document.getElementById('grpo-btn-kill');
    const enabled = !!(sel && sel.value);
    if (refreshBtn) refreshBtn.disabled = !enabled;
    if (killBtn) killBtn.disabled = !enabled;
  }

  function updateTaskButtons(prefix) {
    const sel = document.getElementById(prefix + '-running-tasks');
    const refreshBtn = document.getElementById(prefix + '-btn-refresh');
    const killBtn = document.getElementById(prefix + '-btn-kill');
    const enabled = !!(sel && sel.value);
    if (refreshBtn) refreshBtn.disabled = !enabled;
    if (killBtn) killBtn.disabled = !enabled;
  }

  function updateRlhfTypeUI(clearIrrelevant) {
    const type = val('rlhf-type') || 'dpo';
    const visibleByType = {
      dpo: ['rlhf-beta', 'rlhf-rpo-alpha', 'rlhf-ref-model', 'rlhf-ref-model-type'],
      orpo: ['rlhf-beta'],
      simpo: ['rlhf-beta', 'rlhf-simpo-gamma', 'rlhf-cpo-alpha'],
      kto: ['rlhf-beta', 'rlhf-desirable-weight', 'rlhf-undesirable-weight', 'rlhf-ref-model', 'rlhf-ref-model-type'],
      cpo: ['rlhf-beta', 'rlhf-cpo-alpha'],
      rm: ['rlhf-beta'],
      ppo: ['rlhf-beta', 'rlhf-reward-model', 'rlhf-reward-model-type', 'rlhf-max-completion-length', 'rlhf-ref-model', 'rlhf-ref-model-type'],
      gkd: ['rlhf-beta', 'rlhf-teacher-model', 'rlhf-teacher-model-type', 'rlhf-max-completion-length', 'rlhf-lmbda'],
    };
    const betaDefaults = { simpo: '2', gkd: '0.5' };
    const allIds = [
      'rlhf-beta', 'rlhf-max-completion-length', 'rlhf-loss-scale', 'rlhf-ref-model', 'rlhf-ref-model-type',
      'rlhf-reward-model', 'rlhf-reward-model-type', 'rlhf-teacher-model', 'rlhf-teacher-model-type',
      'rlhf-rpo-alpha', 'rlhf-lmbda', 'rlhf-simpo-gamma', 'rlhf-cpo-alpha', 'rlhf-desirable-weight', 'rlhf-undesirable-weight',
    ];
    const visible = new Set(visibleByType[type] || ['rlhf-beta']);
    allIds.forEach(id => {
      const el = document.getElementById(id);
      const label = el && el.closest('label');
      if (!el || !label) return;
      const show = visible.has(id) || id === 'rlhf-loss-scale';
      label.style.display = show ? '' : 'none';
      if (clearIrrelevant && !show) el.value = '';
    });
    if (clearIrrelevant) {
      const betaEl = document.getElementById('rlhf-beta');
      if (betaEl) betaEl.value = betaDefaults[type] || '0.1';
    }
  }

  function updateGrpoVllmModeUI(clearIrrelevant) {
    const mode = val('grpo-vllm-mode') || 'colocate';
    const colocate = document.getElementById('grpo-vllm-colocate-fields');
    const server = document.getElementById('grpo-vllm-server-fields');
    if (colocate) colocate.style.display = mode === 'colocate' ? '' : 'none';
    if (server) server.style.display = mode === 'server' ? '' : 'none';
    if (clearIrrelevant) {
      if (mode !== 'colocate') {
        ['grpo-vllm-gpu-memory-utilization', 'grpo-vllm-tensor-parallel-size', 'grpo-vllm-max-model-len'].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.value = '';
        });
      }
      if (mode !== 'server') {
        ['grpo-vllm-server-host', 'grpo-vllm-server-port', 'grpo-vllm-server-timeout'].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.value = '';
        });
      }
    }
  }

  function buildRlhfStartBody(dryRun) {
    return {
      model: val('rlhf-model') || '',
      model_type: val('rlhf-model-type'),
      template: val('rlhf-template'),
      dataset: datasetList('rlhf-dataset'),
      tuner_type: val('rlhf-tuner-type'),
      torch_dtype: val('rlhf-torch-dtype'),
      seed: numVal('rlhf-seed'),
      learning_rate: val('rlhf-lr') ? parseFloat(val('rlhf-lr')) : null,
      per_device_train_batch_size: numVal('rlhf-batch-size'),
      num_train_epochs: numVal('rlhf-epochs'),
      output_dir: val('rlhf-output-dir'),
      logging_dir: val('rlhf-logging-dir'),
      deepspeed: val('rlhf-deepspeed'),
      gpu_ids: gpuList('rlhf-gpu-ids'),
      split_dataset_ratio: val('rlhf-split-ratio') ? parseFloat(val('rlhf-split-ratio')) : null,
      max_length: numVal('rlhf-max-length'),
      padding_free: document.getElementById('rlhf-padding-free').checked,
      use_liger_kernel: document.getElementById('rlhf-use-liger').checked,
      use_ddp: document.getElementById('rlhf-use-ddp').checked,
      ddp_num: numVal('rlhf-ddp-num') || 1,
      sequence_parallel_size: numVal('rlhf-seq-parallel'),
      tuner_backend: val('rlhf-tuner-backend'),
      weight_decay: val('rlhf-weight-decay') ? parseFloat(val('rlhf-weight-decay')) : null,
      logging_steps: numVal('rlhf-logging-steps'),
      lr_scheduler_type: val('rlhf-lr-scheduler'),
      warmup_ratio: val('rlhf-warmup-ratio') ? parseFloat(val('rlhf-warmup-ratio')) : null,
      max_steps: numVal('rlhf-max-steps'),
      max_grad_norm: val('rlhf-max-grad-norm') ? parseFloat(val('rlhf-max-grad-norm')) : null,
      eval_steps: numVal('rlhf-eval-steps'),
      save_steps: numVal('rlhf-save-steps'),
      gradient_accumulation_steps: numVal('rlhf-grad-accum'),
      per_device_eval_batch_size: numVal('rlhf-eval-batch-size'),
      attn_impl: val('rlhf-attn-impl'),
      lora_rank: numVal('rlhf-lora-rank'),
      lora_alpha: numVal('rlhf-lora-alpha'),
      lora_dropout: numVal('rlhf-lora-dropout'),
      lora_dtype: val('rlhf-lora-dtype'),
      target_modules: val('rlhf-target-modules'),
      use_rslora: document.getElementById('rlhf-use-rslora').checked,
      use_dora: document.getElementById('rlhf-use-dora').checked,
      rlhf_type: val('rlhf-type'),
      ref_model: val('rlhf-ref-model'),
      ref_model_type: val('rlhf-ref-model-type'),
      reward_model: val('rlhf-reward-model'),
      reward_model_type: val('rlhf-reward-model-type'),
      teacher_model: val('rlhf-teacher-model'),
      teacher_model_type: val('rlhf-teacher-model-type'),
      beta: numVal('rlhf-beta'),
      max_completion_length: numVal('rlhf-max-completion-length'),
      loss_scale: val('rlhf-loss-scale'),
      lmbda: numVal('rlhf-lmbda'),
      cpo_alpha: numVal('rlhf-cpo-alpha'),
      rpo_alpha: numVal('rlhf-rpo-alpha'),
      simpo_gamma: numVal('rlhf-simpo-gamma'),
      desirable_weight: numVal('rlhf-desirable-weight'),
      undesirable_weight: numVal('rlhf-undesirable-weight'),
      system: val('rlhf-system'),
      envs: val('rlhf-envs'),
      report_to: val('rlhf-report-to'),
      swanlab_token: val('rlhf-swanlab-token'),
      swanlab_project: val('rlhf-swanlab-project'),
      swanlab_workspace: val('rlhf-swanlab-workspace'),
      swanlab_exp_name: val('rlhf-swanlab-exp-name'),
      swanlab_mode: val('rlhf-swanlab-mode'),
      dry_run: dryRun,
      more_params: val('rlhf-more-params'),
    };
  }

  function buildGrpoStartBody(dryRun) {
    return {
      model: val('grpo-model') || '',
      model_type: val('grpo-model-type'),
      template: val('grpo-template'),
      dataset: datasetList('grpo-dataset'),
      tuner_type: val('grpo-tuner-type'),
      torch_dtype: val('grpo-torch-dtype'),
      seed: numVal('grpo-seed'),
      learning_rate: val('grpo-lr') ? parseFloat(val('grpo-lr')) : null,
      per_device_train_batch_size: numVal('grpo-batch-size'),
      num_train_epochs: numVal('grpo-epochs'),
      output_dir: val('grpo-output-dir'),
      logging_dir: val('grpo-logging-dir'),
      deepspeed: val('grpo-deepspeed'),
      gpu_ids: gpuList('grpo-gpu-ids'),
      split_dataset_ratio: val('grpo-split-ratio') ? parseFloat(val('grpo-split-ratio')) : null,
      max_length: numVal('grpo-max-length'),
      padding_free: document.getElementById('grpo-padding-free').checked,
      use_liger_kernel: document.getElementById('grpo-use-liger').checked,
      use_ddp: document.getElementById('grpo-use-ddp').checked,
      ddp_num: numVal('grpo-ddp-num') || 1,
      sequence_parallel_size: numVal('grpo-seq-parallel'),
      tuner_backend: val('grpo-tuner-backend'),
      weight_decay: val('grpo-weight-decay') ? parseFloat(val('grpo-weight-decay')) : null,
      logging_steps: numVal('grpo-logging-steps'),
      lr_scheduler_type: val('grpo-lr-scheduler'),
      warmup_ratio: val('grpo-warmup-ratio') ? parseFloat(val('grpo-warmup-ratio')) : null,
      max_steps: numVal('grpo-max-steps'),
      max_grad_norm: val('grpo-max-grad-norm') ? parseFloat(val('grpo-max-grad-norm')) : null,
      eval_steps: numVal('grpo-eval-steps'),
      save_steps: numVal('grpo-save-steps'),
      gradient_accumulation_steps: numVal('grpo-grad-accum'),
      per_device_eval_batch_size: numVal('grpo-eval-batch-size'),
      attn_impl: val('grpo-attn-impl'),
      lora_rank: numVal('grpo-lora-rank'),
      lora_alpha: numVal('grpo-lora-alpha'),
      lora_dropout: numVal('grpo-lora-dropout'),
      lora_dtype: val('grpo-lora-dtype'),
      target_modules: val('grpo-target-modules'),
      use_rslora: document.getElementById('grpo-use-rslora').checked,
      use_dora: document.getElementById('grpo-use-dora').checked,
      vllm_mode: val('grpo-vllm-mode'),
      num_generations: numVal('grpo-num-generations'),
      max_completion_length: numVal('grpo-max-completion-length'),
      reward_funcs: multiSelectList('grpo-reward-funcs'),
      reward_weights: val('grpo-reward-weights'),
      ref_model: val('grpo-ref-model'),
      temperature: numVal('grpo-temperature'),
      top_k: numVal('grpo-top-k'),
      top_p: val('grpo-top-p') ? parseFloat(val('grpo-top-p')) : null,
      repetition_penalty: val('grpo-repetition-penalty') ? parseFloat(val('grpo-repetition-penalty')) : null,
      vllm_gpu_memory_utilization: val('grpo-vllm-gpu-memory-utilization'),
      vllm_tensor_parallel_size: numVal('grpo-vllm-tensor-parallel-size'),
      vllm_max_model_len: numVal('grpo-vllm-max-model-len'),
      vllm_server_host: val('grpo-vllm-server-host'),
      vllm_server_port: numVal('grpo-vllm-server-port'),
      vllm_server_timeout: numVal('grpo-vllm-server-timeout'),
      loss_type: val('grpo-loss-type'),
      beta: val('grpo-beta') ? parseFloat(val('grpo-beta')) : null,
      epsilon: val('grpo-epsilon') ? parseFloat(val('grpo-epsilon')) : null,
      epsilon_high: val('grpo-epsilon-high') ? parseFloat(val('grpo-epsilon-high')) : null,
      num_iterations: numVal('grpo-num-iterations'),
      system: val('grpo-system'),
      envs: val('grpo-envs'),
      report_to: val('grpo-report-to'),
      swanlab_token: val('grpo-swanlab-token'),
      swanlab_project: val('grpo-swanlab-project'),
      swanlab_workspace: val('grpo-swanlab-workspace'),
      swanlab_exp_name: val('grpo-swanlab-exp-name'),
      swanlab_mode: val('grpo-swanlab-mode'),
      dry_run: dryRun,
      more_params: val('grpo-more-params'),
    };
  }

  let rlhfPreviewGen = 0, rlhfPreviewTimer = null;
  function scheduleRlhfCommandPreview() {
    if (rlhfPreviewTimer) clearTimeout(rlhfPreviewTimer);
    rlhfPreviewTimer = setTimeout(() => { rlhfPreviewTimer = null; runRlhfCommandPreview(); }, 450);
  }
  async function runRlhfCommandPreview() {
    const gen = ++rlhfPreviewGen;
    try {
      const result = await apiFetch('/api/v1/rlhf/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildRlhfStartBody(true)),
      });
      if (gen !== rlhfPreviewGen) return;
      const el = document.getElementById('rlhf-cmd');
      if (el) el.value = result.command || '';
    } catch (_) {}
  }

  let grpoPreviewGen = 0, grpoPreviewTimer = null;
  function scheduleGrpoCommandPreview() {
    if (grpoPreviewTimer) clearTimeout(grpoPreviewTimer);
    grpoPreviewTimer = setTimeout(() => { grpoPreviewTimer = null; runGrpoCommandPreview(); }, 450);
  }
  async function runGrpoCommandPreview() {
    const gen = ++grpoPreviewGen;
    try {
      const result = await apiFetch('/api/v1/grpo/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(buildGrpoStartBody(true)),
      });
      if (gen !== grpoPreviewGen) return;
      const el = document.getElementById('grpo-cmd');
      if (el) el.value = result.command || '';
    } catch (_) {}
  }

  async function refreshRlhfTasks() {
    await refreshTasks('rlhf', 'rlhf-running-tasks', 'rlhf-status', { rlhf_mode: 'non_grpo' });
    const sel = document.getElementById('rlhf-running-tasks');
    if (sel && !sel.value && sel.options.length > 0 && sel.options[0].value) {
      sel.selectedIndex = 0;
      sel.dispatchEvent(new Event('change'));
    }
    updateRlhfTaskButtons();
  }

  async function refreshGrpoTasks() {
    await refreshTasks('rlhf', 'grpo-running-tasks', 'grpo-status', { rlhf_mode: 'grpo' });
    const sel = document.getElementById('grpo-running-tasks');
    if (sel && !sel.value && sel.options.length > 0 && sel.options[0].value) {
      sel.selectedIndex = 0;
      sel.dispatchEvent(new Event('change'));
    }
    updateGrpoTaskButtons();
  }

  // ══════════════════════════════════════════════════════════════
  //  TAB: RLHF
  // ══════════════════════════════════════════════════════════════
  document.getElementById('rlhf-btn-start').addEventListener('click', async () => {
    const status = 'rlhf-status';
    const startBtn = document.getElementById('rlhf-btn-start');
    const model = val('rlhf-model');
    const datasets = datasetList('rlhf-dataset');
    if (!model || !model.trim()) {
      showToast(window.i18n[pageLang].errMissingModel, 'error');
      return;
    }
    if (!datasets || datasets.length === 0) {
      showToast(window.i18n[pageLang].errMissingDataset, 'error');
      return;
    }
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    const body = buildRlhfStartBody(false);
    setBtnLoading(startBtn, true);
    try {
      const result = await apiFetch('/api/v1/rlhf/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const cmdEl = document.getElementById('rlhf-cmd');
      if (cmdEl) cmdEl.value = result.command || '';
      rlhfCurrentLoggingDir = _absoluteLoggingDir(result.logging_dir, result.log_file) || rlhfCurrentLoggingDir;
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(() => refreshRlhfTasks(), 1500);
      if (result.log_file) startLogStream('rlhf', result.log_file, 'rlhf-log');
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(startBtn, false);
    }
  });

  (function wireRlhfPreview() {
    const section = document.getElementById('section-rlhf');
    if (!section) return;
    section.addEventListener('input', (e) => {
      if (e.target.closest('#rlhf-log') || e.target.closest('#rlhf-cmd')) return;
      scheduleRlhfCommandPreview();
    });
    section.addEventListener('change', (e) => {
      if (e.target.closest('#rlhf-log') || e.target.closest('#rlhf-cmd')) return;
      scheduleRlhfCommandPreview();
    });
    scheduleRlhfCommandPreview();
  })();

  document.getElementById('rlhf-type').addEventListener('change', () => {
    updateRlhfTypeUI(true);
    scheduleRlhfCommandPreview();
    if (typeof Chart !== 'undefined') {
      for (let i = 0; i < 5; i++) {
        const c = document.getElementById('rlhf-chart-' + i);
        if (c) { const ch = Chart.getChart(c); if (ch) ch.destroy(); }
      }
    }
    syncRlhfMetricsKeys();
    if (metricsTimers.rlhf) fetchMetrics('rlhf');
  });
  document.getElementById('rlhf-btn-refresh').addEventListener('click', refreshRlhfTasks);
  document.getElementById('rlhf-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('rlhf-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('rlhf-running-tasks', 'rlhf', 'rlhf-status', { rlhf_mode: 'non_grpo' });
    setBtnLoading(btn, false);
    updateRlhfTaskButtons();
  });
  document.getElementById('rlhf-running-tasks').addEventListener('change', function () {
    updateRlhfTaskButtons();
    stopLogStream('rlhf');
    if (this.value) {
      const task = (_taskCache['rlhf-running-tasks'] || {})[this.value];
      if (task) restoreFormFromTask('rlhf', task);
      const logEl = document.getElementById('rlhf-log');
      if (logEl) { logEl.style.display = ''; logEl.value = window.i18n[pageLang].loadingLog || '正在加载日志...'; }
      startLogStream('rlhf', this.value, 'rlhf-log');
    } else {
      rlhfCurrentLoggingDir = '';
      parseTaskProgress('rlhf', '');
    }
  });
  wireSimpleTabBar('acc-rlhf-other-tabs', '#acc-rlhf-other');
  wireTbToggle('rlhf', 'rlhf-status');

  // ══════════════════════════════════════════════════════════════
  //  TAB: GRPO
  // ══════════════════════════════════════════════════════════════
  document.getElementById('grpo-btn-start').addEventListener('click', async () => {
    const status = 'grpo-status';
    const startBtn = document.getElementById('grpo-btn-start');
    const model = val('grpo-model');
    const datasets = datasetList('grpo-dataset');
    if (!model || !model.trim()) {
      showToast(window.i18n[pageLang].errMissingModel, 'error');
      return;
    }
    if (!datasets || datasets.length === 0) {
      showToast(window.i18n[pageLang].errMissingDataset, 'error');
      return;
    }
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    const body = buildGrpoStartBody(false);
    setBtnLoading(startBtn, true);
    try {
      const result = await apiFetch('/api/v1/grpo/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const cmdEl = document.getElementById('grpo-cmd');
      if (cmdEl) cmdEl.value = result.command || '';
      grpoCurrentLoggingDir = _absoluteLoggingDir(result.logging_dir, result.log_file) || grpoCurrentLoggingDir;
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(() => refreshGrpoTasks(), 1500);
      if (result.log_file) startLogStream('grpo', result.log_file, 'grpo-log');
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(startBtn, false);
    }
  });

  (function wireGrpoPreview() {
    const section = document.getElementById('section-grpo');
    if (!section) return;
    section.addEventListener('input', (e) => {
      if (e.target.closest('#grpo-log') || e.target.closest('#grpo-cmd')) return;
      scheduleGrpoCommandPreview();
    });
    section.addEventListener('change', (e) => {
      if (e.target.closest('#grpo-log') || e.target.closest('#grpo-cmd')) return;
      scheduleGrpoCommandPreview();
    });
    scheduleGrpoCommandPreview();
  })();

  document.getElementById('grpo-vllm-mode').addEventListener('change', () => {
    updateGrpoVllmModeUI(true);
    scheduleGrpoCommandPreview();
  });
  document.getElementById('grpo-btn-refresh').addEventListener('click', refreshGrpoTasks);
  document.getElementById('grpo-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('grpo-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('grpo-running-tasks', 'rlhf', 'grpo-status', { rlhf_mode: 'grpo' });
    setBtnLoading(btn, false);
    updateGrpoTaskButtons();
  });
  document.getElementById('grpo-running-tasks').addEventListener('change', function () {
    updateGrpoTaskButtons();
    stopLogStream('grpo');
    if (this.value) {
      const task = (_taskCache['grpo-running-tasks'] || {})[this.value];
      if (task) restoreFormFromTask('grpo', task);
      const logEl = document.getElementById('grpo-log');
      if (logEl) { logEl.style.display = ''; logEl.value = window.i18n[pageLang].loadingLog || '正在加载日志...'; }
      startLogStream('grpo', this.value, 'grpo-log');
    } else {
      grpoCurrentLoggingDir = '';
      parseTaskProgress('grpo', '');
    }
  });
  wireSimpleTabBar('acc-grpo-other-tabs', '#acc-grpo-other');
  wireTbToggle('grpo', 'grpo-status');

  // ══════════════════════════════════════════════════════════════
  //  TAB: INFER (deploy + chat)
  // ══════════════════════════════════════════════════════════════
  document.getElementById('infer-btn-deploy').addEventListener('click', async () => {
    const status = 'infer-status';
    const deployBtn = document.getElementById('infer-btn-deploy');
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    setBtnLoading(deployBtn, true);
    const body = {
      model:        val('infer-model') || '',
      model_type:   val('infer-model-type'),
      template:     val('infer-template'),
      adapters:     val('infer-adapters'),
      merge_lora:   document.getElementById('infer-merge-lora').checked,
      infer_backend: val('infer-backend') || 'transformers',
      port:         numVal('infer-port') || 8000,
      gpu_ids:      gpuList('infer-gpu-ids'),
      more_params:  val('infer-more-params'),
    };
    try {
      const result = await apiFetch('/api/v1/infer/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(async () => {
        await refreshTasks('deploy', 'infer-running-tasks', status);
        const sel = document.getElementById('infer-running-tasks');
        if (sel && sel.options.length > 0 && sel.options[0].value && !sel.value) {
          sel.selectedIndex = 0;
          sel.dispatchEvent(new Event('change'));
        }
        updateTaskButtons('infer');
      }, 1500);
      if (result.log_file) {
        document.getElementById('acc-infer-log').classList.add('open');
        startLogStream('infer', result.log_file, 'infer-log');
      }
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(deployBtn, false);
    }
  });

  document.getElementById('infer-btn-refresh').addEventListener('click', async () => {
    await refreshTasks('deploy', 'infer-running-tasks', 'infer-status');
    updateTaskButtons('infer');
  });
  document.getElementById('infer-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('infer-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('infer-running-tasks', 'deploy', 'infer-status');
    setBtnLoading(btn, false);
    updateTaskButtons('infer');
  });

  // When the user selects a running deploy task, restore its parameters and stream the log
  document.getElementById('infer-running-tasks').addEventListener('change', function () {
    updateTaskButtons('infer');
    stopLogStream('infer');
    const logFile = this.value;
    if (!logFile) return;
    const task = (_taskCache['infer-running-tasks'] || {})[logFile];
    if (task) restoreInferFromTask(task);
    document.getElementById('acc-infer-log').classList.add('open');
    const logEl = document.getElementById('infer-log');
    if (logEl) { logEl.style.display = ''; logEl.value = window.i18n[pageLang].loadingLog || '正在加载日志...'; }
    startLogStream('infer', logFile, 'infer-log');
  });

  // ── Train task list: loading state + kill-button sync ──
  document.getElementById('train-running-tasks').addEventListener('change', function () {
    updateTrainKillBtn();
    stopLogStream('train');
    const i18n = window.i18n[pageLang];
    if (this.value) {
      const logEl = document.getElementById('train-log');
      if (logEl) { logEl.style.display = ''; logEl.value = i18n.loadingLog || '正在加载日志...'; }
      startLogStream('train', this.value, 'train-log');
    } else {
      parseTrainProgress('');  // clear progress when no task selected
    }
  });

  // When the user selects a running task on any other tab, restore form + stream log
  ['export', 'eval', 'sample'].forEach(prefix => {
    const cfg = tabCmds[prefix];
    document.getElementById(cfg[1]).addEventListener('change', function () {
      const logFile = this.value;
      updateTaskButtons(prefix);
      stopLogStream(prefix);
      if (!logFile) {
        parseTaskProgress(prefix, '');
        return;
      }
      const task = (_taskCache[cfg[1]] || {})[logFile];
      if (task) restoreFormFromTask(prefix, task, {});
      const logEl = document.getElementById(prefix + '-log');
      if (logEl) { logEl.style.display = ''; logEl.value = window.i18n[pageLang].loadingLog || '正在加载日志...'; }
      startLogStream(prefix, logFile, prefix + '-log');
    });
  });

  // Chat
  const chatHistory = []; // [{role, content}]

  function appendChatMessage(role, content) {
    const box = document.getElementById('infer-chatbot');
    // Remove placeholder
    const placeholder = box.querySelector('.muted');
    if (placeholder) placeholder.remove();

    const wrap = document.createElement('div');
    wrap.className = `chat-message ${role}`;
    const roleLabel = document.createElement('div');
    roleLabel.className = 'chat-role';
    roleLabel.textContent = role === 'user' ? 'You' : 'Assistant';
    const bubble = document.createElement('div');
    bubble.className = 'chat-bubble';
    if (role === 'assistant') {
      renderBubble(bubble, content);
    } else if (Array.isArray(content)) {
      // Multimodal user message: render each part
      content.forEach(part => {
        if (part.type === 'text') {
          const span = document.createElement('span');
          span.style.whiteSpace = 'pre-wrap';
          span.textContent = part.text;
          bubble.appendChild(span);
        } else if (part.type === 'image_url') {
          const img = document.createElement('img');
          img.src = part.image_url.url;
          img.style.cssText = 'max-width:220px;max-height:160px;border-radius:6px;display:block;margin-bottom:4px';
          bubble.insertBefore(img, bubble.firstChild);
        } else if (part.type === 'video_url') {
          const vid = document.createElement('video');
          vid.src = part.video_url.url;
          vid.controls = true;
          vid.style.cssText = 'max-width:220px;border-radius:6px;display:block;margin-bottom:4px';
          bubble.insertBefore(vid, bubble.firstChild);
        } else if (part.type === 'audio_url') {
          const aud = document.createElement('audio');
          aud.src = part.audio_url.url;
          aud.controls = true;
          aud.style.cssText = 'width:200px;display:block;margin-bottom:4px';
          bubble.insertBefore(aud, bubble.firstChild);
        }
      });
    } else {
      bubble.textContent = content;
    }
    wrap.appendChild(roleLabel);
    wrap.appendChild(bubble);
    box.appendChild(wrap);
    box.scrollTop = box.scrollHeight;
    return bubble;
  }

  // Render assistant bubble: split <think>...</think> into a collapsible block
  function renderBubble(bubble, text) {
    bubble.innerHTML = '';
    const t = window.i18n[pageLang];
    const thinkRe = /<think>([\s\S]*?)<\/think>/g;
    let last = 0, m;
    while ((m = thinkRe.exec(text)) !== null) {
      if (m.index > last) {
        const before = document.createElement('span');
        before.textContent = text.slice(last, m.index);
        bubble.appendChild(before);
      }
      const details = document.createElement('details');
      details.className = 'think-block';
      details.open = true;
      const summary = document.createElement('summary');
      summary.textContent = t.thinkTitle;
      const body = document.createElement('div');
      body.className = 'think-body';
      body.textContent = m[1];
      details.appendChild(summary);
      details.appendChild(body);
      bubble.appendChild(details);
      last = m.index + m[0].length;
    }
    // Handle an unclosed <think> tag still streaming
    const openTag = text.indexOf('<think>', last);
    if (openTag !== -1) {
      if (openTag > last) {
        const before = document.createElement('span');
        before.style.whiteSpace = 'pre-wrap';
        before.textContent = text.slice(last, openTag);
        bubble.appendChild(before);
      }
      const details = document.createElement('details');
      details.className = 'think-block';
      details.open = true;
      const summary = document.createElement('summary');
      summary.textContent = t.thinkStreaming;
      const body = document.createElement('div');
      body.className = 'think-body';
      body.textContent = text.slice(openTag + '<think>'.length);
      details.classList.add('streaming');
      details.appendChild(summary);
      details.appendChild(body);
      bubble.appendChild(details);
    } else if (last < text.length) {
      const after = document.createElement('span');
      after.style.whiteSpace = 'pre-wrap';
      after.textContent = text.slice(last);
      bubble.appendChild(after);
    }
  }

  // ── Read a File as a base64 data-URL ──
  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload  = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  document.getElementById('infer-btn-send').addEventListener('click', async () => {
    const prompt = document.getElementById('infer-prompt').value.trim();
    const port = numVal('infer-port') || 8000;

    // Detect selected media file (image / video / audio)
    const activeMedia = document.querySelector('.media-tab.active')?.getAttribute('data-media');
    const mediaInputMap = { image: 'infer-image', video: 'infer-video', audio: 'infer-audio' };
    const mediaInputId  = mediaInputMap[activeMedia];
    const mediaFile     = mediaInputId ? document.getElementById(mediaInputId)?.files[0] : null;

    if (!prompt && !mediaFile) return;

    // Build OpenAI-format content: array when media is present, plain string otherwise
    let userContent;
    let displayContent = prompt;  // text shown in the bubble label
    if (mediaFile) {
      const dataUrl = await readFileAsDataURL(mediaFile);
      const mediaType = activeMedia; // 'image' | 'video' | 'audio'
      const urlKey = `${mediaType}_url`;
      userContent = [];
      userContent.push({ type: urlKey, [urlKey]: { url: dataUrl } });
      if (prompt) userContent.push({ type: 'text', text: prompt });
      // Clear the file input and preview zone after reading
      const zone = document.querySelector(`.media-upload[data-input="${mediaInputId}"]`);
      if (zone && zone._clearPreview) zone._clearPreview();
      else document.getElementById(mediaInputId).value = '';
    } else {
      userContent = prompt;
    }

    // Display user message in chat UI
    appendChatMessage('user', userContent);
    chatHistory.push({ role: 'user', content: userContent });
    document.getElementById('infer-prompt').value = '';

    const assistantBubble = appendChatMessage('assistant', '…');

    const body = {
      port,
      model:               val('infer-model') || '',
      messages: JSON.parse(JSON.stringify(chatHistory)),
      system:              val('infer-system'),
      max_new_tokens:      numVal('infer-max-tokens') || 2048,
      temperature:         numVal('infer-temperature') || 1.0,
      top_k:               numVal('infer-top-k') || 50,
      top_p:               numVal('infer-top-p') || 0.9,
      repetition_penalty:  numVal('infer-rep-penalty') || 1.0,
      stream:              true,
      model_name:          val('infer-model-type-lora'),
    };

    try {
      const res = await fetch(`${apiBase}/api/v1/infer/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        assistantBubble.textContent = err.detail || 'Error';
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';
      let hasError = false;
      assistantBubble.textContent = '';

      outer: while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');
        for (const line of lines) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') break outer;
          try {
            const parsed = JSON.parse(data);
            if (parsed.error) {
              let msg = parsed.error;
              // unwrap nested JSON error strings from the deploy server
              if (typeof msg === 'string') {
                try { msg = JSON.parse(msg).message || msg; } catch (_) {}
              } else if (typeof msg === 'object') {
                msg = msg.message || JSON.stringify(msg);
              }
              assistantBubble.textContent = '⚠ ' + msg;
              assistantBubble.style.color = 'var(--danger)';
              hasError = true;
              // remove the pending user message so the user can retry
              chatHistory.pop();
              break outer;
            }
            if (parsed.delta) {
              accumulated += parsed.delta;
              renderBubble(assistantBubble, accumulated);
              document.getElementById('infer-chatbot').scrollTop = document.getElementById('infer-chatbot').scrollHeight;
            }
          } catch (_) {}
        }
      }
      if (!hasError) chatHistory.push({ role: 'assistant', content: accumulated });
    } catch (e) {
      assistantBubble.textContent = e.message || 'Error';
    }
  });

  // Enter sends message; Shift+Enter inserts a newline
  document.getElementById('infer-prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      document.getElementById('infer-btn-send').click();
    }
  });

  document.getElementById('infer-btn-clear').addEventListener('click', () => {
    chatHistory.length = 0;
    const box = document.getElementById('infer-chatbot');
    box.innerHTML = `<div class="muted" style="text-align:center;padding:20px" data-i18n="chatEmpty">${window.i18n[pageLang].chatEmpty}</div>`;
  });

  // ══════════════════════════════════════════════════════════════
  //  TAB: EXPORT
  // ══════════════════════════════════════════════════════════════
  document.getElementById('export-btn-start').addEventListener('click', async () => {
    const status = 'export-status';
    const startBtn = document.getElementById('export-btn-start');
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    setBtnLoading(startBtn, true);
    const body = {
      model:          val('export-model') || '',
      model_type:     val('export-model-type'),
      template:       val('export-template'),
      merge_lora:     document.getElementById('export-merge-lora').checked,
      quant_bits:     numVal('export-quant-bits'),
      quant_method:   val('export-quant-method'),
      quant_n_samples: numVal('export-quant-n-samples'),
      max_length:     numVal('export-max-length'),
      output_dir:     val('export-output-dir'),
      dataset:        datasetList('export-dataset'),
      gpu_ids:        gpuList('export-gpu-ids'),
      device_map:     val('export-device-map'),
      more_params:    val('export-more-params'),
    };
    try {
      const result = await apiFetch('/api/v1/export/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(async () => {
        await refreshTasks('export', 'export-running-tasks', status);
        const sel = document.getElementById('export-running-tasks');
        if (sel && sel.options.length > 0 && sel.options[0].value && !sel.value) {
          sel.selectedIndex = 0;
          sel.dispatchEvent(new Event('change'));
        }
        updateTaskButtons('export');
      }, 1500);
      if (result.log_file) startLogStream('export', result.log_file, 'export-log');
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(startBtn, false);
    }
  });

  document.getElementById('export-btn-refresh').addEventListener('click', async () => {
    await refreshTasks('export', 'export-running-tasks', 'export-status');
    updateTaskButtons('export');
  });
  document.getElementById('export-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('export-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('export-running-tasks', 'export', 'export-status');
    setBtnLoading(btn, false);
    updateTaskButtons('export');
  });

  // ══════════════════════════════════════════════════════════════
  //  TAB: EVAL
  // ══════════════════════════════════════════════════════════════
  document.getElementById('eval-btn-start').addEventListener('click', async () => {
    const status = 'eval-status';
    const startBtn = document.getElementById('eval-btn-start');
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    setBtnLoading(startBtn, true);
    const body = {
      model:              val('eval-model') || '',
      model_type:         val('eval-model-type'),
      template:           val('eval-template'),
      eval_backend:       val('eval-backend'),
      eval_dataset:       datasetList('eval-dataset'),
      eval_limit:         numVal('eval-limit'),
      infer_backend:      val('eval-infer-backend'),
      custom_eval_config: val('eval-custom-config'),
      eval_output_dir:    val('eval-output-dir'),
      eval_url:           val('eval-url'),
      api_key:            val('eval-api-key'),
      gpu_ids:            gpuList('eval-gpu-ids'),
      more_params:        val('eval-more-params'),
    };
    try {
      const result = await apiFetch('/api/v1/eval/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(async () => {
        await refreshTasks('eval', 'eval-running-tasks', status);
        const sel = document.getElementById('eval-running-tasks');
        if (sel && sel.options.length > 0 && sel.options[0].value && !sel.value) {
          sel.selectedIndex = 0;
          sel.dispatchEvent(new Event('change'));
        }
        updateTaskButtons('eval');
      }, 1500);
      if (result.log_file) startLogStream('eval', result.log_file, 'eval-log');
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(startBtn, false);
    }
  });

  document.getElementById('eval-btn-refresh').addEventListener('click', async () => {
    await refreshTasks('eval', 'eval-running-tasks', 'eval-status');
    updateTaskButtons('eval');
  });
  document.getElementById('eval-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('eval-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('eval-running-tasks', 'eval', 'eval-status');
    setBtnLoading(btn, false);
    updateTaskButtons('eval');
  });

  // ══════════════════════════════════════════════════════════════
  //  TAB: SAMPLE
  // ══════════════════════════════════════════════════════════════
  document.getElementById('sample-btn-start').addEventListener('click', async () => {
    const status = 'sample-status';
    const startBtn = document.getElementById('sample-btn-start');
    setStatus(status, window.i18n[pageLang].statusStarting, 'var(--brand-600)');
    setBtnLoading(startBtn, true);
    const body = {
      model:                  val('sample-model') || '',
      model_type:             val('sample-model-type'),
      template:               val('sample-template'),
      dataset:                datasetList('sample-dataset'),
      system:                 val('sample-system'),
      sampler_type:           val('sample-sampler-type'),
      sampler_engine:         val('sample-sampler-engine'),
      num_return_sequences:   numVal('sample-num-return-seq'),
      num_sampling_batch_size: numVal('sample-batch-size'),
      num_sampling_batches:   numVal('sample-batches'),
      max_new_tokens:         numVal('sample-max-tokens'),
      temperature:            val('sample-temperature') ? parseFloat(val('sample-temperature')) : null,
      top_k:                  numVal('sample-top-k'),
      top_p:                  val('sample-top-p') ? parseFloat(val('sample-top-p')) : null,
      repetition_penalty:     val('sample-rep-penalty') ? parseFloat(val('sample-rep-penalty')) : null,
      prm_model:              val('sample-prm-model'),
      orm_model:              val('sample-orm-model'),
      n_best_to_keep:         numVal('sample-n-best'),
      output_dir:             val('sample-output-dir') || 'sample_output',
      gpu_ids:                gpuList('sample-gpu-ids'),
      more_params:            val('sample-more-params'),
    };
    try {
      const result = await apiFetch('/api/v1/sample/start', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      setStatus(status, window.i18n[pageLang].statusRunning, 'var(--success)');
      setTimeout(async () => {
        await refreshTasks('sample', 'sample-running-tasks', status);
        const sel = document.getElementById('sample-running-tasks');
        if (sel && sel.options.length > 0 && sel.options[0].value && !sel.value) {
          sel.selectedIndex = 0;
          sel.dispatchEvent(new Event('change'));
        }
        updateTaskButtons('sample');
      }, 1500);
      if (result.log_file) startLogStream('sample', result.log_file, 'sample-log');
    } catch (e) {
      setStatus(status, e.message, 'var(--danger)');
    } finally {
      setBtnLoading(startBtn, false);
    }
  });

  document.getElementById('sample-btn-refresh').addEventListener('click', async () => {
    await refreshTasks('sample', 'sample-running-tasks', 'sample-status');
    updateTaskButtons('sample');
  });
  document.getElementById('sample-btn-kill').addEventListener('click', async () => {
    const btn = document.getElementById('sample-btn-kill');
    setBtnLoading(btn, true);
    await killSelectedTask('sample-running-tasks', 'sample', 'sample-status');
    setBtnLoading(btn, false);
    updateTaskButtons('sample');
  });

  // ── Apply i18n on load ──
  window.applyI18n(pageLang);

  // ── Fetch version from /health and display in title ──
  apiFetch('/health').then(data => {
    if (data && data.version) {
      const badge = document.getElementById('header-version');
      if (badge) badge.textContent = 'v' + data.version;
    }
  }).catch(() => {});

  // ── Populate datalists from API ──
  async function onModelChange(prefix) {
    const model = val(prefix + '-model');
    if (!model) return;
    let meta = { template: '', model_type: '', system: '', matched_source: 'none' };
    try {
      const data = await apiFetch('/api/v1/model-meta?model=' + encodeURIComponent(model));
      if (data && typeof data === 'object') meta = { ...meta, ...data };
    } catch (_) {}
    const tplEl = document.getElementById(prefix + '-template');
    if (tplEl) tplEl.value = meta.template || '';
    const mtEl = document.getElementById(prefix + '-model-type');
    if (mtEl) mtEl.value = meta.model_type || '';
    const sysEl = document.getElementById(prefix + '-system');
    if (sysEl) {
      if (prefix === 'grpo') {
        sysEl.value = DEFAULT_GRPO_SYSTEM;
      } else {
        sysEl.value = meta.system || '';
      }
    }
    // For the train tab, also refresh records and auto-restore the latest one
    if (prefix === 'train') {
      if (meta.matched_source === 'none') {
        showToast(window.i18n[pageLang].warnModelMetaUnknown || '无法自动识别该模型的 model_type/template', 'info', 3500);
      } else if (meta.matched_source === 'args_json') {
        showToast(window.i18n[pageLang].infoModelMetaFromArgs || '已从本地 args.json 恢复模型参数', 'info', 2500);
      }
      if (trainRestoring) {
        updateTrainTaskTypeUI(false);
        scheduleTrainCommandPreview();
        return;
      }
      const records = await loadTrainRecords(model);
      // Only auto-restore if the user just changed the model (not restoring from a record)
      // and there's no task currently running
      const tasksSel = document.getElementById('train-running-tasks');
      const hasRunning = tasksSel && tasksSel.options.length > 0 && tasksSel.options[0].value;
      if (!hasRunning && records && records.length > 0) {
        const sel = document.getElementById('train-record');
        if (sel) {
          sel.value = records[0];
          const params = await restoreTrainRecord(model, records[0]);
          maybeShowRecordLog('train', 'train-log', params);
        }
      }
      updateTrainTaskTypeUI(false);
      scheduleTrainCommandPreview();
    }
    if (prefix === 'rlhf') {
      const records = await loadScopedRecords('rlhf', model, 'rlhf-record');
      const tasksSel = document.getElementById('rlhf-running-tasks');
      const hasRunning = tasksSel && tasksSel.options.length > 0 && tasksSel.options[0].value;
      if (!hasRunning && records && records.length > 0) {
        const sel = document.getElementById('rlhf-record');
        if (sel) {
          sel.value = records[0];
          const params = await restoreScopedRecord('rlhf', model, records[0], RLHF_RECORD_FIELD_MAP, 'rlhf-dataset-box', 'rlhf-dataset');
          maybeShowRecordLog('rlhf', 'rlhf-log', params);
        }
      }
      syncRlhfMetricsKeys();
      scheduleRlhfCommandPreview();
    }
    if (prefix === 'grpo') {
      const records = await loadScopedRecords('grpo', model, 'grpo-record');
      const tasksSel = document.getElementById('grpo-running-tasks');
      const hasRunning = tasksSel && tasksSel.options.length > 0 && tasksSel.options[0].value;
      if (!hasRunning && records && records.length > 0) {
        const sel = document.getElementById('grpo-record');
        if (sel) {
          sel.value = records[0];
          const params = await restoreScopedRecord('grpo', model, records[0], GRPO_RECORD_FIELD_MAP, 'grpo-dataset-box', 'grpo-dataset');
          maybeShowRecordLog('grpo', 'grpo-log', params);
        }
      }
      scheduleGrpoCommandPreview();
    }
  }

  // ── Training records (load, restore, clear) ──

  // Map from record field names (backend TrainRequest keys) to HTML element IDs
  const RECORD_FIELD_MAP = {
    model_type:                  'train-model-type',
    template:                    'train-template',
    train_stage:                 'train-stage',
    tuner_type:                  'train-tuner-type',
    seed:                        'train-seed',
    torch_dtype:                 'train-torch-dtype',
    use_liger_kernel:            'train-use-liger',
    use_ddp:                     'train-use-ddp',
    ddp_num:                     'train-ddp-num',
    deepspeed:                   'train-deepspeed',
    sequence_parallel_size:      'train-seq-parallel',
    learning_rate:               'train-lr',
    per_device_train_batch_size: 'train-batch-size',
    per_device_eval_batch_size:  'train-eval-batch-size',
    num_train_epochs:            'train-epochs',
    eval_steps:                  'train-eval-steps',
    save_steps:                  'train-save-steps',
    gradient_accumulation_steps: 'train-grad-accum',
    attn_impl:                   'train-attn-impl',
    neftune_noise_alpha:         'train-neftune-alpha',
    output_dir:                  'train-output-dir',
    logging_dir:                 'train-logging-dir',
    system:                      'train-system',
    envs:                        'train-envs',
    more_params:                 'train-more-params',
    split_dataset_ratio:         'train-split-ratio',
    max_length:                  'train-max-length',
    padding_free:                'train-padding-free',
    tuner_backend:               'train-tuner-backend',
    weight_decay:                'train-weight-decay',
    logging_steps:               'train-logging-steps',
    lr_scheduler_type:           'train-lr-scheduler',
    warmup_ratio:                'train-warmup-ratio',
    truncation_strategy:         'train-truncation',
    max_steps:                   'train-max-steps',
    max_grad_norm:               'train-max-grad-norm',
    lora_rank:                   'train-lora-rank',
    lora_alpha:                  'train-lora-alpha',
    lora_dropout:                'train-lora-dropout',
    lora_dtype:                  'train-lora-dtype',
    use_rslora:                  'train-use-rslora',
    use_dora:                    'train-use-dora',
    target_modules:              'train-target-modules',
    task_type:                   'train-task-type',
    loss_type:                   'train-loss-type',
    num_labels:                  'train-num-labels',
    use_chat_template:           'train-use-chat-template',
    model_name:                  'train-model-name',
    model_author:                'train-model-author',
    push_to_hub:                 'train-push-to-hub',
    hub_model_id:                'train-hub-model-id',
    hub_private_repo:            'train-hub-private',
    hub_strategy:                'train-hub-strategy',
    report_to:                   'train-report-to',
    swanlab_project:             'train-swanlab-project',
    swanlab_workspace:           'train-swanlab-workspace',
    swanlab_exp_name:            'train-swanlab-exp-name',
    swanlab_mode:                'train-swanlab-mode',
  };

  const RLHF_RECORD_FIELD_MAP = {
    model_type: 'rlhf-model-type', template: 'rlhf-template', rlhf_type: 'rlhf-type',
    tuner_type: 'rlhf-tuner-type', seed: 'rlhf-seed', torch_dtype: 'rlhf-torch-dtype',
    use_liger_kernel: 'rlhf-use-liger', use_ddp: 'rlhf-use-ddp', padding_free: 'rlhf-padding-free',
    learning_rate: 'rlhf-lr', per_device_train_batch_size: 'rlhf-batch-size',
    per_device_eval_batch_size: 'rlhf-eval-batch-size', num_train_epochs: 'rlhf-epochs',
    output_dir: 'rlhf-output-dir', logging_dir: 'rlhf-logging-dir', deepspeed: 'rlhf-deepspeed',
    sequence_parallel_size: 'rlhf-seq-parallel', ddp_num: 'rlhf-ddp-num',
    split_dataset_ratio: 'rlhf-split-ratio', max_length: 'rlhf-max-length',
    tuner_backend: 'rlhf-tuner-backend', weight_decay: 'rlhf-weight-decay',
    logging_steps: 'rlhf-logging-steps', lr_scheduler_type: 'rlhf-lr-scheduler',
    warmup_ratio: 'rlhf-warmup-ratio', max_steps: 'rlhf-max-steps', max_grad_norm: 'rlhf-max-grad-norm',
    eval_steps: 'rlhf-eval-steps', save_steps: 'rlhf-save-steps', gradient_accumulation_steps: 'rlhf-grad-accum',
    attn_impl: 'rlhf-attn-impl', lora_rank: 'rlhf-lora-rank', lora_alpha: 'rlhf-lora-alpha',
    lora_dropout: 'rlhf-lora-dropout', lora_dtype: 'rlhf-lora-dtype', target_modules: 'rlhf-target-modules',
    use_rslora: 'rlhf-use-rslora', use_dora: 'rlhf-use-dora',
    ref_model: 'rlhf-ref-model', ref_model_type: 'rlhf-ref-model-type',
    reward_model: 'rlhf-reward-model', reward_model_type: 'rlhf-reward-model-type',
    teacher_model: 'rlhf-teacher-model', teacher_model_type: 'rlhf-teacher-model-type',
    beta: 'rlhf-beta', max_completion_length: 'rlhf-max-completion-length', loss_scale: 'rlhf-loss-scale',
    lmbda: 'rlhf-lmbda', cpo_alpha: 'rlhf-cpo-alpha', rpo_alpha: 'rlhf-rpo-alpha',
    simpo_gamma: 'rlhf-simpo-gamma', desirable_weight: 'rlhf-desirable-weight', undesirable_weight: 'rlhf-undesirable-weight',
    system: 'rlhf-system', envs: 'rlhf-envs', more_params: 'rlhf-more-params', report_to: 'rlhf-report-to',
    swanlab_project: 'rlhf-swanlab-project', swanlab_workspace: 'rlhf-swanlab-workspace',
    swanlab_exp_name: 'rlhf-swanlab-exp-name', swanlab_mode: 'rlhf-swanlab-mode',
  };

  const GRPO_RECORD_FIELD_MAP = {
    model_type: 'grpo-model-type', template: 'grpo-template', tuner_type: 'grpo-tuner-type',
    seed: 'grpo-seed', torch_dtype: 'grpo-torch-dtype', learning_rate: 'grpo-lr',
    use_liger_kernel: 'grpo-use-liger', use_ddp: 'grpo-use-ddp', padding_free: 'grpo-padding-free',
    per_device_train_batch_size: 'grpo-batch-size', per_device_eval_batch_size: 'grpo-eval-batch-size',
    num_train_epochs: 'grpo-epochs', output_dir: 'grpo-output-dir', logging_dir: 'grpo-logging-dir',
    deepspeed: 'grpo-deepspeed', sequence_parallel_size: 'grpo-seq-parallel', ddp_num: 'grpo-ddp-num',
    split_dataset_ratio: 'grpo-split-ratio', max_length: 'grpo-max-length',
    tuner_backend: 'grpo-tuner-backend', weight_decay: 'grpo-weight-decay',
    logging_steps: 'grpo-logging-steps', lr_scheduler_type: 'grpo-lr-scheduler',
    warmup_ratio: 'grpo-warmup-ratio', max_steps: 'grpo-max-steps', max_grad_norm: 'grpo-max-grad-norm',
    eval_steps: 'grpo-eval-steps', save_steps: 'grpo-save-steps', gradient_accumulation_steps: 'grpo-grad-accum',
    attn_impl: 'grpo-attn-impl', lora_rank: 'grpo-lora-rank', lora_alpha: 'grpo-lora-alpha',
    lora_dropout: 'grpo-lora-dropout', lora_dtype: 'grpo-lora-dtype', target_modules: 'grpo-target-modules',
    use_rslora: 'grpo-use-rslora', use_dora: 'grpo-use-dora',
    vllm_mode: 'grpo-vllm-mode', num_generations: 'grpo-num-generations', max_completion_length: 'grpo-max-completion-length',
    reward_funcs: 'grpo-reward-funcs', reward_weights: 'grpo-reward-weights', ref_model: 'grpo-ref-model',
    temperature: 'grpo-temperature', top_k: 'grpo-top-k', top_p: 'grpo-top-p', repetition_penalty: 'grpo-repetition-penalty',
    vllm_gpu_memory_utilization: 'grpo-vllm-gpu-memory-utilization', vllm_tensor_parallel_size: 'grpo-vllm-tensor-parallel-size',
    vllm_max_model_len: 'grpo-vllm-max-model-len', vllm_server_host: 'grpo-vllm-server-host',
    vllm_server_port: 'grpo-vllm-server-port', vllm_server_timeout: 'grpo-vllm-server-timeout',
    loss_type: 'grpo-loss-type', beta: 'grpo-beta', epsilon: 'grpo-epsilon', epsilon_high: 'grpo-epsilon-high',
    num_iterations: 'grpo-num-iterations', system: 'grpo-system', envs: 'grpo-envs', more_params: 'grpo-more-params',
    report_to: 'grpo-report-to', swanlab_project: 'grpo-swanlab-project',
    swanlab_workspace: 'grpo-swanlab-workspace', swanlab_exp_name: 'grpo-swanlab-exp-name', swanlab_mode: 'grpo-swanlab-mode',
  };

  async function loadScopedRecords(scope, model, selectId) {
    const sel = document.getElementById(selectId);
    if (!sel || !model) return [];
    try {
      const data = await apiFetch(`/api/v1/${scope}/records?model=` + encodeURIComponent(model));
      const records = data.records || [];
      sel.innerHTML = '<option value="">' + (window.i18n[pageLang].optNoRecord || '-- 无记录 --') + '</option>';
      records.forEach(ts => {
        const opt = document.createElement('option');
        opt.value = ts;
        opt.textContent = ts;
        sel.appendChild(opt);
      });
      return records;
    } catch (_) {
      return [];
    }
  }

  async function restoreScopedRecord(scope, model, timestamp, fieldMap, boxId, hiddenId) {
    if (!model || !timestamp) return null;
    let restoredParams = null;
    try {
      const params = await apiFetch(`/api/v1/${scope}/records/detail?model=` + encodeURIComponent(model) + '&timestamp=' + encodeURIComponent(timestamp));
      restoredParams = params;
      if (params.dataset && Array.isArray(params.dataset)) {
        const box = document.getElementById(boxId);
        const hidden = document.getElementById(hiddenId);
        if (box && hidden) {
          box.querySelectorAll('.tag').forEach(t => t.remove());
          const textIn = box.querySelector('.tag-text-input');
          params.dataset.forEach(ds => {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.dataset.value = ds;
            const label = document.createElement('span');
            label.textContent = ds;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.textContent = '×';
            btn.addEventListener('click', e => { e.stopPropagation(); tag.remove(); hidden.value = Array.from(box.querySelectorAll('.tag')).map(t => t.dataset.value).join(' '); });
            tag.appendChild(label);
            tag.appendChild(btn);
            if (textIn) box.insertBefore(tag, textIn); else box.appendChild(tag);
          });
          hidden.value = params.dataset.join(' ');
        }
      }
      for (const [key, elId] of Object.entries(fieldMap)) {
        const el = document.getElementById(elId);
        if (!el || params[key] === undefined || params[key] === null) continue;
        if (el.multiple) {
          const values = Array.isArray(params[key]) ? params[key] : String(params[key]).split(/[,\s]+/).filter(Boolean);
          Array.from(el.options).forEach(o => { o.selected = values.includes(o.value); });
        } else if (el.type === 'checkbox') {
          el.checked = Boolean(params[key]);
        } else {
          el.value = params[key];
        }
      }
    } catch (_) {}
    if (scope === 'rlhf') {
      updateRlhfTypeUI(false);
      scheduleRlhfCommandPreview();
    } else if (scope === 'grpo') {
      updateGrpoVllmModeUI(false);
      scheduleGrpoCommandPreview();
    }
    return restoredParams;
  }

  async function loadTrainRecords(model) {
    const sel = document.getElementById('train-record');
    if (!sel || !model) return;
    try {
      const data = await apiFetch('/api/v1/train/records?model=' + encodeURIComponent(model));
      const records = data.records || [];
      sel.innerHTML = '<option value="">' + (window.i18n[pageLang].optNoRecord || '-- 无记录 --') + '</option>';
      records.forEach(ts => {
        const opt = document.createElement('option');
        opt.value = ts;
        opt.textContent = ts;
        sel.appendChild(opt);
      });
      return records;
    } catch (_) {
      return [];
    }
  }

  async function restoreTrainRecord(model, timestamp) {
    if (!model || !timestamp) return null;
    trainRestoring = true;
    let restoredParams = null;
    try {
      const params = await apiFetch(
        '/api/v1/train/records/detail?model=' + encodeURIComponent(model) +
        '&timestamp=' + encodeURIComponent(timestamp)
      );
      restoredParams = params;
      // Restore datasets tag-input
      if (params.dataset && Array.isArray(params.dataset)) {
        const box = document.getElementById('train-dataset-box');
        const hidden = document.getElementById('train-dataset');
        if (box && hidden) {
          box.querySelectorAll('.tag').forEach(t => t.remove());
          const textIn = box.querySelector('.tag-text-input');
          params.dataset.forEach(ds => {
            const tag = document.createElement('span');
            tag.className = 'tag';
            tag.dataset.value = ds;
            tag.title = ds;
            const label = document.createElement('span');
            label.textContent = ds;
            const btn = document.createElement('button');
            btn.type = 'button';
            btn.textContent = '×';
            btn.addEventListener('click', e => { e.stopPropagation(); tag.remove(); hidden.value = Array.from(box.querySelectorAll('.tag')).map(t => t.dataset.value).join(' '); });
            tag.appendChild(label);
            tag.appendChild(btn);
            if (textIn) box.insertBefore(tag, textIn);
            else box.appendChild(tag);
          });
          hidden.value = params.dataset.join(' ');
        }
      }
      // Restore all mapped fields
      for (const [key, elId] of Object.entries(RECORD_FIELD_MAP)) {
        const el = document.getElementById(elId);
        if (!el || params[key] === undefined || params[key] === null) continue;
        if (el.type === 'checkbox') {
          el.checked = Boolean(params[key]);
        } else {
          el.value = params[key];
        }
      }
      // Sync sliders to restored number-input values
      document.querySelectorAll('.range-slider').forEach(slider => {
        const targetId = slider.getAttribute('data-target');
        const numberEl = targetId ? document.getElementById(targetId) : null;
        if (numberEl && numberEl.value !== '') {
          const v = parseFloat(numberEl.value);
          if (!isNaN(v)) slider.value = Math.min(Math.max(v, parseFloat(slider.min)), parseFloat(slider.max));
        }
      });
    } catch (_) {}
    trainRestoring = false;
    updateTrainTaskTypeUI(false);
    scheduleTrainCommandPreview();
    return restoredParams;
  }

  // Wire record dropdown
  document.getElementById('train-record').addEventListener('change', async () => {
    const model = val('train-model');
    const ts = document.getElementById('train-record').value;
    if (ts) {
      const params = await restoreTrainRecord(model, ts);
      maybeShowRecordLog('train', 'train-log', params);
    }
    scheduleTrainCommandPreview();
  });

  // Wire clear-records button
  document.getElementById('train-btn-clear-records').addEventListener('click', async () => {
    const model = val('train-model');
    if (!model) return;
    try {
      await apiFetch('/api/v1/train/records?model=' + encodeURIComponent(model), { method: 'DELETE' });
      const sel = document.getElementById('train-record');
      if (sel) sel.innerHTML = '<option value="">' + (window.i18n[pageLang].optNoRecord || '-- 无记录 --') + '</option>';
    } catch (_) {}
  });

  document.getElementById('rlhf-record').addEventListener('change', async () => {
    const model = val('rlhf-model');
    const ts = document.getElementById('rlhf-record').value;
    if (ts) {
      const params = await restoreScopedRecord('rlhf', model, ts, RLHF_RECORD_FIELD_MAP, 'rlhf-dataset-box', 'rlhf-dataset');
      syncRlhfMetricsKeys();
      maybeShowRecordLog('rlhf', 'rlhf-log', params);
    }
  });

  document.getElementById('grpo-record').addEventListener('change', async () => {
    const model = val('grpo-model');
    const ts = document.getElementById('grpo-record').value;
    if (ts) {
      const params = await restoreScopedRecord('grpo', model, ts, GRPO_RECORD_FIELD_MAP, 'grpo-dataset-box', 'grpo-dataset');
      maybeShowRecordLog('grpo', 'grpo-log', params);
    }
  });

  document.getElementById('rlhf-btn-clear-records').addEventListener('click', async () => {
    const model = val('rlhf-model');
    if (!model) return;
    try {
      await apiFetch('/api/v1/rlhf/records?model=' + encodeURIComponent(model), { method: 'DELETE' });
      const sel = document.getElementById('rlhf-record');
      if (sel) sel.innerHTML = '<option value="">' + (window.i18n[pageLang].optNoRecord || '-- 无记录 --') + '</option>';
    } catch (_) {}
  });

  document.getElementById('grpo-btn-clear-records').addEventListener('click', async () => {
    const model = val('grpo-model');
    if (!model) return;
    try {
      await apiFetch('/api/v1/grpo/records?model=' + encodeURIComponent(model), { method: 'DELETE' });
      const sel = document.getElementById('grpo-record');
      if (sel) sel.innerHTML = '<option value="">' + (window.i18n[pageLang].optNoRecord || '-- 无记录 --') + '</option>';
    } catch (_) {}
  });
  function initMediaUpload() {
    document.querySelectorAll('.media-upload').forEach(zone => {
      const fileInput = document.getElementById(zone.getAttribute('data-input'));
      const mediaType = zone.getAttribute('data-type');
      const promptEl  = zone.querySelector('.media-upload-prompt');
      const previewEl = zone.querySelector('.media-preview');
      if (!fileInput || !promptEl || !previewEl) return;

      function showPreview(file) {
        const url = URL.createObjectURL(file);
        promptEl.style.display = 'none';
        previewEl.style.display = 'inline-block';
        if (mediaType === 'image') {
          previewEl.querySelector('.media-preview-img').src = url;
        } else if (mediaType === 'video') {
          const vid = previewEl.querySelector('.media-preview-vid');
          vid.src = url; vid.load();
        } else if (mediaType === 'audio') {
          const aud = previewEl.querySelector('.media-preview-aud');
          aud.src = url; aud.load();
        }
      }

      function clearPreview() {
        fileInput.value = '';
        promptEl.style.display = '';
        previewEl.style.display = 'none';
        const img = previewEl.querySelector('.media-preview-img');
        const vid = previewEl.querySelector('.media-preview-vid');
        const aud = previewEl.querySelector('.media-preview-aud');
        if (img) img.src = '';
        if (vid) { vid.pause(); vid.src = ''; vid.load(); }
        if (aud) { aud.pause(); aud.src = ''; aud.load(); }
      }

      // Expose clearPreview so the send handler can call it
      zone._clearPreview = clearPreview;

      // Click anywhere in zone (except clear button / media element) → open file dialog
      zone.addEventListener('click', e => {
        if (!e.target.closest('.media-clear') &&
            !e.target.closest('.media-preview-img') &&
            !e.target.closest('.media-preview-vid') &&
            !e.target.closest('.media-preview-aud')) {
          fileInput.click();
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files[0]) showPreview(fileInput.files[0]);
        else clearPreview();
      });

      previewEl.querySelector('.media-clear').addEventListener('click', e => {
        e.stopPropagation();
        clearPreview();
      });

      // Drag-and-drop
      zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
      zone.addEventListener('dragleave', ()  => zone.classList.remove('drag-over'));
      zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (!file) return;
        try {
          const dt = new DataTransfer();
          dt.items.add(file);
          fileInput.files = dt.files;
        } catch (_) {}
        showPreview(file);
      });
    });
  }

  async function loadDatalistOptions() {
    const populate = (id, items) => {
      const dl = document.getElementById(id);
      if (!dl) return;
      dl.innerHTML = '';
      items.forEach(v => {
        const opt = document.createElement('option');
        opt.value = v;
        dl.appendChild(opt);
      });
    };

    try {
      const [mRes, mtRes, tmRes, dsRes, dvRes] = await Promise.allSettled([
        apiFetch('/api/v1/models'),
        apiFetch('/api/v1/model-types'),
        apiFetch('/api/v1/templates'),
        apiFetch('/api/v1/datasets'),
        apiFetch('/api/v1/devices'),
      ]);
      if (mRes.status  === 'fulfilled') populate('dl-models',      mRes.value.models      || []);
      if (mtRes.status === 'fulfilled') populate('dl-model-types', mtRes.value.model_types || []);
      if (tmRes.status === 'fulfilled') populate('dl-templates',   tmRes.value.templates   || []);
      if (dsRes.status === 'fulfilled') populate('dl-datasets',    dsRes.value.datasets    || []);
      if (dvRes.status === 'fulfilled' && dvRes.value.devices) {
        const devices = dvRes.value.devices;
        const defaultDevice = dvRes.value.default;
        const gpuSelectIds = ['train-gpu-ids','rlhf-gpu-ids','grpo-gpu-ids',
                              'infer-gpu-ids','export-gpu-ids','eval-gpu-ids','sample-gpu-ids'];
        gpuSelectIds.forEach(id => {
          const sel = document.getElementById(id);
          if (!sel) return;
          sel.innerHTML = '';
          devices.forEach(d => {
            const o = document.createElement('option');
            o.value = d;
            o.textContent = d;
            if (d === defaultDevice) o.selected = true;
            sel.appendChild(o);
          });
        });
      }
    } catch (_) {}
  }

  // ── Slider + number input sync ──
  function initSliderInputs() {
    document.querySelectorAll('.range-slider').forEach(slider => {
      const targetId = slider.getAttribute('data-target');
      const numberEl = targetId ? document.getElementById(targetId) : null;
      if (!numberEl) return;
      // Set slider to current input value, or fall back to data-default, then min
      const initVal = numberEl.value !== '' ? numberEl.value
                    : (slider.getAttribute('data-default') ?? slider.min);
      slider.value = initVal;
      // Slider → number input
      slider.addEventListener('input', () => { numberEl.value = slider.value; });
      // Number input → slider (clamp to slider range)
      numberEl.addEventListener('input', () => {
        const v = parseFloat(numberEl.value);
        if (!isNaN(v)) {
          slider.value = Math.min(Math.max(v, parseFloat(slider.min)), parseFloat(slider.max));
        }
      });
    });
    document.querySelectorAll('.reset-btn').forEach(btn => {
      const targetId = btn.getAttribute('data-target');
      const def = btn.getAttribute('data-default');
      const numberEl = targetId ? document.getElementById(targetId) : null;
      const sliderEl = document.querySelector(`.range-slider[data-target="${targetId}"]`);
      if (!numberEl) return;
      btn.addEventListener('click', () => {
        const resetVal = def ?? '';
        numberEl.value = resetVal;
        if (sliderEl) sliderEl.value = resetVal !== '' ? resetVal : sliderEl.min;
        if (btn.closest('#section-train')) scheduleTrainCommandPreview();
      });
    });
  }

  // ── Tag-input (multi-select with custom values) ──
  function initTagInputs() {
    document.querySelectorAll('.tag-box').forEach(box => {
      const hiddenId = box.getAttribute('data-for');
      const hidden   = document.getElementById(hiddenId);
      const textIn   = box.querySelector('.tag-text-input');
      if (!textIn || !hidden) return;

      function currentTags() {
        return Array.from(box.querySelectorAll('.tag')).map(t => t.dataset.value);
      }

      function syncHidden() {
        hidden.value = currentTags().join(' ');
        if (hiddenId === 'train-dataset') scheduleTrainCommandPreview();
        if (hiddenId === 'rlhf-dataset') scheduleRlhfCommandPreview();
        if (hiddenId === 'grpo-dataset') scheduleGrpoCommandPreview();
      }

      function addTag(value) {
        value = value.trim();
        if (!value) return;
        if (currentTags().includes(value)) return; // no duplicates
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.dataset.value = value;
        tag.title = value;
        const label = document.createElement('span');
        label.textContent = value;
        label.style.overflow = 'hidden';
        label.style.textOverflow = 'ellipsis';
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'tag-remove';
        btn.innerHTML = '×';
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          tag.remove();
          syncHidden();
        });
        tag.appendChild(label);
        tag.appendChild(btn);
        box.insertBefore(tag, textIn);
        syncHidden();
      }

      textIn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ',') {
          e.preventDefault();
          addTag(textIn.value);
          textIn.value = '';
        } else if (e.key === 'Backspace' && textIn.value === '') {
          const tags = box.querySelectorAll('.tag');
          if (tags.length) { tags[tags.length - 1].remove(); syncHidden(); }
        }
      });

      // Also accept selection from datalist (fires 'input' then the value is set)
      textIn.addEventListener('change', () => {
        if (textIn.value) { addTag(textIn.value); textIn.value = ''; }
      });

      // Click anywhere in the box focuses the text input
      box.addEventListener('click', () => textIn.focus());
    });
  }

  // ── Initial task refresh on page load ──
  // Train is the active tab; refresh sft+pt like infer tab, then auto-restore if a single task
  refreshTasks(['sft', 'pt'], 'train-running-tasks', 'train-status').then(() => {
    const sel = document.getElementById('train-running-tasks');
    if (sel && sel.options.length === 1 && sel.options[0].value) {
      sel.selectedIndex = 0;
      sel.dispatchEvent(new Event('change'));
    }
    updateTrainKillBtn();
  });

  // Load train records on page load and auto-restore last record if no task is running
  const initModel = val('train-model');
  if (initModel) {
    loadTrainRecords(initModel).then(records => {
      if (!records || records.length === 0) return;
      // Check if a train task is running; if not, restore the last record
      const tasksSel = document.getElementById('train-running-tasks');
      const hasRunning = tasksSel && tasksSel.options.length > 0 && tasksSel.options[0].value;
      if (!hasRunning) {
        const sel = document.getElementById('train-record');
        if (sel && records[0]) {
          sel.value = records[0];
          restoreTrainRecord(initModel, records[0]).then(params => {
            maybeShowRecordLog('train', 'train-log', params);
          });
        }
      }
    });
  }
  const initRlhfModel = val('rlhf-model');
  if (initRlhfModel) {
    loadScopedRecords('rlhf', initRlhfModel, 'rlhf-record').then(records => {
      if (!records || records.length === 0) return;
      const tasksSel = document.getElementById('rlhf-running-tasks');
      const hasRunning = tasksSel && tasksSel.options.length > 0 && tasksSel.options[0].value;
      if (!hasRunning) {
        const sel = document.getElementById('rlhf-record');
        if (sel && records[0]) {
          sel.value = records[0];
          restoreScopedRecord('rlhf', initRlhfModel, records[0], RLHF_RECORD_FIELD_MAP, 'rlhf-dataset-box', 'rlhf-dataset').then(params => {
            maybeShowRecordLog('rlhf', 'rlhf-log', params);
          });
        }
      }
    });
  }
  const initGrpoModel = val('grpo-model');
  if (initGrpoModel) {
    loadScopedRecords('grpo', initGrpoModel, 'grpo-record').then(records => {
      if (!records || records.length === 0) return;
      const tasksSel = document.getElementById('grpo-running-tasks');
      const hasRunning = tasksSel && tasksSel.options.length > 0 && tasksSel.options[0].value;
      if (!hasRunning) {
        const sel = document.getElementById('grpo-record');
        if (sel && records[0]) {
          sel.value = records[0];
          restoreScopedRecord('grpo', initGrpoModel, records[0], GRPO_RECORD_FIELD_MAP, 'grpo-dataset-box', 'grpo-dataset').then(params => {
            maybeShowRecordLog('grpo', 'grpo-log', params);
          });
        }
      }
    });
  }

  // Refresh all other tabs; auto-select the first task if exactly one is running
  ['infer', 'export', 'eval', 'sample'].forEach(prefix => {
    const cfg = tabCmds[prefix];
    refreshTasks(cfg[0], cfg[1], cfg[2]).then(() => {
      const sel = document.getElementById(cfg[1]);
      if (sel && sel.options.length === 1 && sel.options[0].value) {
        sel.selectedIndex = 0;
        sel.dispatchEvent(new Event('change'));
      }
    });
  });
  refreshRlhfTasks().then(() => {
    const sel = document.getElementById('rlhf-running-tasks');
    if (sel && sel.options.length === 1 && sel.options[0].value) {
      sel.selectedIndex = 0;
      sel.dispatchEvent(new Event('change'));
    }
  });
  refreshGrpoTasks().then(() => {
    const sel = document.getElementById('grpo-running-tasks');
    if (sel && sel.options.length === 1 && sel.options[0].value) {
      sel.selectedIndex = 0;
      sel.dispatchEvent(new Event('change'));
    }
  });

  // Wire model change → template/model_type/system linkage for all tabs
  ['train','rlhf','grpo','infer','export','eval','sample'].forEach(p => {
    const el = document.getElementById(p + '-model');
    if (el) el.addEventListener('change', () => onModelChange(p));
  });

  // Load all dropdown options
  loadDatalistOptions();
  initTagInputs();
  initSliderInputs();
  updateTrainTaskTypeUI(false);
  updateRlhfTypeUI(false);
  updateGrpoVllmModeUI(false);
  updateTrainKillBtn();
  updateRlhfTaskButtons();
  updateGrpoTaskButtons();
  initMediaUpload();

})();
