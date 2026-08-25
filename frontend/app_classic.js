// app.js — Agentic Multi-LLM Validation System
// v3 — Original tracker restored · Parallel pairs visualised · Step-wise cards · Context panel
'use strict';

// ── Stage definitions ─────────────────────────────────────────────────────
// parallelWith: id of the stage this runs alongside (null = sequential)
const STAGES = [
  { id: 1, name: 'LLaMA 3',  icon: '🦙', iconClass: 'icon-llama',   parallelWith: 2 },
  { id: 2, name: 'Mistral',  icon: '⚡', iconClass: 'icon-mistral', parallelWith: 1 },
  { id: 3, name: 'Verifier', icon: '🔍', iconClass: 'icon-verify',  parallelWith: 4 },
  { id: 4, name: 'Scorer',   icon: '📊', iconClass: 'icon-score',   parallelWith: 3 },
  { id: 5, name: 'Judge',    icon: '⚖️', iconClass: 'icon-judge',   parallelWith: null },
  { id: 6, name: 'Combiner', icon: '✦',  iconClass: 'icon-final',   parallelWith: null },
];

// ── DOM refs ─────────────────────────────────────────────────────────────
const queryInput    = document.getElementById('query-input');
const submitBtn     = document.getElementById('submit-btn');
const charCount     = document.getElementById('char-count');
const healthBar     = document.getElementById('health-bar');
const healthMsg     = document.getElementById('health-msg');
const tracker       = document.getElementById('pipeline-tracker');
const trackerBody   = document.getElementById('tracker-body');
const resultsCont   = document.getElementById('results-container');
const timerDisplay  = document.getElementById('timer-display');
const postActions   = document.getElementById('post-actions');
const newQBtn       = document.getElementById('new-question-btn');
const queryCard     = document.getElementById('query-card');
const contextBtn    = document.getElementById('context-btn');
const contextPanel  = document.getElementById('context-panel');
const ctxRecords    = document.getElementById('ctx-records');
const ctxBadge      = document.getElementById('ctx-badge');
const ctxCloseBtn   = document.getElementById('ctx-close-btn');
const ctxClearBtn   = document.getElementById('ctx-clear-btn');

// ── State ─────────────────────────────────────────────────────────────────
let timerInterval = null;
let startTime     = null;
// For parallel card grouping
let parallelAnswerRow  = null;   // wrapper for stage 1+2 side-by-side
let parallelVerifyRow  = null;   // wrapper for stage 3+4 side-by-side
// Skeleton tracking
let activeSkelIds = new Set();

// ── Utility ───────────────────────────────────────────────────────────────
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// Light markdown: **bold**, ## heading, - list item, newlines
function md(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^## (.+)$/gm,  '<div class="md-h3">$1</div>')
    .replace(/^- (.+)$/gm,   '<div class="md-li">· $1</div>')
    .replace(/\n/g, '<br>');
}

// ── Health check ──────────────────────────────────────────────────────────
async function checkHealth() {
  healthBar.className = '';
  healthMsg.textContent = 'Checking Ollama connection…';
  try {
    const d = await fetch('/api/health').then(r => r.json());
    if (d.ok) {
      const miss = [];
      if (!d.has_llama3)  miss.push('llama3');
      if (!d.has_mistral) miss.push('mistral');
      if (!miss.length) {
        healthBar.className = 'ok';
        const ready = d.models.filter(m => m.includes('llama3') || m.includes('mistral')).join(', ');
        healthMsg.textContent = `Ollama connected · Models ready: ${ready}`;
      } else {
        healthBar.className = 'warn';
        healthMsg.textContent = `Missing models: ${miss.join(', ')} — run ollama pull`;
      }
    } else {
      healthBar.className = 'error';
      healthMsg.textContent = 'Ollama not reachable — run: ollama serve';
    }
  } catch {
    healthBar.className = 'error';
    healthMsg.textContent = 'Cannot reach server — is server.py running?';
  }
}

// ── Context panel ─────────────────────────────────────────────────────────
async function loadContext() {
  try {
    const d = await fetch('/api/context').then(r => r.json());
    if (d.count === 0) {
      ctxBadge.style.display = 'none';
      ctxRecords.innerHTML = '<p class="ctx-empty">No queries stored yet. Run your first query!</p>';
      return;
    }
    ctxBadge.textContent = d.count;
    ctxBadge.style.display = 'inline-flex';
    ctxRecords.innerHTML = d.records.map((rec, i) => {
      const ts = new Date(rec.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      return `
        <div class="ctx-entry">
          <div class="ctx-entry-meta">
            <span class="ctx-num">Q${d.count - i}</span>
            <span class="ctx-ts">${ts}</span>
            <span class="ctx-words">${rec.word_count} words</span>
          </div>
          <div class="ctx-q">${esc(rec.query)}</div>
          ${rec.answer_summary ? `<div class="ctx-a">${esc(rec.answer_summary)}</div>` : ''}
        </div>`;
    }).join('');
  } catch {
    ctxRecords.innerHTML = '<p class="ctx-empty" style="color:var(--rose)">Failed to load context.</p>';
  }
}

contextBtn.addEventListener('click', async () => {
  if (!contextPanel.hidden) {
    contextPanel.hidden = true;
    contextBtn.classList.remove('active');
  } else {
    await loadContext();
    contextPanel.hidden = false;
    contextBtn.classList.add('active');
    contextPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});
ctxCloseBtn.addEventListener('click', () => {
  contextPanel.hidden = true;
  contextBtn.classList.remove('active');
});

ctxClearBtn.addEventListener('click', async () => {
  if (!confirm('Delete the stored conversation history (last 5 queries)? This cannot be undone.')) return;
  ctxClearBtn.disabled = true;
  try {
    await fetch('/api/context', { method: 'DELETE' });
    await loadContext();
  } catch {
    ctxRecords.innerHTML = '<p class="ctx-empty" style="color:var(--rose)">Failed to clear context.</p>';
  } finally {
    ctxClearBtn.disabled = false;
  }
});

// ── Pipeline tracker — vertical flow ─────────────────────────────────────
// Layout:  [1][2]  →  [3][4]  →  [5]  →  [6]
//          parallel   parallel   seq       seq

function makeBubbleItem(s) {
  const item = document.createElement('div');
  item.className = 'stage-item';
  item.id = `stage-item-${s.id}`;
  item.innerHTML = `
    <div class="stage-bubble" id="stage-bubble-${s.id}">${s.icon}</div>
    <span class="stage-label" id="stage-label-${s.id}">${s.name}</span>
  `;
  return item;
}

function makeParallelRow(stageA, stageB) {
  const wrap = document.createElement('div');
  wrap.className = 'track-row track-parallel';
  // Left bubble
  wrap.appendChild(makeBubbleItem(stageA));
  // H-connector between the two parallel stages
  const hconn = document.createElement('div');
  hconn.className = 'h-connector';
  hconn.id = `hconn-${stageA.id}`;
  wrap.appendChild(hconn);
  // Right bubble
  wrap.appendChild(makeBubbleItem(stageB));
  // Parallel label underneath
  const lbl = document.createElement('div');
  lbl.className = 'parallel-tag';
  lbl.textContent = '⟳ parallel';
  wrap.appendChild(lbl);
  return wrap;
}

function makeSeqRow(s) {
  const wrap = document.createElement('div');
  wrap.className = 'track-row track-seq';
  wrap.appendChild(makeBubbleItem(s));
  return wrap;
}

function makeVConnector(id) {
  const vc = document.createElement('div');
  vc.className = 'v-connector';
  vc.id = id;
  vc.innerHTML = '<div class="v-line"></div><div class="v-arrow">▼</div>';
  return vc;
}

function buildTracker() {
  trackerBody.innerHTML = '';
  // Row 1 — stages 1 + 2 in parallel
  trackerBody.appendChild(makeParallelRow(STAGES[0], STAGES[1]));
  trackerBody.appendChild(makeVConnector('vc-12'));
  // Row 2 — stages 3 + 4 in parallel
  trackerBody.appendChild(makeParallelRow(STAGES[2], STAGES[3]));
  trackerBody.appendChild(makeVConnector('vc-34'));
  // Row 3 — stage 5 sequential
  trackerBody.appendChild(makeSeqRow(STAGES[4]));
  trackerBody.appendChild(makeVConnector('vc-56'));
  // Row 4 — stage 6 sequential
  trackerBody.appendChild(makeSeqRow(STAGES[5]));
}

function setStageRunning(id) {
  const item   = document.getElementById(`stage-item-${id}`);
  const bubble = document.getElementById(`stage-bubble-${id}`);
  const label  = document.getElementById(`stage-label-${id}`);
  if (item)   item.classList.add('running');
  if (bubble) bubble.className = 'stage-bubble running';
  if (label)  label.classList.add('running');
}

// Track which of each parallel pair are done so we activate the v-connector
// only when BOTH in the group are complete
const _donePairs = { 'vc-12': new Set(), 'vc-34': new Set() };

function setStageDone(id) {
  const item   = document.getElementById(`stage-item-${id}`);
  const bubble = document.getElementById(`stage-bubble-${id}`);
  const label  = document.getElementById(`stage-label-${id}`);
  if (item)   { item.classList.remove('running'); item.classList.add('done'); }
  if (bubble) { bubble.className = 'stage-bubble done'; bubble.textContent = '✓'; }
  if (label)  { label.classList.remove('running'); label.classList.add('done'); }

  // H-connector inside the parallel row (1→2 or 3→4)
  // Both bubble A and B share one h-connector keyed on the lower id
  const hconn = document.getElementById(`hconn-${id === 2 ? 1 : id === 4 ? 3 : id}`);
  if (hconn) hconn.classList.add('active');

  // Activate v-connector once the whole group is done
  if (id === 1 || id === 2) {
    _donePairs['vc-12'].add(id);
    if (_donePairs['vc-12'].size === 2) activateVC('vc-12');
  } else if (id === 3 || id === 4) {
    _donePairs['vc-34'].add(id);
    if (_donePairs['vc-34'].size === 2) activateVC('vc-34');
  } else if (id === 5) {
    activateVC('vc-56');
  }
}

function activateVC(vcId) {
  const el = document.getElementById(vcId);
  if (el) el.classList.add('active');
}

// ── Skeleton helpers ──────────────────────────────────────────────────────
function addSkeleton(id) {
  if (document.getElementById(id)) return; // don't double-add
  const s = document.createElement('div');
  s.className = 'skeleton-card';
  s.id = id;
  s.innerHTML = `
    <div class="skeleton-line" style="width:38%;margin-bottom:18px"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line" style="width:90%"></div>
    <div class="skeleton-line" style="width:75%"></div>
    <div class="skeleton-line" style="width:60%"></div>
  `;
  activeSkelIds.add(id);
  return s;
}

function addSkeletonTo(container, id) {
  const s = addSkeleton(id);
  if (s) container.appendChild(s);
}

function addSkeletonToMain(id) {
  const s = addSkeleton(id);
  if (s) resultsCont.appendChild(s);
  scroll(s);
}

function removeSkel(id) {
  document.getElementById(id)?.remove();
  activeSkelIds.delete(id);
}

function scroll(el) {
  if (el) setTimeout(() => el.scrollIntoView({ behavior:'smooth', block:'nearest' }), 80);
}

// ── Parallel row wrappers ─────────────────────────────────────────────────
// Returns (or creates) the 2-column row for stages 1+2
function getAnswerRow() {
  if (parallelAnswerRow) return parallelAnswerRow;
  // Remove any full-width skeletons for stages 1 or 2 first
  removeSkel('skel-s1'); removeSkel('skel-s2');
  const row = document.createElement('div');
  row.className = 'pair-row';
  row.id = 'pair-answers';
  resultsCont.appendChild(row);
  scroll(row);
  parallelAnswerRow = row;
  return row;
}

function getVerifyRow() {
  if (parallelVerifyRow) return parallelVerifyRow;
  removeSkel('skel-s3'); removeSkel('skel-s4');
  const row = document.createElement('div');
  row.className = 'pair-row';
  row.id = 'pair-verify';
  resultsCont.appendChild(row);
  scroll(row);
  parallelVerifyRow = row;
  return row;
}

// ── Card builders ──────────────────────────────────────────────────────────
function iconHtml(emoji, cls) {
  return `<div class="card-icon ${cls}">${emoji}</div>`;
}

function titleHtml(title, tag) {
  return `<div>
    <div class="card-title">${title}</div>
    ${tag ? `<span class="card-tag ${tag.cls}">${tag.label}</span>` : ''}
  </div>`;
}

function makeCard(id, header, body, extra = '') {
  const c = document.createElement('div');
  c.className = `result-card glass ${extra}`;
  c.id = id;
  c.innerHTML = `<div class="card-header">${header}</div><div class="card-body">${body}</div>`;
  return c;
}

// Answers (LLaMA3 + Mistral) → side-by-side
function renderAnswerCard(stage, name, output) {
  const isLlama = stage === 1;
  const row   = getAnswerRow();
  removeSkel(isLlama ? 'skel-s1' : 'skel-s2');

  const card = makeCard(
    `card-s${stage}`,
    `${iconHtml(isLlama ? '🦙' : '⚡', isLlama ? 'icon-llama' : 'icon-mistral')}
     ${titleHtml(`Answer — ${name}`, isLlama
       ? { cls: 'tag-llama',   label: 'LLaMA 3' }
       : { cls: 'tag-mistral', label: 'Mistral'  })}`,
    esc(output)
  );
  row.appendChild(card);
  scroll(card);
}

// Verifier → left side of verify row
function renderVerifierCard(output) {
  const row = getVerifyRow();
  removeSkel('skel-s3');
  const card = makeCard('card-verifier',
    `${iconHtml('🔍','icon-verify')}${titleHtml('Verifier Feedback')}`,
    esc(output)
  );
  row.appendChild(card);
  scroll(card);
}

// Similarity scores → right side of verify row
function renderScoreCard(s1, s2) {
  const row = getVerifyRow();
  removeSkel('skel-s4');
  const w1 = s1.composite_score >= s2.composite_score;
  const card = document.createElement('div');
  card.className = 'result-card glass';
  card.id = 'card-scores';
  card.innerHTML = `
    <div class="card-header">${iconHtml('📊','icon-score')}${titleHtml('Similarity Scores')}</div>
    <div class="score-grid">
      ${scoreModelHtml('LLM 1 — LLaMA 3', s1, 'var(--indigo-g)')}
      ${scoreModelHtml('LLM 2 — Mistral',  s2, 'var(--cyan)')}
    </div>
    <div class="winner-banner">🏆 Similarity winner: ${w1 ? 'LLM 1 (LLaMA 3)' : 'LLM 2 (Mistral)'}</div>
  `;
  row.appendChild(card);
  scroll(card);
  setTimeout(() => {
    animBar(card, '.bar-sem-1', s1.semantic_similarity);
    animBar(card, '.bar-sem-2', s2.semantic_similarity);
    animBar(card, '.bar-len-1', s1.length_score);
    animBar(card, '.bar-len-2', s2.length_score);
  }, 50);
}

function scoreModelHtml(name, s, color) {
  const n = name.includes('LLaMA') ? '1' : '2';
  return `<div class="score-model-card">
    <div class="score-model-name" style="color:${color}">${name}</div>
    <div class="score-row">
      <div class="score-lbl"><span>Semantic Similarity</span><span>${s.semantic_similarity}</span></div>
      <div class="score-bar-bg"><div class="score-bar bar-sem bar-sem-${n}" style="width:0"></div></div>
    </div>
    <div class="score-row">
      <div class="score-lbl"><span>Length / Completeness</span><span>${s.length_score} <small>(${s.word_count}w)</small></span></div>
      <div class="score-bar-bg"><div class="score-bar bar-len bar-len-${n}" style="width:0"></div></div>
    </div>
    <div class="composite-score">
      <span class="comp-lbl">Composite (/ 10)</span>
      <span class="comp-val">${s.composite_score}</span>
    </div>
  </div>`;
}

function animBar(root, sel, val) {
  const el = root.querySelector(sel);
  if (el) el.style.width = `${(val * 100).toFixed(1)}%`;
}

// Judge → full width
function renderJudgeCard(output) {
  const card = makeCard('card-judge',
    `${iconHtml('⚖️','icon-judge')}${titleHtml('Judge Evaluation & Verdict')}`,
    md(output), 'card-judge'
  );
  resultsCont.appendChild(card);
  scroll(card);
}

// Final → full width, gold glow
function renderFinalCard(output) {
  const card = makeCard('card-final',
    `${iconHtml('✦','icon-final')}${titleHtml('✦ Final Combined Answer')}`,
    md(output), 'card-final'
  );
  card.querySelector('.card-body').classList.add('final-body');
  // Copy button
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.innerHTML = '📋 Copy Answer';
  btn.onclick = () => {
    navigator.clipboard.writeText(output).then(() => {
      btn.className = 'copy-btn copied'; btn.innerHTML = '✓ Copied!';
      setTimeout(() => { btn.className = 'copy-btn'; btn.innerHTML = '📋 Copy Answer'; }, 2000);
    });
  };
  card.appendChild(btn);
  resultsCont.appendChild(card);
  scroll(card);
}

// ── Timer ─────────────────────────────────────────────────────────────────
function startTimer() {
  startTime = Date.now();
  timerDisplay.className = 'visible';
  timerInterval = setInterval(() => {
    timerDisplay.textContent = `⏱ Pipeline running… ${((Date.now() - startTime) / 1000).toFixed(0)}s`;
  }, 1000);
}
function stopTimer(elapsed) {
  clearInterval(timerInterval);
  timerDisplay.textContent = `✓ Pipeline completed in ${elapsed}s`;
}

// ── Reset ─────────────────────────────────────────────────────────────────
function resetUI() {
  parallelAnswerRow = null;
  parallelVerifyRow = null;
  activeSkelIds.clear();
  _donePairs['vc-12'].clear();
  _donePairs['vc-34'].clear();
  resultsCont.innerHTML = '';
  postActions.hidden = true;
  timerDisplay.className = '';
  timerDisplay.textContent = '';
}

// ── SSE pipeline ──────────────────────────────────────────────────────────
async function runPipeline(query) {
  resetUI();
  tracker.className = 'glass visible';
  buildTracker();
  submitBtn.disabled = true;
  submitBtn.classList.add('loading');
  startTimer();

  try {
    const useRag = document.getElementById('rag-checkbox') ? document.getElementById('rag-checkbox').checked : true;
    const res = await fetch('/api/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, use_rag: useRag }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data:')) continue;
        let p;
        try { p = JSON.parse(line.slice(5).trim()); } catch { continue; }
        handleEvent(p);
      }
    }
  } catch (err) {
    stopTimer(((Date.now() - startTime) / 1000).toFixed(1));
    const e = document.createElement('div');
    e.className = 'error-card';
    e.innerHTML = `<strong>Pipeline Error</strong>${esc(err.message)}`;
    resultsCont.appendChild(e);
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('loading');
    checkHealth();
  }
}

// ── SSE event handler ─────────────────────────────────────────────────────
function handleEvent(p) {

  // ── Stage started ─────────────────────────────────────────────────────
  if (p.status === 'running') {
    const id = p.stage;
    setStageRunning(id);

    // Add skeletons appropriately
    if (id === 1 || id === 2) {
      // Parallel pair: add skeleton inside the answer row
      const row = getAnswerRow();
      addSkeletonTo(row, `skel-s${id}`);
    } else if (id === 3 || id === 4) {
      // Parallel pair: add skeleton inside the verify row
      const row = getVerifyRow();
      addSkeletonTo(row, `skel-s${id}`);
    } else {
      // Sequential: full-width skeleton in main container
      addSkeletonToMain(`skel-s${id}`);
    }
    return;
  }

  // ── Context loaded ────────────────────────────────────────────────────
  if (p.loaded !== undefined) {
    ctxBadge.textContent = p.entries;
    ctxBadge.style.display = 'inline-flex';
    return;
  }

  // ── Stage result ──────────────────────────────────────────────────────
  if (p.output !== undefined) {
    const { stage, name, output } = p;
    setStageDone(stage);
    removeSkel(`skel-s${stage}`);

    if (stage === 1) renderAnswerCard(1, name, output);
    else if (stage === 2) renderAnswerCard(2, name, output);
    else if (stage === 3) renderVerifierCard(output);
    else if (stage === 5) { removeSkel('skel-s5'); renderJudgeCard(output); }
    else if (stage === 6) {
      removeSkel('skel-s6');
      renderFinalCard(output);
      postActions.hidden = false;
      scroll(postActions);
      loadContext();
    }
    return;
  }

  // ── Scores ────────────────────────────────────────────────────────────
  if (p.scores1 !== undefined) {
    setStageDone(4);
    removeSkel('skel-s4');
    renderScoreCard(p.scores1, p.scores2);
    return;
  }

  // ── Done ──────────────────────────────────────────────────────────────
  if (p.elapsed !== undefined) {
    stopTimer(p.elapsed);
    return;
  }

  // ── Error ─────────────────────────────────────────────────────────────
  if (p.message !== undefined) {
    stopTimer(((Date.now() - startTime) / 1000).toFixed(1));
    const e = document.createElement('div');
    e.className = 'error-card';
    e.innerHTML = `<strong>Agent Error</strong>${esc(p.message)}`;
    resultsCont.appendChild(e);
  }
}

// ── New Question button ───────────────────────────────────────────────────
newQBtn.addEventListener('click', () => {
  resetUI();
  tracker.className = 'glass';
  queryCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  queryInput.value = '';
  charCount.textContent = '0 characters';
  queryInput.focus();
});

// ── Input listeners ───────────────────────────────────────────────────────
queryInput.addEventListener('input', () => {
  charCount.textContent = `${queryInput.value.length} characters`;
});
queryInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
    e.preventDefault(); submitBtn.click();
  }
});
submitBtn.addEventListener('click', () => {
  const q = queryInput.value.trim();
  if (!q) {
    queryInput.focus();
    queryInput.style.borderColor = 'var(--rose)';
    setTimeout(() => { queryInput.style.borderColor = ''; }, 1500);
    return;
  }
  runPipeline(q);
});

// ── Init ─────────────────────────────────────────────────────────────────
checkHealth();
loadContext();
// Show the tracker layout statically on load (idle state)
buildTracker();
tracker.className = 'glass visible';

// ── Optional RAG File Uploader (Multi-File Support) ─────────────────────────
async function uploadRagDocs(input) {
  const files = Array.from(input.files || []);
  if (!files.length) return;
  const statusMsg = document.getElementById('rag-status-msg');

  let totalChunks = 0;
  let successCount = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (statusMsg) {
      statusMsg.textContent = `Uploading ${i + 1}/${files.length}: ${file.name}...`;
    }

    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/rag/upload', {
        method: 'POST',
        body: formData
      }).then(r => r.json());

      if (res.ok) {
        successCount++;
        totalChunks += (res.chunks_created || 0);
      }
    } catch (err) {
      console.error(`Error uploading ${file.name}:`, err);
    }
  }

  if (statusMsg) {
    if (files.length === 1) {
      statusMsg.textContent = `✓ Indexed ${files[0].name} (${totalChunks} chunks)`;
    } else {
      statusMsg.textContent = `✓ Indexed ${successCount}/${files.length} docs (${totalChunks} total chunks)`;
    }
  }
  input.value = '';
}
window.uploadRagDocs = uploadRagDocs;
window.uploadRagDoc = uploadRagDocs;
