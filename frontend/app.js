// app.js — Neural Stream UI for the Agentic Multi-LLM Validation System
// Completely new visual language from the "classic" glassmorphism build
// (still on disk as index_classic.html / app_classic.js / style_classic.css).
// Talks to the exact same /api/stream SSE contract, PLUS the newer
// `confidence` and `regenerate` events the classic UI never displayed.
'use strict';

const STAGES = [
  { id: 1, name: 'LLaMA 3',  icon: '🦙', edge: 'llama'   },
  { id: 2, name: 'Mistral',  icon: '⚡', edge: 'mistral' },
  { id: 3, name: 'Verifier', icon: '🔍', edge: 'verify'  },
  { id: 4, name: 'Scorer',   icon: '📊', edge: 'score'   },
  { id: 5, name: 'Judge',    icon: '⚖️', edge: 'judge'   },
  { id: 6, name: 'Combiner', icon: '✦',  edge: 'final'   },
];

// ── DOM refs ─────────────────────────────────────────────────────────────
const queryInput     = document.getElementById('query-input');
const submitBtn      = document.getElementById('submit-btn');
const charCount       = document.getElementById('char-count');
const healthBar       = document.getElementById('health-bar');
const healthMsg       = document.getElementById('health-msg');
const railWrap        = document.getElementById('rail-wrap');
const railTrack       = document.getElementById('rail-track');
const stream          = document.getElementById('stream');
const timerDisplay    = document.getElementById('timer-display');
const postActions     = document.getElementById('post-actions');
const newQBtn         = document.getElementById('new-question-btn');
const queryCard       = document.getElementById('query-card');
const regenBanner     = document.getElementById('regen-banner');
const regenText       = document.getElementById('regen-text');
const confidenceCard  = document.getElementById('confidence-card');
const gauge           = document.getElementById('gauge');
const gaugeVal        = document.getElementById('gauge-val');
const confTitle       = document.getElementById('conf-title');
const confSub         = document.getElementById('conf-sub');
const confChips       = document.getElementById('conf-chips');
const contextBtn      = document.getElementById('context-btn');
const drawer          = document.getElementById('drawer');
const drawerOverlay   = document.getElementById('drawer-overlay');
const ctxRecords      = document.getElementById('ctx-records');
const ctxBadge        = document.getElementById('ctx-badge');
const ctxCloseBtn     = document.getElementById('ctx-close-btn');
const ctxClearBtn     = document.getElementById('ctx-clear-btn');

// ── State ────────────────────────────────────────────────────────────────
let timerInterval = null;
let startTime = null;
let parallelAnswerRow = null;
let parallelVerifyRow = null;

// ── Utility ──────────────────────────────────────────────────────────────
function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function md(text) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^## (.+)$/gm,  '<div class="md-h3">$1</div>')
    .replace(/^- (.+)$/gm,   '<div class="md-li">· $1</div>')
    .replace(/\n/g, '<br>');
}
function scrollTo(el) {
  if (el) setTimeout(() => el.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 80);
}

// ── Health check ─────────────────────────────────────────────────────────
async function checkHealth() {
  healthBar.className = 'health';
  healthMsg.textContent = 'Checking Ollama connection…';
  try {
    const d = await fetch('/api/health').then(r => r.json());
    if (d.ok) {
      const miss = [];
      if (!d.has_llama3)  miss.push('llama3');
      if (!d.has_mistral) miss.push('mistral');
      if (!miss.length) {
        healthBar.className = 'health ok';
        const ready = d.models.filter(m => m.includes('llama3') || m.includes('mistral')).join(', ');
        healthMsg.textContent = `Ollama connected · Models ready: ${ready}`;
      } else {
        healthBar.className = 'health warn';
        healthMsg.textContent = `Missing models: ${miss.join(', ')} — run ollama pull`;
      }
    } else {
      healthBar.className = 'health error';
      healthMsg.textContent = 'Ollama not reachable — run: ollama serve';
    }
  } catch {
    healthBar.className = 'health error';
    healthMsg.textContent = 'Cannot reach server — is server.py running?';
  }
}

// ── Context drawer ───────────────────────────────────────────────────────
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
        <div class="ctx-item">
          <div class="ctx-meta"><span>Q${d.count - i}</span><span>${ts}</span><span>${rec.word_count}w</span></div>
          <div class="ctx-q">${esc(rec.query)}</div>
          ${rec.answer_summary ? `<div class="ctx-a">${esc(rec.answer_summary)}</div>` : ''}
        </div>`;
    }).join('');
  } catch {
    ctxRecords.innerHTML = '<p class="ctx-empty" style="color:var(--red)">Failed to load context.</p>';
  }
}

function openDrawer() {
  drawer.classList.add('visible');
  drawerOverlay.classList.add('visible');
  contextBtn.classList.add('active');
  loadContext();
}
function closeDrawer() {
  drawer.classList.remove('visible');
  drawerOverlay.classList.remove('visible');
  contextBtn.classList.remove('active');
}
contextBtn.addEventListener('click', () => {
  drawer.classList.contains('visible') ? closeDrawer() : openDrawer();
});
ctxCloseBtn.addEventListener('click', closeDrawer);
drawerOverlay.addEventListener('click', closeDrawer);
ctxClearBtn.addEventListener('click', async () => {
  if (!confirm('Delete the stored conversation history (last 5 queries)? This cannot be undone.')) return;
  ctxClearBtn.disabled = true;
  try {
    await fetch('/api/context', { method: 'DELETE' });
    await loadContext();
  } catch {
    ctxRecords.innerHTML = '<p class="ctx-empty" style="color:var(--red)">Failed to clear context.</p>';
  } finally {
    ctxClearBtn.disabled = false;
  }
});

// ── Pipeline rail ────────────────────────────────────────────────────────
function nodeColHtml(s) {
  return `
    <div class="node-col">
      <div class="node" id="node-${s.id}">${s.icon}</div>
      <div class="node-lbl" id="lbl-${s.id}">${s.name}</div>
    </div>`;
}

function buildRail() {
  railTrack.innerHTML = `
    <div class="rail-seg pair">${nodeColHtml(STAGES[0])}${nodeColHtml(STAGES[1])}</div>
    <div class="rail-seg pair">${nodeColHtml(STAGES[2])}${nodeColHtml(STAGES[3])}</div>
    <div class="rail-seg">${nodeColHtml(STAGES[4])}</div>
    <div class="rail-seg">${nodeColHtml(STAGES[5])}</div>
  `;
}

function setNodeState(id, state) {
  const node = document.getElementById(`node-${id}`);
  const lbl  = document.getElementById(`lbl-${id}`);
  if (!node) return;
  node.className = `node ${state}`;
  if (state === 'done') node.textContent = '✓';
  else node.textContent = STAGES.find(s => s.id === id).icon;
  if (lbl) lbl.style.color = state === 'running' ? 'var(--cyan)' : state === 'done' ? 'var(--green)' : '';
}

// ── Skeleton helper ──────────────────────────────────────────────────────
function makeSkeleton(id) {
  const s = document.createElement('div');
  s.className = 'skel';
  s.id = id;
  s.innerHTML = `<div class="skel-line" style="width:35%"></div><div class="skel-line"></div><div class="skel-line" style="width:80%"></div><div class="skel-line" style="width:55%"></div>`;
  return s;
}
function removeSkel(id) { document.getElementById(id)?.remove(); }

// ── Row helpers (side-by-side pairs) ────────────────────────────────────
function getAnswerRow() {
  if (parallelAnswerRow) return parallelAnswerRow;
  const row = document.createElement('div');
  row.className = 'pair-row';
  stream.appendChild(row);
  scrollTo(row);
  parallelAnswerRow = row;
  return row;
}
function getVerifyRow() {
  if (parallelVerifyRow) return parallelVerifyRow;
  const row = document.createElement('div');
  row.className = 'pair-row';
  stream.appendChild(row);
  scrollTo(row);
  parallelVerifyRow = row;
  return row;
}

// ── Card builders ────────────────────────────────────────────────────────
function cardShell(id, edge, icon, title, tag, bodyHtml) {
  const c = document.createElement('div');
  c.className = `stream-card ${edge}`;
  c.id = id;
  c.innerHTML = `
    <div class="sc-head">
      <div class="sc-icon">${icon}</div>
      <div>
        <div class="sc-title">${title}</div>
        ${tag ? `<div class="sc-tag">${tag}</div>` : ''}
      </div>
    </div>
    <div class="sc-body">${bodyHtml}</div>
  `;
  return c;
}

function renderAnswerCard(stageId, name, output) {
  const isLlama = stageId === 1;
  removeSkel(`skel-${stageId}`);
  const row = getAnswerRow();
  const card = cardShell(`card-s${stageId}`, isLlama ? 'llama' : 'mistral', isLlama ? '🦙' : '⚡',
    `Answer — ${name}`, isLlama ? 'LLaMA 3' : 'Mistral', esc(output));
  row.appendChild(card);
  scrollTo(card);
}

function renderVerifierCard(output) {
  removeSkel('skel-3');
  const row = getVerifyRow();
  const card = cardShell('card-verifier', 'verify', '🔍', 'Verifier Feedback', 'Fact-check', esc(output));
  row.appendChild(card);
  scrollTo(card);
}

function renderScoreCard(s1, s2) {
  removeSkel('skel-4');
  const row = getVerifyRow();
  const w1 = s1.composite_score >= s2.composite_score;
  const card = cardShell('card-scores', 'score', '📊', 'Similarity Scores', 'Embedding-based', `
    <div class="score-block">
      <div class="score-name" style="color:var(--llama)">LLM 1 — LLaMA 3</div>
      ${scoreRow('Semantic sim.', s1.semantic_similarity, 'fill-sem-1')}
      ${scoreRow(`Length (${s1.word_count}w)`, s1.length_score, 'fill-len-1')}
    </div>
    <div class="score-block">
      <div class="score-name" style="color:var(--mistral)">LLM 2 — Mistral</div>
      ${scoreRow('Semantic sim.', s2.semantic_similarity, 'fill-sem-2')}
      ${scoreRow(`Length (${s2.word_count}w)`, s2.length_score, 'fill-len-2')}
    </div>
    <div class="win-tag">🏆 Similarity winner: ${w1 ? 'LLaMA 3' : 'Mistral'}</div>
  `);
  row.appendChild(card);
  scrollTo(card);
  setTimeout(() => {
    ['fill-sem-1','fill-len-1','fill-sem-2','fill-len-2'].forEach(cls => {
      const el = card.querySelector('.' + cls);
      if (!el) return;
      const val = cls.includes('sem-1') ? s1.semantic_similarity : cls.includes('len-1') ? s1.length_score
                : cls.includes('sem-2') ? s2.semantic_similarity : s2.length_score;
      el.style.width = `${(val * 100).toFixed(1)}%`;
    });
  }, 50);
}
function scoreRow(label, val, fillClass) {
  return `<div class="score-row">
    <span class="score-lbl">${label}</span>
    <div class="score-track"><div class="score-fill ${fillClass}" style="width:0"></div></div>
    <span class="score-num">${val}</span>
  </div>`;
}

function renderJudgeCard(output) {
  removeSkel('skel-5');
  const card = cardShell('card-judge', 'judge', '⚖️', 'Judge Evaluation & Verdict', '5-criteria rubric', md(output));
  stream.appendChild(card);
  scrollTo(card);
}

function renderFinalCard(output) {
  removeSkel('skel-6');
  const card = cardShell('card-final', 'final', '✦', '✦ Final Combined Answer', 'Verified & synthesised', md(output));
  const btn = document.createElement('button');
  btn.className = 'copy-btn';
  btn.textContent = '📋 Copy Answer';
  btn.onclick = () => {
    navigator.clipboard.writeText(output).then(() => {
      btn.className = 'copy-btn copied'; btn.textContent = '✓ Copied!';
      setTimeout(() => { btn.className = 'copy-btn'; btn.textContent = '📋 Copy Answer'; }, 2000);
    });
  };
  card.appendChild(btn);
  stream.appendChild(card);
  scrollTo(card);
}

function renderErrorCard(message) {
  const e = document.createElement('div');
  e.className = 'error-card';
  e.innerHTML = `<strong>Pipeline Error</strong>${esc(message)}`;
  stream.appendChild(e);
  scrollTo(e);
}

// ── Confidence gauge + regenerate banner (NEW — not in the classic UI) ──
function updateConfidence(p) {
  confidenceCard.classList.add('visible');
  const pct = Math.round(p.weighted_score * 100);
  const thresholdPct = Math.round(p.threshold * 100);
  const color = p.weighted_score >= p.threshold ? 'var(--green)' : (pct >= thresholdPct - 10 ? 'var(--gold)' : 'var(--red)');
  gauge.style.setProperty('--pct', pct);
  gauge.style.setProperty('--gauge-color', color);
  gaugeVal.textContent = `${pct}%`;
  gaugeVal.style.color = color;

  const winnerName = p.winner === 'A' ? 'LLaMA 3' : 'Mistral';
  confTitle.textContent = p.weighted_score >= p.threshold
    ? `✓ High confidence — ${winnerName} wins (attempt ${p.attempt})`
    : `⚠ Low confidence — ${winnerName} wins, but below threshold (attempt ${p.attempt})`;

  // Both the Judge (qwen3.5:9b) and Verifier (phi3) now independently score all
  // 8 rubric factors out of 80 each - showing both totals surfaces their
  // (dis)agreement instead of only the Judge's final numbers.
  const totalsLines = [];
  if (p.total_a !== undefined && p.total_b !== undefined) {
    totalsLines.push(`Judge totals: LLaMA 3 <b>${p.total_a}/80</b> &middot; Mistral <b>${p.total_b}/80</b>`);
  }
  if (p.verifier_total_a !== undefined && p.verifier_total_b !== undefined) {
    totalsLines.push(`Verifier totals: LLaMA 3 <b>${p.verifier_total_a}/80</b> &middot; Mistral <b>${p.verifier_total_b}/80</b>`);
  }
  confSub.innerHTML = `Threshold: ${thresholdPct}% &middot; Weighted blend of judge rubric, similarity, cross-model agreement, verifier-judge agreement, Wikipedia, and Wikidata.${totalsLines.length ? '<br>' + totalsLines.join('<br>') : ''}`;

  const chip = (label, val) => `<span class="chip">${label}: <b>${val === null || val === undefined ? 'n/a' : Math.round(val * 100) + '%'}</b></span>`;
  const sourceChip = (label, source) => source ? `<span class="chip">${label} src: <b>${esc(source)}</b></span>` : '';
  confChips.innerHTML =
    chip('Judge', p.judge) + chip('Similarity', p.similarity) + chip('Model agreement', p.agreement) +
    chip('Verifier↔Judge', p.verifier_judge_agreement) +
    chip('Wikipedia', p.wikipedia) + sourceChip('Wikipedia', p.wikipedia_source) +
    chip('Wikidata', p.wikidata) + sourceChip('Wikidata', p.wikidata_source);
}

function showRegenerating(p) {
  regenBanner.classList.add('visible');
  const thresholdPct = Math.round(p.threshold * 100);
  const scoreText = (p.previous_score !== null && p.previous_score !== undefined)
    ? `Score ${Math.round(p.previous_score * 100)}% is below the ${thresholdPct}% threshold`
    : `Confidence below ${thresholdPct}%`;
  regenText.textContent = `${scoreText} — regenerating both answers (attempt ${p.attempt})…`;
  scrollTo(regenBanner);
}
function hideRegenerating() {
  regenBanner.classList.remove('visible');
}

// ── Timer ────────────────────────────────────────────────────────────────
function startTimer() {
  startTime = Date.now();
  timerDisplay.classList.add('visible');
  timerInterval = setInterval(() => {
    timerDisplay.textContent = `⏱ Pipeline running… ${((Date.now() - startTime) / 1000).toFixed(0)}s`;
  }, 1000);
}
function stopTimer(elapsed) {
  clearInterval(timerInterval);
  timerDisplay.textContent = `✓ Pipeline completed in ${elapsed}s`;
}

// ── Reset ────────────────────────────────────────────────────────────────
function resetUI() {
  parallelAnswerRow = null;
  parallelVerifyRow = null;
  stream.innerHTML = '';
  postActions.hidden = true;
  timerDisplay.classList.remove('visible');
  timerDisplay.textContent = '';
  confidenceCard.classList.remove('visible');
  hideRegenerating();
  buildRail();
  STAGES.forEach(s => setNodeState(s.id, ''));
}

// ── SSE pipeline ─────────────────────────────────────────────────────────
async function runPipeline(query) {
  resetUI();
  railWrap.classList.add('visible');
  submitBtn.disabled = true;
  submitBtn.classList.add('loading');
  startTimer();

  try {
    const useRag = document.getElementById('rag-checkbox').checked;
    const res = await fetch('/api/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, use_rag: useRag }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

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
    renderErrorCard(err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove('loading');
    checkHealth();
  }
}

// ── SSE event dispatcher ─────────────────────────────────────────────────
function handleEvent(p) {
  // Stage started
  if (p.status === 'running') {
    setNodeState(p.stage, 'running');
    const target = (p.stage === 1 || p.stage === 2) ? getAnswerRow()
                  : (p.stage === 3 || p.stage === 4) ? getVerifyRow()
                  : stream;
    if (!document.getElementById(`skel-${p.stage}`)) {
      const skel = makeSkeleton(`skel-${p.stage}`);
      target.appendChild(skel);
      scrollTo(skel);
    }
    return;
  }

  // Context loaded (RAG-independent rolling history indicator)
  if (p.loaded !== undefined) {
    ctxBadge.textContent = p.entries;
    ctxBadge.style.display = 'inline-flex';
    return;
  }

  // RAG chunks retrieved — informational, no dedicated card needed beyond console
  if (p.retrieved !== undefined) {
    return;
  }

  // Regenerate — low confidence, retrying stages 1-5
  if (p.reason !== undefined) {
    showRegenerating(p);
    // Reset stage 1-5 nodes back to idle so the rail visibly restarts
    [1,2,3,4,5].forEach(id => setNodeState(id, ''));
    parallelAnswerRow = null;
    parallelVerifyRow = null;
    return;
  }

  // Confidence score for the winning answer this attempt
  if (p.weighted_score !== undefined) {
    hideRegenerating();
    updateConfidence(p);
    return;
  }

  // Stage result
  if (p.output !== undefined) {
    const { stage, name, output } = p;
    setNodeState(stage, 'done');
    removeSkel(`skel-${stage}`);

    if (stage === 1) renderAnswerCard(1, name, output);
    else if (stage === 2) renderAnswerCard(2, name, output);
    else if (stage === 3) renderVerifierCard(output);
    else if (stage === 5) renderJudgeCard(output);
    else if (stage === 6) {
      renderFinalCard(output);
      postActions.hidden = false;
      scrollTo(postActions);
      loadContext();
    }
    return;
  }

  // Similarity scores
  if (p.scores1 !== undefined) {
    setNodeState(4, 'done');
    renderScoreCard(p.scores1, p.scores2);
    return;
  }

  // Done
  if (p.elapsed !== undefined) {
    stopTimer(p.elapsed);
    return;
  }

  // Error
  if (p.message !== undefined) {
    stopTimer(((Date.now() - startTime) / 1000).toFixed(1));
    renderErrorCard(p.message);
  }
}

// ── New Question button ───────────────────────────────────────────────────
newQBtn.addEventListener('click', () => {
  resetUI();
  railWrap.classList.remove('visible');
  queryCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  queryInput.value = '';
  charCount.textContent = '0 characters';
  queryInput.focus();
});

// ── Input listeners ────────────────────────────────────────────────────────
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
    queryInput.style.borderColor = 'var(--red)';
    setTimeout(() => { queryInput.style.borderColor = ''; }, 1500);
    return;
  }
  runPipeline(q);
});

// ── RAG file uploader (multi-file) ─────────────────────────────────────────
async function uploadRagDocs(input) {
  const files = Array.from(input.files || []);
  if (!files.length) return;
  const statusMsg = document.getElementById('rag-status-msg');

  let totalChunks = 0;
  let successCount = 0;

  for (let i = 0; i < files.length; i++) {
    const file = files[i];
    if (statusMsg) statusMsg.textContent = `Uploading ${i + 1}/${files.length}: ${file.name}…`;
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch('/api/rag/upload', { method: 'POST', body: formData }).then(r => r.json());
      if (res.ok) { successCount++; totalChunks += (res.chunks_created || 0); }
    } catch (err) {
      console.error(`Error uploading ${file.name}:`, err);
    }
  }

  if (statusMsg) {
    statusMsg.textContent = files.length === 1
      ? `✓ Indexed ${files[0].name} (${totalChunks} chunks)`
      : `✓ Indexed ${successCount}/${files.length} docs (${totalChunks} total chunks)`;
  }
  input.value = '';
}
window.uploadRagDocs = uploadRagDocs;

// ── Init ─────────────────────────────────────────────────────────────────
checkHealth();
loadContext();
buildRail();
