/* Roitelet web client.
 *
 * Vanilla JS, no build step. Talks to the FastAPI backend over its native
 * endpoints (no SSE — the OpenAI-compat streaming chunks an already-complete
 * answer, so honest loading state beats fake typing).
 *
 * All visible strings flow through `window.RoiteletI18n.t(key, vars)` so
 * the EN / FR toggle in the sidebar can re-render the whole UI without
 * a reload. Static elements in index.html carry `data-i18n` attributes;
 * dynamic strings here pull from `t()` directly.
 */

const $ = (id) => document.getElementById(id);
// ``t`` is provided as a global by ``web/i18n.js`` (a top-level
// ``function t(...)`` declaration). We rely on that global directly to
// avoid a duplicate-identifier collision at script-eval time, which
// would silently brick every dynamic surface that uses it.
const escapeHtml = (s) => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const state = {
  conversationId: null,
  messages: [],          // {role, content, metadata?}
  conversations: [],
  busy: false,
  attachments: [],       // File objects queued for the next send
  urlAttachments: [],    // string URLs queued for Firecrawl scrape on send
  allowVlms: false,      // mirrors the persisted preference; gates image attachments
  // Per-turn preferences. These start at the persisted-settings defaults
  // and the user adjusts them via the sliders popover next to the send
  // button. They reset to the persisted defaults on every new chat.
  prefs: {
    independence: false,
    pseudonymize: false,
    topK: 2,
    maxCostUsd: null,
  },
  // Persisted defaults — mirror of the AppSettingsPayload server-side.
  // Kept separately so "reset to defaults" in the popover works without
  // hitting the network.
  prefDefaults: {
    independence: false,
    pseudonymize: false,
    topK: 2,
  },
};

const AUDIO_RX = /\.(wav|mp3|m4a|flac|ogg|opus|aac)$/i;
const IMAGE_RX = /\.(jpe?g|png|webp|gif|bmp|heic|heif)$/i;
const PDF_RX   = /\.pdf$/i;

function classifyFile(f) {
  const t = (f.type || '').toLowerCase();
  if (t.startsWith('audio/') || AUDIO_RX.test(f.name)) return 'audio';
  if (t.startsWith('image/') || IMAGE_RX.test(f.name)) return 'image';
  if (t === 'application/pdf' || PDF_RX.test(f.name)) return 'pdf';
  return null;
}

// ─── API helpers ─────────────────────────────────────────────────────────────

async function apiGet(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${await r.text()}`);
  return r.json();
}

async function apiPostMultipart(path, formData) {
  const r = await fetch(path, {method: 'POST', body: formData});
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} — ${await r.text()}`);
  return r.json();
}

// ─── UI rendering ────────────────────────────────────────────────────────────

function renderMessages() {
  const inner = $('messagesInner');
  inner.innerHTML = '';
  if (state.messages.length === 0) {
    // Welcome state — every visible string carries data-i18n so the
    // language toggle re-renders correctly. We also write the current
    // value as the initial textContent so the screen reads correctly
    // even before applyStaticTranslations() fires.
    inner.innerHTML = `
      <div id="welcome" class="text-center py-24">
        <img src="/assets/roitelet.png" alt="Roitelet" width="72" height="72" class="mx-auto mb-5 rounded-[10px] shadow-lg">
        <h2 class="text-[22px] font-semibold tracking-tight" data-i18n="welcome.title">${escapeHtml(t('welcome.title'))}</h2>
        <p class="text-[14px] text-gray-500 dark:text-gray-400 mt-2 max-w-md mx-auto leading-relaxed" data-i18n="welcome.subtitle">${escapeHtml(t('welcome.subtitle'))}</p>
        <p class="text-[12px] text-gray-400 dark:text-gray-500 mt-4 max-w-md mx-auto leading-relaxed" data-i18n-html="welcome.tip">${t('welcome.tip')}</p>
      </div>`;
    return;
  }
  for (const m of state.messages) inner.appendChild(messageNode(m));
  scrollToBottom();
}

function messageNode(m) {
  const wrap = document.createElement('div');
  wrap.className = m.role === 'user' ? 'flex flex-col items-end' : 'flex justify-start';

  const bubble = document.createElement('div');
  if (m.role === 'user') {
    bubble.className = 'bg-sysblue text-white px-4 py-2.5 max-w-[78%] text-[15px] leading-relaxed shadow-sm whitespace-pre-wrap';
    bubble.textContent = m.content;
    wrap.appendChild(bubble);
    // Pseudonymization audit lives on the user bubble (it's the user's
    // prompt that was rewritten). Show the diff inline so the user can
    // verify what actually went to remote candidates.
    if (m.metadata?.pseudonymization) {
      wrap.appendChild(pseudonymizationNode(m.metadata.pseudonymization));
    }
  } else {
    bubble.className = 'bg-gray-100 dark:bg-[#2c2c2e] px-4 py-3 max-w-[85%] text-[15px] leading-relaxed shadow-sm prose-msg';
    bubble.innerHTML = marked.parse(m.content || '', {breaks: true, gfm: true});
    // Image-gen assistant turns: render the generated bytes inline. The
    // static mount at /data/images/<uuid>.png serves them straight from
    // disk, so the <img> src is the on-disk path the API returned with
    // /data prepended (the API returns absolute paths; we use basename).
    if (m.metadata?.imagegen?.images?.length) {
      bubble.appendChild(imageGenNode(m.metadata.imagegen));
    } else if (m.metadata) {
      bubble.appendChild(metadataNode(m.metadata));
    }
    wrap.appendChild(bubble);
  }
  return wrap;
}

// Audit affordance — collapsible card with the substitution table the
// pseudonymizer produced. Lives under the user bubble; the user can
// open it to see exactly what left the box on this turn.
function pseudonymizationNode(audit) {
  const det = document.createElement('details');
  det.className = 'mt-1.5 max-w-[78%] text-[11px] text-gray-500 dark:text-gray-400';
  const summary = document.createElement('summary');
  summary.className = 'cursor-pointer select-none hover:text-gray-700 dark:hover:text-gray-200 transition-colors flex items-center gap-1.5';
  const count = (audit.mappings || []).length;
  let summaryKey;
  if (count === 0) summaryKey = 'audit.summary.zero';
  else if (count === 1) summaryKey = 'audit.summary';
  else summaryKey = 'audit.summary.plural';
  const summaryText = document.createElement('span');
  summaryText.textContent = t(summaryKey, {count});
  summary.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M12 1l3 6 6 .9-4.5 4.3 1 6.3L12 15.8 6.5 18.5l1-6.3L3 7.9 9 7z"/></svg>`;
  summary.appendChild(summaryText);
  det.appendChild(summary);

  const body = document.createElement('div');
  body.className = 'mt-2 p-3 rounded-[10px] bg-black/5 dark:bg-white/[0.05] space-y-2';

  const sentLabel = document.createElement('div');
  sentLabel.className = 'text-gray-400';
  sentLabel.textContent = t('audit.sentLabel');
  body.appendChild(sentLabel);

  const sentPre = document.createElement('pre');
  sentPre.className = 'whitespace-pre-wrap text-gray-700 dark:text-gray-300 text-[12px] leading-relaxed';
  sentPre.textContent = audit.pseudonymized_prompt || '';
  body.appendChild(sentPre);

  if (count) {
    const tableLabel = document.createElement('div');
    tableLabel.className = 'text-gray-400 mt-2';
    tableLabel.textContent = t('audit.tableLabel');
    body.appendChild(tableLabel);

    const table = document.createElement('div');
    table.className = 'grid grid-cols-[auto_1fr_auto_1fr] gap-x-2 gap-y-0.5 text-[11px]';
    for (const mapping of audit.mappings) {
      const kind = document.createElement('span');
      kind.className = 'text-gray-400 tabular-nums';
      kind.textContent = mapping.kind;
      const original = document.createElement('span');
      original.className = 'text-gray-700 dark:text-gray-200 truncate';
      original.textContent = mapping.original;
      const arrow = document.createElement('span');
      arrow.className = 'text-gray-400';
      arrow.textContent = '→';
      const substitute = document.createElement('span');
      substitute.className = 'text-sysblue truncate';
      substitute.textContent = mapping.substitute;
      table.append(kind, original, arrow, substitute);
    }
    body.appendChild(table);
  } else {
    const noPii = document.createElement('div');
    noPii.className = 'italic text-gray-400';
    noPii.textContent = t('audit.empty');
    body.appendChild(noPii);
  }

  const timing = document.createElement('div');
  timing.className = 'text-gray-400 pt-1 border-t border-black/5 dark:border-white/10';
  timing.textContent = t('audit.timing', {
    model: audit.model_id,
    fwd: (audit.forward_latency_s || 0).toFixed(2),
    rev: (audit.reverse_latency_s || 0).toFixed(2),
    repair: audit.repair_used ? t('audit.repair.used') : t('audit.repair.skipped'),
  });
  body.appendChild(timing);

  det.appendChild(body);
  return det;
}

function imageGenNode(imgGen) {
  const div = document.createElement('div');
  div.className = 'mt-3 grid grid-cols-1 sm:grid-cols-2 gap-2';
  for (const img of imgGen.images || []) {
    if (img.error) {
      const err = document.createElement('div');
      err.className = 'text-[12px] text-red-500';
      err.textContent = `Image error: ${img.error}`;
      div.appendChild(err);
      continue;
    }
    if (!img.path) continue;
    const basename = img.path.split('/').pop();
    const node = document.createElement('img');
    node.src = `/data/images/${basename}`;
    node.alt = `Generated by ${imgGen.model_id}`;
    node.loading = 'lazy';
    node.className = 'rounded-[10px] shadow-sm';
    div.appendChild(node);
  }
  return div;
}

function metadataNode(meta) {
  const det = document.createElement('details');
  det.className = 'mt-3 pt-3 border-t border-black/5 dark:border-white/10 text-[12px] text-gray-500 dark:text-gray-400';
  const summary = document.createElement('summary');
  summary.className = 'cursor-pointer select-none hover:text-gray-700 dark:hover:text-gray-200 transition-colors';
  const winners = (meta.synthesis?.winning_model_ids || []).map(shortModel).join(', ');
  const selected = (meta.router?.selected_model_ids || []).map(shortModel).join(' · ');
  const caps = Object.entries(meta.router?.categories || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([k, v]) => `${k} ${(v * 100).toFixed(0)}%`)
    .join(' · ');
  summary.textContent = `Roitelet · ${selected || '—'}${winners ? '  ·  won: ' + winners : ''}`;
  det.appendChild(summary);

  // Latency breakdown: candidate fan-out is bounded by the slowest
  // candidate; the judge runs on top of that; the total is the full
  // pipeline wall-clock (router + gather + judge + telemetry write).
  const candMax = Math.max(0, ...((meta.responses || []).map(r => r.latency_s || 0)));
  const judgeLat = meta.synthesis?.latency_s || 0;
  const totalLat = meta.total_latency_s ?? (candMax + judgeLat);
  const latencyLine = `total ${totalLat.toFixed(1)}s = cand_max ${candMax.toFixed(1)}s + judge ${judgeLat.toFixed(1)}s`;

  const body = document.createElement('div');
  body.className = 'mt-2 space-y-2';
  body.innerHTML = `
    <div><span class="text-gray-400">Capabilities</span> &nbsp;${escapeHtml(caps)}</div>
    <div><span class="text-gray-400">Latency</span> &nbsp;<span class="tabular-nums">${escapeHtml(latencyLine)}</span></div>
    <div class="space-y-1.5">
      ${(meta.responses || []).map(r => `
        <div class="flex items-baseline justify-between gap-3">
          <span class="truncate">${escapeHtml(shortModel(r.model_id))}</span>
          <span class="text-gray-400 text-[11px] tabular-nums">${r.latency_s?.toFixed(2) || '—'}s · $${(r.cost_usd || 0).toFixed(4)}</span>
        </div>`).join('')}
    </div>`;
  det.appendChild(body);
  return det;
}

function shortModel(id) {
  if (!id) return '';
  const parts = id.split('/');
  return parts.slice(-2).join('/');
}

function renderConversationList() {
  const list = $('conversationList');
  list.innerHTML = '';
  for (const c of state.conversations) {
    const a = document.createElement('button');
    const active = c.conversation_id === state.conversationId;
    a.className = `w-full text-left px-2.5 py-2 text-[13px] truncate transition-colors ${
      active
        ? 'bg-black/[0.06] dark:bg-white/[0.1] text-gray-900 dark:text-white'
        : 'text-gray-700 dark:text-gray-300 hover:bg-black/[0.03] dark:hover:bg-white/[0.05]'
    }`;
    a.textContent = c.title || t('misc.untitled');
    a.onclick = () => loadConversation(c.conversation_id);
    list.appendChild(a);
  }
}

function setBusy(busy, label) {
  state.busy = busy;
  $('sendBtn').disabled = busy;
  $('prompt').disabled = busy;
  $('sendIcon').classList.toggle('hidden', busy);
  $('sendSpinner').classList.toggle('hidden', !busy);
  $('statusDot').className = `w-1.5 h-1.5 rounded-full ${busy ? 'bg-sysblue animate-pulse' : 'bg-uGreen'}`;
  if (busy) {
    $('statusText').textContent = label || t('header.status.thinking');
  } else {
    renderStatusPill();
  }
}

function showThinking() {
  const node = document.createElement('div');
  node.id = 'thinkingNode';
  node.className = 'flex justify-start';
  node.innerHTML = `
    <div class="bg-gray-100 dark:bg-[#2c2c2e] px-4 py-3.5 shadow-sm flex items-center gap-1.5">
      <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-gray-400 dark:bg-gray-500"></span>
      <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-gray-400 dark:bg-gray-500"></span>
      <span class="thinking-dot w-1.5 h-1.5 rounded-full bg-gray-400 dark:bg-gray-500"></span>
    </div>`;
  $('messagesInner').appendChild(node);
  scrollToBottom();
}

function hideThinking() {
  $('thinkingNode')?.remove();
}

function scrollToBottom() {
  const m = $('messages');
  m.scrollTop = m.scrollHeight;
}

function showToast(text, kind = 'info') {
  const t = $('toast');
  $('toastText').textContent = text;
  t.classList.remove('hidden');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => t.classList.add('hidden'), 3500);
}

// ─── Actions ─────────────────────────────────────────────────────────────────

async function refreshConversations() {
  try {
    state.conversations = await apiGet('/api/conversations');
    renderConversationList();
  } catch (err) {
    showToast(t('toast.loadFailed', {message: err.message}));
  }
}

async function refreshPreferences() {
  // Mirror the server's persisted flags so the local gates + the status
  // pill match what the backend will do on the next turn, and so the
  // per-turn popover starts at the user's defaults.
  try {
    const cur = await apiGet('/api/settings');
    state.allowVlms = !!cur.enable_vlms;
    state.prefDefaults.independence = !!cur.independence_local_only;
    state.prefDefaults.pseudonymize = !!cur.enable_pseudonymization;
    state.prefs.independence = state.prefDefaults.independence;
    state.prefs.pseudonymize = state.prefDefaults.pseudonymize;
    renderStatusPill();
    renderPrefsDot();
  } catch { /* harmless: defaults stay */ }
}

// Tiny helper: rewrite the header's status line so the user sees
// at-a-glance which non-default preferences are in effect. Each tag is
// a plain-language phrase, not a code shorthand.
function renderStatusPill() {
  const text = $('statusText');
  if (!text || state.busy) return;
  const tags = [];
  if (state.prefs.pseudonymize) tags.push(t('header.status.tag.pseudo'));
  if (state.prefs.independence) tags.push(t('header.status.tag.local'));
  text.textContent = tags.length
    ? `${t('header.status.ready')} · ${tags.join(' · ')}`
    : t('header.status.ready');
}

// Show a blue dot on the sliders icon whenever the per-turn prefs
// diverge from the persisted defaults. Visible state for visible state.
function renderPrefsDot() {
  const dot = $('prefsDot');
  if (!dot) return;
  const dirty =
    state.prefs.independence !== state.prefDefaults.independence ||
    state.prefs.pseudonymize !== state.prefDefaults.pseudonymize ||
    state.prefs.topK !== state.prefDefaults.topK ||
    state.prefs.maxCostUsd !== null;
  dot.classList.toggle('hidden', !dirty);
}

// ─── Per-turn preferences popover ────────────────────────────────────────────

function openPrefsPopover() {
  const popover = $('prefsPopover');
  const btn = $('prefsBtn');
  if (!popover || !btn) return;
  // Position above the sliders button, right-aligned to the composer.
  const rect = btn.getBoundingClientRect();
  popover.style.bottom = (window.innerHeight - rect.top + 8) + 'px';
  popover.style.left = Math.max(8, rect.right - 280) + 'px';
  // Seed the inputs from current per-turn state.
  $('prefIndependence').checked = state.prefs.independence;
  $('prefPseudonymize').checked = state.prefs.pseudonymize;
  $('prefTopK').value = state.prefs.topK;
  $('prefMaxCost').value = state.prefs.maxCostUsd ?? '';
  popover.classList.remove('hidden');
  // Click-outside-to-close. Bound on the next tick so the click that
  // opened the popover doesn't immediately close it.
  setTimeout(() => document.addEventListener('click', closePrefsOnOutside), 0);
}

function closePrefsPopover() {
  $('prefsPopover')?.classList.add('hidden');
  document.removeEventListener('click', closePrefsOnOutside);
}

function closePrefsOnOutside(event) {
  const popover = $('prefsPopover');
  if (!popover) return;
  if (popover.contains(event.target) || $('prefsBtn').contains(event.target)) return;
  closePrefsPopover();
}

function wirePrefsPopover() {
  $('prefsBtn')?.addEventListener('click', (e) => {
    e.stopPropagation();
    const popover = $('prefsPopover');
    if (popover && !popover.classList.contains('hidden')) closePrefsPopover();
    else openPrefsPopover();
  });
  $('prefIndependence')?.addEventListener('change', (e) => {
    state.prefs.independence = e.target.checked;
    renderStatusPill();
    renderPrefsDot();
  });
  $('prefPseudonymize')?.addEventListener('change', (e) => {
    state.prefs.pseudonymize = e.target.checked;
    renderStatusPill();
    renderPrefsDot();
  });
  $('prefTopK')?.addEventListener('change', (e) => {
    const raw = parseInt(e.target.value, 10);
    state.prefs.topK = Math.min(8, Math.max(1, Number.isNaN(raw) ? 2 : raw));
    e.target.value = state.prefs.topK;
    renderPrefsDot();
  });
  $('prefMaxCost')?.addEventListener('change', (e) => {
    const raw = parseFloat(e.target.value);
    state.prefs.maxCostUsd = Number.isFinite(raw) && raw > 0 ? raw : null;
    if (state.prefs.maxCostUsd === null) e.target.value = '';
    renderPrefsDot();
  });
  $('prefsToSettings')?.addEventListener('click', () => {
    closePrefsPopover();
    openSettings();
  });
}

async function loadConversation(id) {
  try {
    setBusy(true, 'Loading…');
    const convo = await apiGet(`/api/conversations/${id}`);
    state.conversationId = convo.conversation_id;
    state.messages = convo.messages.map(m => ({
      role: m.role,
      content: m.content,
      metadata: m.metadata && Object.keys(m.metadata).length ? m.metadata : null,
    }));
    $('conversationTitle').textContent = convo.title || 'Chat';
    renderMessages();
    renderConversationList();
  } catch (err) {
    showToast(t('toast.convoFailed', {message: err.message}));
  } finally {
    setBusy(false);
  }
}

function newChat() {
  state.conversationId = null;
  state.messages = [];
  state.attachments = [];
  state.urlAttachments = [];
  // Reset per-turn prefs to the persisted defaults — a new chat
  // shouldn't inherit the previous chat's one-off overrides.
  state.prefs.independence = state.prefDefaults.independence;
  state.prefs.pseudonymize = state.prefDefaults.pseudonymize;
  state.prefs.topK = state.prefDefaults.topK;
  state.prefs.maxCostUsd = null;
  renderAttachments();
  renderStatusPill();
  renderPrefsDot();
  $('conversationTitle').textContent = 'New chat';
  renderMessages();
  renderConversationList();
  $('prompt').focus();
}

// ─── Slash-command pre-dispatch ──────────────────────────────────────────────
//
// Mirrors `core/commands.py`: only the leading token matters and unknown
// commands pass through as plain text. Done client-side so an `/image`
// prompt goes straight to `/api/images` instead of bouncing off the
// chat endpoint's 400. Keep this list in sync with `core/commands.py`.
const IMAGE_CMD_RX  = /^\/(image|image-gen|img)\b\s*/i;
const SPEECH_CMD_RX = /^\/(speech|stt|transcribe)\b\s*/i;

// Helpers for the Personal panel — opening the modal calls the API
// once, the ingest button calls it again, the status is rendered
// inline. Defined at module scope so they're hoisted alongside the
// other API helpers.
async function fetchPersonalStatus() {
  try { return await apiGet('/api/personal'); }
  catch { return null; }
}
async function triggerPersonalIngest(force) {
  return apiPost(`/api/personal/ingest?force=${force ? 'true' : 'false'}`, {});
}
async function refreshPersonalSummary() {
  const s = await fetchPersonalStatus();
  const el = document.getElementById('personalSummary');
  if (!el) return;
  if (!s) { el.textContent = '—'; return; }
  el.textContent = t('personal.subtitle.summary', {wiki: s.wiki, inbox: s.inbox, mode: s.mode});
}

// ─── Personal embedding viz (Karpathy-style) ─────────────────────────────────
//
// 2-D PCA scatter where each dot is one wiki chunk and color is the
// source file. The whole thing renders inline SVG; no extra deps. The
// server projects via numpy SVD and returns the coordinates.

// Aligned to https://harchaoui.org/warith/colors — the 8 base accents
// (Red, Orange, Yellow, Green, Blue, Turquoise, Purple, Pink). Gray
// closes the cycle as a neutral so 9 source files still get distinct
// colors before we wrap.
const VIZ_PALETTE = [
  '#007AFF', '#FF9500', '#FFCC00', '#28CD41',
  '#79DBDC', '#AF52DE', '#FF2D55', '#FF3B30',
  '#808080',
];

async function openPersonalViz() {
  let payload;
  try { payload = await apiGet('/api/personal/embeddings'); }
  catch (err) { showToast(t('personal.viz.fetchFailed', {message: err.message})); return; }
  const points = payload?.points || [];
  if (!points.length) {
    showToast(t('personal.viz.empty'));
    return;
  }
  renderVizModal(points);
}

function renderVizModal(points) {
  // Lazy: build the modal nodes on first open so the empty case
  // doesn't bloat the DOM.
  let modal = document.getElementById('vizModal');
  if (!modal) {
    modal = document.createElement('div');
    modal.id = 'vizModal';
    modal.className = 'fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm';
    modal.innerHTML = `
      <div class="bg-white dark:bg-[#1c1c1e] w-[95vw] h-[90vh] flex flex-col overflow-hidden rounded-[10px] shadow-2xl">
        <div class="px-5 py-3 border-b border-gray-200 dark:border-white/[0.08] flex items-center justify-between">
          <div>
            <div class="text-[14px] font-semibold">${escapeHtml(t('personal.viz.title'))}</div>
            <div class="text-[11px] text-gray-500 dark:text-gray-400" id="vizMeta"></div>
          </div>
          <button id="vizClose" class="w-9 h-9 rounded-full flex items-center justify-center hover:bg-black/5 dark:hover:bg-white/10" aria-label="${escapeHtml(t('personal.viz.close'))}">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M18 6L6 18M6 6l12 12"/></svg>
          </button>
        </div>
        <div class="flex-1 relative">
          <svg id="vizSvg" class="w-full h-full"></svg>
          <div id="vizTooltip"
            class="absolute pointer-events-none max-w-[360px] hidden z-10 px-3 py-2 rounded-[10px] bg-black/85 text-white text-[12px] leading-relaxed shadow-lg"></div>
        </div>
      </div>`;
    document.body.appendChild(modal);
    modal.querySelector('#vizClose').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', e => { if (e.target === modal) modal.remove(); });
  }

  const svg = modal.querySelector('#vizSvg');
  const tooltip = modal.querySelector('#vizTooltip');
  const meta = modal.querySelector('#vizMeta');
  const fileToColor = new Map();
  const colorFor = p => {
    if (!fileToColor.has(p)) fileToColor.set(p, VIZ_PALETTE[fileToColor.size % VIZ_PALETTE.length]);
    return fileToColor.get(p);
  };
  const sources = Array.from(new Set(points.map(p => p.path)));
  meta.textContent = t('personal.viz.meta', {points: points.length, sources: sources.length});

  const draw = () => {
    const rect = svg.getBoundingClientRect();
    const W = rect.width, H = rect.height;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.innerHTML = '';
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const margin = 32;
    const sx = x => margin + (W - 2*margin) * (xmax === xmin ? 0.5 : (x - xmin) / (xmax - xmin));
    const sy = y => H - margin - (H - 2*margin) * (ymax === ymin ? 0.5 : (y - ymin) / (ymax - ymin));
    for (const p of points) {
      const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
      c.setAttribute('cx', sx(p.x)); c.setAttribute('cy', sy(p.y));
      c.setAttribute('r', 6); c.setAttribute('fill', colorFor(p.path));
      c.setAttribute('opacity', 0.75);
      c.style.cursor = 'pointer';
      c.addEventListener('mousemove', e => {
        tooltip.classList.remove('hidden');
        tooltip.style.left = (e.clientX - rect.left + 12) + 'px';
        tooltip.style.top  = (e.clientY - rect.top + 12) + 'px';
        tooltip.innerHTML = `<b>${escapeHtml(p.path)}</b> #${p.chunk_index}<br>${escapeHtml(p.text.slice(0, 280))}${p.text.length > 280 ? '…' : ''}`;
      });
      c.addEventListener('mouseleave', () => tooltip.classList.add('hidden'));
      svg.appendChild(c);
    }
  };
  draw();
  // Redraw on window resize so the scatter scales with the modal.
  const onResize = () => { if (document.body.contains(modal)) draw(); else window.removeEventListener('resize', onResize); };
  window.addEventListener('resize', onResize);
}

async function send() {
  const prompt = $('prompt').value.trim();
  const files = state.attachments;
  const urls = state.urlAttachments;
  if ((!prompt && files.length === 0 && urls.length === 0) || state.busy) return;

  const imageMatch  = prompt.match(IMAGE_CMD_RX);
  const speechMatch = prompt.match(SPEECH_CMD_RX);

  const fileLabel = files.length ? files.map(f => `📎 ${f.name}`).join('\n') : '';
  const urlLabel  = urls.length  ? urls.map(u => `🔗 ${u}`).join('\n')  : '';
  const userBubble = [urlLabel, fileLabel, prompt].filter(Boolean).join('\n\n') || t('misc.attachmentOnly');
  state.messages.push({role: 'user', content: userBubble});
  $('prompt').value = '';
  renderMessages();
  showThinking();
  setBusy(true);

  try {
    let res;

    if (imageMatch && !files.length) {
      // `/image <prompt>` → image-gen pipeline. K=1, no fusion.
      const cleanPrompt = prompt.slice(imageMatch[0].length).trim();
      if (!cleanPrompt) throw new Error(t('toast.imageNeedsPrompt'));
      const payload = {
        prompt: cleanPrompt,
        conversation_id: state.conversationId,
        size: '1024x1024',
        n: 1,
      };
      const imgRes = await apiPost('/api/images', payload);
      state.conversationId = imgRes.conversation_id;
      state.messages.push({
        role: 'assistant',
        content: t('toast.imageGenerated', {n: imgRes.images?.length || 0, model: imgRes.model_id}),
        metadata: {imagegen: imgRes},
      });
    } else if (speechMatch && !files.length) {
      // `/speech` without an attachment is meaningless — explain.
      throw new Error(t('toast.speechNeedsAudio'));
    } else if (files.length || urls.length) {
      // Route through the multimodal endpoint when there's any
      // attachment — files OR website URLs (Firecrawl-scraped).
      const fd = new FormData();
      const stripped = speechMatch ? prompt.slice(speechMatch[0].length).trim() : prompt;
      fd.append('prompt', stripped);
      if (state.conversationId) fd.append('conversation_id', state.conversationId);
      fd.append('top_k', String(state.prefs.topK));
      fd.append('allow_vlms', state.allowVlms ? 'true' : 'false');
      fd.append('pseudonymize', state.prefs.pseudonymize ? 'true' : 'false');
      for (const f of files) fd.append('files', f);
      for (const u of urls)  fd.append('urls',  u);
      res = await apiPostMultipart('/api/chat/multimodal', fd);
      finalizeChatResponse(res, prompt);
    } else {
      const prefs = {
        raw_power: 0.7,
        ecofrugality: 0.3,
        independence: state.prefs.independence,
        allow_vlms: state.allowVlms,
        pseudonymize: state.prefs.pseudonymize,
      };
      if (state.prefs.maxCostUsd !== null) prefs.max_cost_usd = state.prefs.maxCostUsd;
      const payload = {
        prompt,
        conversation_id: state.conversationId,
        preferences: prefs,
        top_k: state.prefs.topK,
      };
      res = await apiPost('/api/chat', payload);
      finalizeChatResponse(res, prompt);
    }
    state.attachments = [];
    state.urlAttachments = [];
    renderAttachments();
    hideThinking();
    renderMessages();
    await refreshConversations();
  } catch (err) {
    hideThinking();
    showToast(t('toast.pipelineError', {message: err.message}));
    // Keep the user's message visible; don't auto-rewind.
  } finally {
    setBusy(false);
    $('prompt').focus();
  }
}

// Push the standard `/api/chat` response into the message list.
// Extracted so the `/image` and `/speech` branches can take their own
// shape without forking the whole send() body.
//
// When pseudonymization fired, attach the audit to the most recent
// user message so the diff renders right under the user bubble. The
// optional ``originalPrompt`` argument is the un-stripped text the
// user typed; we use it to identify the message we just pushed in
// ``send()`` (last user entry).
function finalizeChatResponse(res, originalPrompt) {
  state.conversationId = res.conversation_id;
  if (res.pseudonymization) {
    // Walk backwards to find the matching user bubble. The last user
    // entry is almost always the one we just pushed; the loop is a
    // safety net for re-renders triggered by conversation reloads.
    for (let i = state.messages.length - 1; i >= 0; i--) {
      if (state.messages[i].role === 'user') {
        state.messages[i].metadata = {
          ...(state.messages[i].metadata || {}),
          pseudonymization: res.pseudonymization,
        };
        break;
      }
    }
  }
  state.messages.push({
    role: 'assistant',
    content: res.synthesis?.content || t('misc.noAnswer'),
    metadata: {
      router: res.router,
      responses: res.responses,
      synthesis: res.synthesis,
      total_latency_s: res.total_latency_s,
      pseudonymization: res.pseudonymization || null,
    },
  });
}

// ─── Attachments ─────────────────────────────────────────────────────────────

function renderAttachments() {
  const box = $('attachmentChips');
  const hasAny = state.attachments.length || state.urlAttachments.length;
  if (!hasAny) {
    box.classList.add('hidden');
    box.innerHTML = '';
    return;
  }
  box.classList.remove('hidden');
  box.classList.add('flex');
  const fileChips = state.attachments.map((f, i) => `
    <span class="inline-flex items-center gap-1.5 px-2.5 py-1 text-[12px] bg-black/5 dark:bg-white/10 text-gray-700 dark:text-gray-200">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>
      <span class="truncate max-w-[180px]">${escapeHtml(f.name)}</span>
      <button data-rm-file="${i}" class="text-gray-400 hover:text-red-500" title="Remove">×</button>
    </span>`).join('');
  const urlChips = state.urlAttachments.map((u, i) => `
    <span class="inline-flex items-center gap-1.5 px-2.5 py-1 text-[12px] bg-black/5 dark:bg-white/10 text-gray-700 dark:text-gray-200">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M10 13a5 5 0 0 0 7.07 0l3-3a5 5 0 0 0-7.07-7.07l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.07 0l-3 3a5 5 0 0 0 7.07 7.07l1.7-1.7"/></svg>
      <span class="truncate max-w-[260px]">${escapeHtml(u)}</span>
      <button data-rm-url="${i}" class="text-gray-400 hover:text-red-500" title="Remove">×</button>
    </span>`).join('');
  box.innerHTML = fileChips + urlChips;
  for (const btn of box.querySelectorAll('button[data-rm-file]')) {
    btn.addEventListener('click', () => {
      state.attachments.splice(parseInt(btn.dataset.rmFile, 10), 1);
      renderAttachments();
    });
  }
  for (const btn of box.querySelectorAll('button[data-rm-url]')) {
    btn.addEventListener('click', () => {
      state.urlAttachments.splice(parseInt(btn.dataset.rmUrl, 10), 1);
      renderAttachments();
    });
  }
}

function onFilesPicked(fileList) {
  for (const f of fileList) {
    const kind = classifyFile(f);
    if (!kind) {
      showToast(t('toast.skippedAttachment', {name: f.name}));
      continue;
    }
    if (kind === 'image' && !state.allowVlms) {
      showToast(t('toast.skippedVision', {name: f.name}));
      continue;
    }
    state.attachments.push(f);
  }
  renderAttachments();
}

// ─── Settings sheet ──────────────────────────────────────────────────────────

// `csv-list` fields are persisted as `List[str]` server-side but edited
// as a comma-separated string in the UI for one-line tractability. The
// (de)serialisation happens at the form-submit boundary, not in state.
// The Settings sheet has three sections:
//
//   1. Personal panel (rendered first; built dynamically).
//   2. Engines panel — the "+ Add engine" list. Each engine is an
//      OpenAI-compatible endpoint with its own label/URL/key/models.
//      Replaces the previous hardcoded enumeration. Built dynamically
//      from current.custom_engines.
//   3. The fields below — pre-built integrations (Ollama local stack,
//      well-known providers we ship dedicated wiring for), and the
//      routing knobs. Pre-built integrations stay first-class because
//      they have curated bootstrap priors keyed on their provider
//      prefix (``openai/...``, ``openrouter/...``); rebuilding them
//      as custom engines would lose those priors.
// Settings catalogue. `labelKey` is an i18n key fed to ``t()`` at render
// time so the language toggle updates labels without reloading. The
// ``section`` field drives the progressive-disclosure layout: 'basics'
// renders inline, 'advanced' goes under a collapsed <details>.
const SETTINGS_FIELDS = [
  // Basics — what a non-tech user is most likely to touch.
  {key: 'ollama_base_url',                  labelKey: 'field.ollama_base_url',         type: 'text',           placeholder: 'http://localhost:11434',  section: 'basics'},
  // ``select-local-model`` becomes a <select> whose options are the
  // local Ollama models the registry reports. Far less error-prone
  // than a free-text input for a non-tech user who has to match the
  // exact tag (`qwen3:8b`, not `qwen3` or `Qwen3:8B`).
  {key: 'local_synthesis_model',            labelKey: 'field.local_synthesis_model',   type: 'select-local-model',                                      section: 'basics'},
  {key: 'enable_pseudonymization',          labelKey: 'field.enable_pseudonymization', type: 'checkbox',                                                 section: 'basics'},
  {key: 'independence_local_only',          labelKey: 'field.independence_local_only', type: 'checkbox',                                                 section: 'basics'},
  {key: 'enable_vlms',                      labelKey: 'field.enable_vlms',             type: 'checkbox',                                                 section: 'basics'},

  // Advanced — power-user knobs. Hidden behind a <details> in openSettings.
  {key: 'local_vlm_model',                  labelKey: 'field.local_vlm_model',         type: 'select-local-model',                                      section: 'advanced'},
  {key: 'selected_ollama_models',           labelKey: 'field.selected_ollama_models',  type: 'csv-list',       placeholder: 'phi4-mini:3.8b, gemma3:4b', section: 'advanced'},
  // ``allow_blank`` lets the redactor model fall back to the judge.
  {key: 'pseudo_model_id',                  labelKey: 'field.pseudo_model_id',         type: 'select-local-model', allow_blank: true,                    section: 'advanced'},
  {key: 'openrouter_api_key',               labelKey: 'field.openrouter_api_key',      type: 'password',                                          section: 'advanced'},
  {key: 'paid_openrouter_models',           labelKey: 'field.paid_openrouter_models',  type: 'csv-list', placeholder: 'anthropic/claude-3.7-sonnet', section: 'advanced'},
  {key: 'openai_api_key',                   labelKey: 'field.openai_api_key',          type: 'password',                                          section: 'advanced'},
  {key: 'anthropic_api_key',                labelKey: 'field.anthropic_api_key',       type: 'password',                                          section: 'advanced'},
  {key: 'gemini_api_key',                   labelKey: 'field.gemini_api_key',          type: 'password',                                          section: 'advanced'},
  {key: 'perplexity_api_key',               labelKey: 'field.perplexity_api_key',      type: 'password',                                          section: 'advanced'},
  {key: 'raw_power_weight',                 labelKey: 'field.raw_power_weight',        type: 'number',  step: '0.05', min: 0, max: 1,            section: 'advanced'},
  {key: 'ecofrugality_weight',              labelKey: 'field.ecofrugality_weight',     type: 'number',  step: '0.05', min: 0, max: 1,            section: 'advanced'},
];

// Live state of the engines panel between open/save. Each entry has
// {label, base_url, api_key, models}. Kept at module scope so the
// "+ Add engine" handler can mutate it without re-fetching settings.
let engineState = [];

// Cached list of available local model ids — populated on each
// settings-sheet open via /v1/models. Filtered to `ollama/*` so the
// "Local Judge" dropdown only shows things the local stack can serve.
let localModelOptions = [];

async function fetchLocalModelOptions() {
  try {
    const data = await apiGet('/v1/models');
    return (data.data || [])
      .filter(m => typeof m.id === 'string' && m.id.startsWith('ollama/'))
      .map(m => m.id.slice('ollama/'.length))
      .sort();
  } catch {
    return [];
  }
}

// Preset URLs we'll suggest as placeholders for new engine rows.
const ENGINE_PRESETS = [
  {label: 'mistral',  base_url: 'https://api.mistral.ai/v1'},
  {label: 'together', base_url: 'https://api.together.xyz/v1'},
  {label: 'groq',     base_url: 'https://api.groq.com/openai/v1'},
  {label: 'fireworks', base_url: 'https://api.fireworks.ai/inference/v1'},
];

function csvParse(raw) {
  return String(raw || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
}
function csvFormat(list) {
  return Array.isArray(list) ? list.join(', ') : '';
}

function renderEngineList() {
  const host = document.getElementById('engineList');
  if (!host) return;
  if (!engineState.length) {
    const presets = ENGINE_PRESETS.map(p => `<code>${p.label}</code> (${p.base_url})`).join(', ');
    host.innerHTML = `<p class="text-[11px] text-gray-500 dark:text-gray-400 italic">${t('engines.empty', {presets})}</p>`;
    return;
  }
  host.innerHTML = '';
  engineState.forEach((engine, idx) => {
    const card = document.createElement('div');
    card.className = 'p-2 rounded-[10px] bg-white/60 dark:bg-white/[0.04] space-y-1.5 border border-gray-200 dark:border-white/[0.08]';
    card.innerHTML = `
      <div class="flex items-center justify-between gap-2">
        <input type="text" data-i="${idx}" data-field="label" value="${escapeHtml(engine.label)}" placeholder="${escapeHtml(t('engines.label.placeholder'))}"
          class="flex-1 px-2 py-1 text-[12px] font-medium border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20">
        <button type="button" data-rm="${idx}" class="w-8 h-8 rounded-full flex items-center justify-center text-gray-400 hover:text-red-500 hover:bg-black/5 dark:hover:bg-white/10" title="${escapeHtml(t('engines.remove'))}">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>
        </button>
      </div>
      <input type="text" data-i="${idx}" data-field="base_url" value="${escapeHtml(engine.base_url)}" placeholder="${escapeHtml(t('engines.baseUrl.placeholder'))}"
        class="w-full px-2 py-1 text-[12px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20">
      <input type="password" data-i="${idx}" data-field="api_key" value="${escapeHtml(engine.api_key)}" placeholder="${escapeHtml(t('engines.apiKey.placeholder'))}"
        class="w-full px-2 py-1 text-[12px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20">
      <input type="text" data-i="${idx}" data-field="models" value="${escapeHtml(csvFormat(engine.models))}" placeholder="${escapeHtml(t('engines.models.placeholder'))}"
        class="w-full px-2 py-1 text-[12px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20">
    `;
    host.appendChild(card);
  });
  // Wire inputs to mutate engineState live so the form-submit handler
  // can just read the array (no per-card DOM scraping).
  host.querySelectorAll('input[data-i]').forEach(input => {
    input.addEventListener('input', e => {
      const i = parseInt(e.target.dataset.i, 10);
      const field = e.target.dataset.field;
      if (Number.isNaN(i) || !engineState[i]) return;
      if (field === 'models') engineState[i].models = csvParse(e.target.value);
      else engineState[i][field] = e.target.value;
    });
  });
  host.querySelectorAll('button[data-rm]').forEach(btn => {
    btn.addEventListener('click', () => {
      const i = parseInt(btn.dataset.rm, 10);
      if (Number.isNaN(i)) return;
      engineState.splice(i, 1);
      renderEngineList();
    });
  });
}

async function openSettings() {
  let current = {};
  try { current = await apiGet('/api/settings'); }
  catch (err) { showToast(t('toast.settingsLoadFailed', {message: err.message})); return; }
  // Refresh the local model catalogue used by the "Local Judge"
  // dropdown. Cheap (one /v1/models GET) and always up-to-date.
  localModelOptions = await fetchLocalModelOptions();

  const form = $('settingsForm');
  form.innerHTML = '';

  // Personal-mode panel — shows current corpus state and exposes an
  // "Ingest now" button that walks data/personal/inbox/ via the
  // multimodal extractors. Lives at the top of the sheet so it's
  // discoverable without scrolling past every credential.
  const personalPanel = document.createElement('div');
  personalPanel.className = 'mb-4 p-3 rounded-[10px] bg-black/5 dark:bg-white/[0.05] space-y-2';
  personalPanel.id = 'personalPanel';
  personalPanel.innerHTML = `
    <div class="flex items-center justify-between">
      <div>
        <div class="text-[13px] font-semibold">${escapeHtml(t('personal.title'))}</div>
        <div class="text-[11px] text-gray-500 dark:text-gray-400" id="personalSummary">${escapeHtml(t('personal.subtitle.loading'))}</div>
      </div>
      <div class="flex items-center gap-1.5">
        <button type="button" id="personalVizBtn"
          class="min-h-[32px] px-3 py-1 text-[12px] font-medium border border-gray-300 dark:border-white/[0.12] hover:bg-black/5 dark:hover:bg-white/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-sysblue/60">
          ${escapeHtml(t('personal.visualize'))}
        </button>
        <button type="button" id="personalIngestBtn"
          class="min-h-[32px] px-3 py-1 text-[12px] font-medium bg-sysblue text-white hover:bg-sysblueHover focus:outline-none focus-visible:ring-2 focus-visible:ring-sysblue/60">
          ${escapeHtml(t('personal.ingest'))}
        </button>
      </div>
    </div>
    <p class="text-[11px] text-gray-500 dark:text-gray-400 leading-relaxed">${t('personal.help')}</p>
  `;
  form.appendChild(personalPanel);
  refreshPersonalSummary();
  $('personalVizBtn').addEventListener('click', openPersonalViz);
  $('personalIngestBtn').addEventListener('click', async () => {
    const btn = $('personalIngestBtn');
    btn.disabled = true;
    btn.textContent = t('personal.ingesting');
    try {
      const res = await triggerPersonalIngest(false);
      const added = (res.results || []).filter(r => r.wiki_path && !r.error).length;
      showToast(t('personal.ingested', {added, wiki: res.status?.wiki ?? '?'}));
      refreshPersonalSummary();
    } catch (err) {
      showToast(t('personal.ingest.failed', {message: err.message}));
    } finally {
      btn.disabled = false;
      btn.textContent = t('personal.ingest');
    }
  });

  // Engines panel — dynamic OpenAI-compatible engine list. Every
  // engine has a label, a base URL, an API key, and a comma-separated
  // model list. The "+ Add engine" button appends an empty card; the
  // trash icon on each card removes it. Render is reactive to
  // ``engineState``; the form-submit handler reads ``engineState``
  // directly rather than the input nodes.
  engineState = Array.isArray(current.custom_engines)
    ? current.custom_engines.map(e => ({
        label: e.label || '',
        base_url: e.base_url || '',
        api_key: e.api_key || '',
        models: Array.isArray(e.models) ? [...e.models] : [],
      }))
    : [];

  const enginesPanel = document.createElement('div');
  enginesPanel.className = 'mb-4 p-3 rounded-[10px] bg-black/5 dark:bg-white/[0.05] space-y-2';
  enginesPanel.id = 'enginesPanel';
  enginesPanel.innerHTML = `
    <div class="flex items-center justify-between">
      <div>
        <div class="text-[13px] font-semibold">${escapeHtml(t('engines.title'))}</div>
        <div class="text-[11px] text-gray-500 dark:text-gray-400">${escapeHtml(t('engines.subtitle'))}</div>
      </div>
      <button type="button" id="addEngineBtn"
        class="min-h-[32px] px-3 py-1 text-[12px] font-medium bg-sysblue text-white hover:bg-sysblueHover focus:outline-none focus-visible:ring-2 focus-visible:ring-sysblue/60">
        ${escapeHtml(t('engines.addEngine'))}
      </button>
    </div>
    <div id="engineList" class="space-y-2"></div>
  `;
  form.appendChild(enginesPanel);
  renderEngineList();
  $('addEngineBtn').addEventListener('click', () => {
    engineState.push({label: '', base_url: '', api_key: '', models: []});
    renderEngineList();
  });

  // Basics section — visible inline.
  const basicsHeader = document.createElement('div');
  basicsHeader.className = 'pt-1 text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 font-semibold';
  basicsHeader.textContent = t('settings.section.basics');
  form.appendChild(basicsHeader);
  for (const f of SETTINGS_FIELDS.filter(x => x.section === 'basics')) {
    form.appendChild(renderSettingsField(f, current));
  }

  // Advanced section — collapsed by default; a non-tech user can ignore.
  const advWrap = document.createElement('details');
  advWrap.className = 'pt-2 mt-2 border-t border-gray-100 dark:border-white/[0.06]';
  const advSummary = document.createElement('summary');
  advSummary.className = 'cursor-pointer select-none text-[11px] uppercase tracking-wider text-gray-500 dark:text-gray-400 font-semibold flex items-center gap-1';
  advSummary.innerHTML = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"><path d="M9 6l6 6-6 6"/></svg><span>${escapeHtml(t('settings.section.advanced'))}</span>`;
  advWrap.appendChild(advSummary);
  const advHint = document.createElement('p');
  advHint.className = 'text-[11px] text-gray-500 dark:text-gray-400 mt-1 mb-2 leading-relaxed';
  advHint.textContent = t('settings.section.advanced.hint');
  advWrap.appendChild(advHint);
  for (const f of SETTINGS_FIELDS.filter(x => x.section === 'advanced')) {
    advWrap.appendChild(renderSettingsField(f, current));
  }
  form.appendChild(advWrap);
  $('settingsSheet').classList.remove('hidden');
}

// Settings field renderer extracted so Basics and Advanced sections
// share the same look. Returns a fully-built <label> element.
//
// Checkboxes are wired so the ``checked`` attribute mirrors the
// persisted value — that's the contract for "if something is set,
// show it set". A missing value renders unchecked (the default).
function renderSettingsField(f, current) {
  const labelText = t(f.labelKey);
  const wrap = document.createElement('label');
  wrap.className = 'flex flex-col gap-1.5';
  const val = current[f.key];
  let inputHtml;
  if (f.type === 'checkbox') {
    wrap.className = 'flex items-center justify-between py-1 gap-2';
    const checked = val === true ? 'checked' : '';
    inputHtml = `<input type="checkbox" name="${f.key}" ${checked} class="w-5 h-5 accent-sysblue shrink-0">`;
    wrap.innerHTML = `<span class="text-[13px] text-gray-700 dark:text-gray-300">${escapeHtml(labelText)}</span>${inputHtml}`;
    return wrap;
  }
  const labelHtml = `<span class="text-[12px] font-medium text-gray-600 dark:text-gray-400">${escapeHtml(labelText)}</span>`;
  if (f.type === 'select-local-model') {
    // Dropdown of available local Ollama models. The current persisted
    // value stays as the first option even if Ollama doesn't list it
    // (e.g. the user wiped the model but the setting still references
    // it), so saving doesn't silently blank the field.
    const currentVal = (val ?? '').toString();
    const options = new Set(localModelOptions);
    if (currentVal) options.add(currentVal);
    const orderedOptions = Array.from(options).sort();
    const blankOption = f.allow_blank
      ? `<option value="" ${currentVal === '' ? 'selected' : ''}>—</option>`
      : '';
    let bodyHtml;
    if (orderedOptions.length === 0) {
      // No models available — render a disabled select + a hint so the
      // user sees what to do.
      bodyHtml = `<select name="${f.key}" disabled
        class="px-3 py-2 text-[13px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20 transition-all opacity-60"><option>—</option></select>
        <span class="text-[11px] text-gray-500 dark:text-gray-400 italic">${escapeHtml(t('field.localModels.empty'))}</span>`;
    } else {
      const optionsHtml = orderedOptions.map(id => {
        const selected = id === currentVal ? 'selected' : '';
        return `<option value="${escapeHtml(id)}" ${selected}>${escapeHtml(id)}</option>`;
      }).join('');
      bodyHtml = `<select name="${f.key}"
        class="px-3 py-2 text-[13px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20 transition-all">${blankOption}${optionsHtml}</select>`;
    }
    wrap.innerHTML = labelHtml + bodyHtml;
    return wrap;
  }
  if (f.type === 'csv-list') {
    const placeholder = f.placeholder || '';
    inputHtml = `<input type="text" name="${f.key}" value="${escapeHtml(csvFormat(val))}" placeholder="${escapeHtml(placeholder)}"
      class="px-3 py-2 text-[13px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20 transition-all">`;
  } else {
    const extra = f.step ? `step="${f.step}"` : '';
    const range = f.min !== undefined ? `min="${f.min}" max="${f.max}"` : '';
    inputHtml = `<input type="${f.type}" name="${f.key}" value="${escapeHtml(String(val ?? ''))}" placeholder="${escapeHtml(f.placeholder || '')}" ${extra} ${range}
      class="px-3 py-2 text-[13px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20 transition-all">`;
  }
  wrap.innerHTML = labelHtml + inputHtml;
  return wrap;
}

function closeSettings() {
  $('settingsSheet').classList.add('hidden');
}

async function saveSettings(ev) {
  ev.preventDefault();
  const fd = new FormData(ev.target);
  // Start from the current settings on the server so we don't blank any unmodelled fields.
  let current = {};
  try { current = await apiGet('/api/settings'); } catch {}
  const next = {...current};
  for (const f of SETTINGS_FIELDS) {
    if (f.type === 'checkbox') {
      next[f.key] = fd.get(f.key) !== null;
    } else if (f.type === 'number') {
      const raw = fd.get(f.key);
      next[f.key] = raw === '' || raw === null ? 0 : Number(raw);
    } else if (f.type === 'csv-list') {
      next[f.key] = csvParse(fd.get(f.key));
    } else {
      next[f.key] = fd.get(f.key) ?? '';
    }
  }
  // Engines panel — read straight from the in-memory ``engineState``.
  // Skip rows that the user left completely blank (saves the user
  // from accidentally persisting a half-edited card).
  next.custom_engines = engineState
    .map(e => ({
      label: (e.label || '').trim(),
      base_url: (e.base_url || '').trim(),
      api_key: e.api_key || '',
      models: (e.models || []).map(m => (m || '').trim()).filter(Boolean),
    }))
    .filter(e => e.label || e.base_url || e.api_key || e.models.length);
  try {
    await apiPost('/api/settings', next);
    state.allowVlms = !!next.enable_vlms;
    state.prefDefaults.independence = !!next.independence_local_only;
    state.prefDefaults.pseudonymize = !!next.enable_pseudonymization;
    state.prefs.independence = state.prefDefaults.independence;
    state.prefs.pseudonymize = state.prefDefaults.pseudonymize;
    renderStatusPill();
    renderPrefsDot();
    showToast(t('toast.saved'));
    closeSettings();
  } catch (err) {
    showToast(t('toast.saveFailed', {message: err.message}));
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  $('composer').addEventListener('submit', (e) => { e.preventDefault(); send(); });
  $('attachBtn').addEventListener('click', () => $('fileInput').click());
  $('fileInput').addEventListener('change', (e) => { onFilesPicked(e.target.files); e.target.value = ''; });
  $('attachUrlBtn')?.addEventListener('click', () => {
    // Native prompt is simple and works in both desktop + mobile; the
    // GUI design language doesn't require a custom modal for one input.
    const raw = window.prompt(t('composer.urlPrompt'));
    const url = (raw || '').trim();
    if (!url) return;
    if (!/^https?:\/\//i.test(url)) {
      showToast(t('toast.skippedAttachment', {name: url}));
      return;
    }
    state.urlAttachments.push(url);
    renderAttachments();
  });
  $('newChatBtn').addEventListener('click', newChat);
  $('settingsBtn').addEventListener('click', openSettings);
  $('closeSettingsBtn').addEventListener('click', closeSettings);
  $('cancelSettingsBtn').addEventListener('click', closeSettings);
  $('settingsOverlay').addEventListener('click', closeSettings);
  $('settingsForm').addEventListener('submit', saveSettings);

  $('prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); }
  });
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'n') { e.preventDefault(); newChat(); }
    if (e.key === 'Escape') closeSettings();
  });

  // Apply initial translations to every annotated element, then wire
  // the EN/FR toggle. Re-rendering on toggle re-translates the messages
  // list too, so a chat in flight switches language mid-conversation.
  window.RoiteletI18n.applyStaticTranslations();
  $('langToggle')?.addEventListener('click', () => {
    const next = window.RoiteletI18n.currentLang() === 'fr' ? 'en' : 'fr';
    window.RoiteletI18n.setLang(next);
    // Re-render the dynamic surfaces too: status pill, messages list,
    // and the conversation list (titles are server-provided so they
    // don't move, but the "Untitled" fallback flips language).
    renderStatusPill();
    renderMessages();
    renderConversationList();
  });

  wirePrefsPopover();
  refreshConversations();
  refreshPreferences();
  $('prompt').focus();
});
