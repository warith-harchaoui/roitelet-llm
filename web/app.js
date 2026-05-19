/* Roitelet web client.
 *
 * Vanilla JS, no build step. Talks to the FastAPI backend over its native
 * endpoints (no SSE — the OpenAI-compat streaming chunks an already-complete
 * answer, so honest loading state beats fake typing).
 */

const $ = (id) => document.getElementById(id);
const escapeHtml = (s) => s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

const state = {
  conversationId: null,
  messages: [],          // {role, content, metadata?}
  conversations: [],
  busy: false,
};

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

// ─── UI rendering ────────────────────────────────────────────────────────────

function renderMessages() {
  const inner = $('messagesInner');
  inner.innerHTML = '';
  if (state.messages.length === 0) {
    inner.innerHTML = `
      <div id="welcome" class="text-center py-24">
        <div class="inline-flex items-center justify-center w-14 h-14 mb-5 bg-gradient-to-br from-sysblue to-violet-500 text-white shadow-lg">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2L3 22h18z"/></svg>
        </div>
        <h2 class="text-[22px] font-semibold tracking-tight">Ask any question.</h2>
        <p class="text-[14px] text-gray-500 dark:text-gray-400 mt-2 max-w-md mx-auto leading-relaxed">Three frontier models answer in parallel. A local model picks the best.</p>
      </div>`;
    return;
  }
  for (const m of state.messages) inner.appendChild(messageNode(m));
  scrollToBottom();
}

function messageNode(m) {
  const wrap = document.createElement('div');
  wrap.className = m.role === 'user' ? 'flex justify-end' : 'flex justify-start';

  const bubble = document.createElement('div');
  if (m.role === 'user') {
    bubble.className = 'bg-sysblue text-white px-4 py-2.5 max-w-[78%] text-[15px] leading-relaxed shadow-sm whitespace-pre-wrap';
    bubble.textContent = m.content;
  } else {
    bubble.className = 'bg-gray-100 dark:bg-[#2c2c2e] px-4 py-3 max-w-[85%] text-[15px] leading-relaxed shadow-sm prose-msg';
    bubble.innerHTML = marked.parse(m.content || '', {breaks: true, gfm: true});
    if (m.metadata) bubble.appendChild(metadataNode(m.metadata));
  }
  wrap.appendChild(bubble);
  return wrap;
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

  const body = document.createElement('div');
  body.className = 'mt-2 space-y-2';
  body.innerHTML = `
    <div><span class="text-gray-400">Capabilities</span> &nbsp;${escapeHtml(caps)}</div>
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
    a.textContent = c.title || 'Untitled';
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
  $('statusDot').className = `w-1.5 h-1.5 rounded-full ${busy ? 'bg-sysblue animate-pulse' : 'bg-emerald-500'}`;
  $('statusText').textContent = busy ? (label || 'Thinking…') : 'Ready';
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
    showToast('Could not load conversations: ' + err.message);
  }
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
    showToast('Could not load conversation: ' + err.message);
  } finally {
    setBusy(false);
  }
}

function newChat() {
  state.conversationId = null;
  state.messages = [];
  $('conversationTitle').textContent = 'New chat';
  renderMessages();
  renderConversationList();
  $('prompt').focus();
}

async function send() {
  const prompt = $('prompt').value.trim();
  if (!prompt || state.busy) return;

  state.messages.push({role: 'user', content: prompt});
  $('prompt').value = '';
  renderMessages();
  showThinking();
  setBusy(true);

  try {
    const payload = {
      prompt,
      conversation_id: state.conversationId,
      preferences: {raw_power: 0.7, frugality: 0.3, independence: false, allow_vlms: false},
      top_k: 3,
    };
    const res = await apiPost('/api/chat', payload);
    state.conversationId = res.conversation_id;
    state.messages.push({
      role: 'assistant',
      content: res.synthesis?.content || '(no answer)',
      metadata: {
        router: res.router,
        responses: res.responses,
        synthesis: res.synthesis,
      },
    });
    hideThinking();
    renderMessages();
    await refreshConversations();
  } catch (err) {
    hideThinking();
    showToast('Pipeline error: ' + err.message);
    // Keep the user's message visible; don't auto-rewind.
  } finally {
    setBusy(false);
    $('prompt').focus();
  }
}

// ─── Settings sheet ──────────────────────────────────────────────────────────

const SETTINGS_FIELDS = [
  {key: 'ollama_base_url',           label: 'Ollama base URL',        type: 'text',     placeholder: 'http://localhost:11434'},
  {key: 'local_synthesis_model',     label: 'Local synthesis model',  type: 'text',     placeholder: 'qwen2.5:14b-instruct'},
  {key: 'openrouter_api_key',        label: 'OpenRouter API key',     type: 'password'},
  {key: 'openai_api_key',            label: 'OpenAI API key',         type: 'password'},
  {key: 'anthropic_api_key',         label: 'Anthropic API key',      type: 'password'},
  {key: 'gemini_api_key',            label: 'Gemini API key',         type: 'password'},
  {key: 'perplexity_api_key',        label: 'Perplexity API key',     type: 'password'},
  {key: 'raw_power_weight',          label: 'Raw power weight',       type: 'number', step: '0.05', min: 0, max: 1},
  {key: 'frugality_weight',          label: 'Frugality weight',       type: 'number', step: '0.05', min: 0, max: 1},
  {key: 'independence_local_only',   label: 'Local models only',      type: 'checkbox'},
  {key: 'enable_vlms',               label: 'Allow vision-language',  type: 'checkbox'},
];

async function openSettings() {
  let current = {};
  try { current = await apiGet('/api/settings'); }
  catch (err) { showToast('Could not load settings: ' + err.message); return; }

  const form = $('settingsForm');
  form.innerHTML = '';
  for (const f of SETTINGS_FIELDS) {
    const wrap = document.createElement('label');
    wrap.className = 'flex flex-col gap-1.5';
    const labelText = `<span class="text-[12px] font-medium text-gray-600 dark:text-gray-400">${f.label}</span>`;
    const val = current[f.key];
    let inputHtml;
    if (f.type === 'checkbox') {
      wrap.className = 'flex items-center justify-between py-1';
      inputHtml = `<input type="checkbox" name="${f.key}" ${val ? 'checked' : ''} class="w-5 h-5 accent-sysblue">`;
      wrap.innerHTML = `<span class="text-[13px] text-gray-700 dark:text-gray-300">${f.label}</span>${inputHtml}`;
    } else {
      const extra = f.step ? `step="${f.step}"` : '';
      const range = f.min !== undefined ? `min="${f.min}" max="${f.max}"` : '';
      inputHtml = `<input type="${f.type}" name="${f.key}" value="${escapeHtml(String(val ?? ''))}" placeholder="${f.placeholder || ''}" ${extra} ${range}
        class="px-3 py-2 text-[13px] border border-gray-300 dark:border-white/[0.12] bg-white dark:bg-[#2c2c2e] focus:outline-none focus:border-sysblue focus:ring-2 focus:ring-sysblue/20 transition-all">`;
      wrap.innerHTML = labelText + inputHtml;
    }
    form.appendChild(wrap);
  }
  $('settingsSheet').classList.remove('hidden');
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
    } else {
      next[f.key] = fd.get(f.key) ?? '';
    }
  }
  try {
    await apiPost('/api/settings', next);
    showToast('Saved');
    closeSettings();
  } catch (err) {
    showToast('Save failed: ' + err.message);
  }
}

// ─── Init ────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  $('composer').addEventListener('submit', (e) => { e.preventDefault(); send(); });
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

  refreshConversations();
  $('prompt').focus();
});
