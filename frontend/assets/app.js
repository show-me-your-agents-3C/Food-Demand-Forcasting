'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const families = {
  BEVERAGES: ['Beverages', 'B'], 'BREAD/BAKERY': ['Bread & Bakery', 'Br'], DAIRY: ['Dairy', 'D'],
  DELI: ['Deli', 'De'], EGGS: ['Eggs', 'E'], 'FROZEN FOODS': ['Frozen Foods', 'F'],
  'GROCERY I': ['Grocery I', 'G1'], 'GROCERY II': ['Grocery II', 'G2'], MEATS: ['Meats', 'M'],
  POULTRY: ['Poultry', 'Po'], 'PREPARED FOODS': ['Prepared Foods', 'Pf'], PRODUCE: ['Produce', 'Pr'],
  SEAFOOD: ['Seafood', 'S'],
};
const toolNames = {
  get_forecast: 'Read demand forecast', get_forecast_overview: 'Read forecast overview',
  get_replenishment: 'Compute replenishment', get_priority_replenishments: 'Rank replenishment priorities',
  explain_forecast: 'Explain forecast drivers', what_if_promotion: 'Re-forecast promotion scenario',
  get_model_reliability: 'Read backtest reliability',
};
const state = { ready: false, store: '1', family: '', view: 'all', plans: [], forecasts: [],
  metrics: {}, health: {}, focus: null, history: [], sessionId: null, chatController: null, bootController: null };
const number = (value, digits = 0) => Number(value).toLocaleString('en-US', { maximumFractionDigits: digits });
const familyName = (family) => families[family]?.[0] || family;
const key = (row) => `${row.store_nbr}:${row.family}`;
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const dateOnly = (value) => String(value).slice(0, 10);
const currentLabel = () => state.focus ? `Store ${state.focus.store_nbr} · ${familyName(state.focus.family)}` : 'Select a store and category';

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`The service could not complete the request (${response.status}). Please try again.`);
  return response.json();
}

function toast(message) {
  $('#toast').textContent = message;
  $('#toast').hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { $('#toast').hidden = true; }, 3200);
}

async function boot() {
  state.bootController?.abort();
  state.bootController = new AbortController();
  const controller = state.bootController;
  const timeout = setTimeout(() => controller.abort(), 15000);
  state.ready = false;
  $('#error-banner').hidden = true;
  $('#retry-btn').disabled = true;
  $('#sidebar-status-text').textContent = 'Connecting to data';
  $('#export-btn').disabled = true;
  $('#store-select').disabled = true;
  $('#family-select').disabled = true;
  setChatEnabled(false);
  try {
    const [health, scope, metrics, plans, forecasts] = await Promise.all([
      '/health', '/api/scope', '/api/metrics', '/api/replenishment', '/api/forecast',
    ].map((url) => request(url, { signal: controller.signal })));
    if (!health.data_available || !Array.isArray(plans) || !Array.isArray(forecasts) || !plans.length || !forecasts.length) {
      throw new Error('Forecast or replenishment data is not ready. Check the backend artifacts and retry.');
    }
    Object.assign(state, { health, metrics, plans, forecasts, ready: true });
    $('#store-select').innerHTML = '<option value="">All stores</option>' + scope.stores.map((store) => `<option value="${Number(store)}">Store ${Number(store)}</option>`).join('');
    $('#family-select').innerHTML = '<option value="">All categories</option>' + scope.families.map((family) => `<option value="${esc(family)}">${esc(familyName(family))}</option>`).join('');
    if (state.store && !scope.stores.includes(Number(state.store))) state.store = String(scope.stores[0]);
    if (!scope.families.includes(state.family)) state.family = '';
    $('#store-select').value = state.store;
    $('#family-select').value = state.family;
    $('#store-select').disabled = false;
    $('#family-select').disabled = false;
    $('#sidebar-status-text').textContent = 'Data connected';
    $('#sidebar-status').classList.remove('failed');
    const dates = forecasts.map((r) => dateOnly(r.date)).sort();
    $('#date-range').textContent = `${dates[0]} — ${dates.at(-1).slice(5)}`;
    render();
  } catch (error) {
    if (state.bootController !== controller) return;
    state.ready = false;
    $('#error-message').textContent = error.name === 'AbortError' ? 'Connection timed out. Make sure the service is running, then reconnect.' : error.message;
    $('#error-banner').hidden = false;
    $('#sidebar-status').classList.add('failed');
    $('#sidebar-status-text').textContent = 'Connection failed';
    $('#mode-badge').textContent = 'Offline';
    $('#assistant-footnote').textContent = 'The assistant will be available once data is back.';
    if (!state.plans.length) $('#chart').innerHTML = '<div class="loading-placeholder">Waiting for data</div>';
    setChatEnabled(false);
  } finally {
    clearTimeout(timeout);
    $('#retry-btn').disabled = false;
  }
}

function scopedPlans() {
  return state.plans.filter((r) => (!state.store || r.store_nbr === Number(state.store)) && (!state.family || r.family === state.family));
}
function visiblePlans() {
  return scopedPlans().filter((r) => state.view === 'all' || (state.view === 'waste' ? r.waste_risk === 'high' : r.action === 'order_today'));
}

function render() {
  const scope = scopedPlans();
  const rows = visiblePlans();
  const previousKey = state.focus ? key(state.focus) : null;
  if (!rows.some((r) => key(r) === previousKey)) {
    state.focus = rows.find((r) => r.family === 'DAIRY') || rows[0] || null;
  }
  $('#kpi-orders').innerHTML = `${scope.filter((r) => r.action === 'order_today').length}<small>items</small>`;
  $('#kpi-waste').innerHTML = `${scope.filter((r) => r.waste_risk === 'high').length}<small>items</small>`;
  $('#kpi-scope').innerHTML = `${scope.length}<small>pairs</small>`;
  $('#kpi-scope-note').textContent = `${new Set(scope.map((r) => r.store_nbr)).size} stores · ${new Set(scope.map((r) => r.family)).size} food categories`;
  $('#kpi-wape').innerHTML = state.metrics.wape == null ? '—' : `${number(state.metrics.wape * 100, 2)}<small>%</small>`;
  $('#nav-count').textContent = scope.filter((r) => r.action === 'order_today').length;
  $('#table-count').textContent = rows.length;
  $('#table-footer-count').textContent = `Showing ${rows.length} / ${scope.length} store-category pairs`;
  $('#export-btn').disabled = !state.ready || rows.length === 0;
  renderTable(rows);
  renderChart();
  if (previousKey !== (state.focus ? key(state.focus) : null) || !$('#chat-log').childElementCount) resetChat();
  setChatEnabled(state.ready && !!state.focus && !state.chatController);
}

function renderTable(rows) {
  $('#plan-body').innerHTML = rows.length ? rows.map((r) => `<tr class="plan-row ${state.focus && key(state.focus) === key(r) ? 'is-selected' : ''}" data-key="${esc(key(r))}">
    <td><div class="family-cell"><span class="family-initial" data-family="${esc(r.family)}">${esc(families[r.family]?.[1] || '?')}</span><div><strong>${esc(familyName(r.family))}</strong><small>Store ${r.store_nbr} · ${esc(r.family)}</small></div></div></td>
    <td class="numeric">${number(r.forecast_7d, 1)}</td><td class="numeric">${number(r.current_stock_simulated, 1)}</td>
    <td class="numeric order-value">${number(r.recommended_order_qty)}</td>
    <td><span class="badge ${r.action === 'order_today' ? 'order' : 'monitor'}">${r.action === 'order_today' ? 'Order today' : 'Monitor'}</span></td>
    <td><span class="badge ${r.waste_risk === 'high' ? 'high' : 'low'}">${r.waste_risk === 'high' ? 'High' : 'Low'}</span></td>
    <td><button class="row-select" aria-label="View store ${r.store_nbr} ${esc(familyName(r.family))}" aria-pressed="${!!state.focus && key(state.focus) === key(r)}">${icon('arrow')}</button></td></tr>`).join('')
    : '<tr><td colspan="7" class="empty-state">No recommendations match this view. Switch to “All” or pick another category.</td></tr>';
}

function renderChart() {
  if (!state.focus) {
    $('#chart-total').textContent = '—';
    $('#chart-subtitle').textContent = 'No store-category pair matches the current filters';
    $('#chart').innerHTML = '<div class="loading-placeholder">Adjust the filters to see a forecast</div>';
    $('#chart-point-detail').textContent = 'Sales in source-data units';
    return;
  }
  const rows = state.forecasts.filter((r) => r.store_nbr === state.focus.store_nbr && r.family === state.focus.family)
    .sort((a, b) => String(a.date).localeCompare(String(b.date)));
  $('#chart-subtitle').textContent = `${currentLabel()} · source-data units`;
  $('#chart-total').textContent = number(rows.reduce((sum, r) => sum + Number(r.forecast_sales), 0), 1);
  $('#chart-point-detail').textContent = 'Hover or focus a point for daily detail';
  if (!rows.length) {
    $('#chart').innerHTML = '<div class="loading-placeholder">No forecast for this pair</div>';
    return;
  }
  const width = 640, height = 224, left = 47, right = 19, top = 17, bottom = 179;
  const maximum = Math.max(1, ...rows.map((r) => Number(r.upper_bound) || Number(r.forecast_sales)));
  const magnitude = 10 ** Math.floor(Math.log10(maximum));
  const ceiling = Math.ceil(maximum / magnitude * 2) / 2 * magnitude;
  const x = (i) => left + i * (width - left - right) / Math.max(1, rows.length - 1);
  const y = (v) => bottom - Number(v) / ceiling * (bottom - top);
  const coords = (field) => rows.map((r, i) => `${x(i).toFixed(2)},${y(r[field]).toFixed(2)}`);
  const line = `M${coords('forecast_sales').join(' L')}`;
  const band = `M${coords('upper_bound').join(' L')} L${coords('lower_bound').reverse().join(' L')} Z`;
  const area = `${line} L${x(rows.length - 1)},${bottom} L${x(0)},${bottom} Z`;
  let grid = '';
  for (let i = 0; i <= 4; i++) {
    const value = ceiling * i / 4;
    grid += `<line x1="${left}" y1="${y(value)}" x2="${width - right}" y2="${y(value)}" stroke="#eaf0e1" stroke-dasharray="3 5"/><text x="${left - 12}" y="${y(value) + 3}" text-anchor="end">${esc(number(value))}</text>`;
  }
  const points = rows.map((r, i) => `<g class="chart-target" tabindex="0" role="img" aria-label="${dateOnly(r.date)} forecast ${number(r.forecast_sales, 1)}" data-index="${i}">
    <rect x="${x(i) - 16}" y="${top}" width="32" height="${bottom - top}" fill="transparent"/>
    <circle class="point-dot" cx="${x(i)}" cy="${y(r.forecast_sales)}" r="3.5"/>
    <text x="${x(i)}" y="${bottom + 24}" text-anchor="middle">${dateOnly(r.date).slice(5).replace('-', '/')}</text>
    <title>${dateOnly(r.date)} · forecast ${number(r.forecast_sales, 1)}</title></g>`).join('');
  $('#chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="7-day demand forecast for ${esc(currentLabel())}">
    <defs><linearGradient id="forecast-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#c5d8ad" stop-opacity=".35"/><stop offset="100%" stop-color="#c5d8ad" stop-opacity=".02"/></linearGradient></defs>
    ${grid}<path d="${band}" fill="#e9f0df" opacity=".66"/><path d="${area}" fill="url(#forecast-fill)"/>
    <path d="${line}" fill="none" stroke="#4b7958" stroke-width="2.3" stroke-linejoin="round"/>${points}</svg>`;
  $$('.chart-target').forEach((point) => {
    const reveal = () => {
      const r = rows[Number(point.dataset.index)];
      $('#chart-point-detail').textContent = `${dateOnly(r.date)} · forecast (p50) ${number(r.forecast_sales, 1)} · 80% range (p10–p90) ${number(r.lower_bound, 1)}–${number(r.upper_bound, 1)}`;
    };
    point.addEventListener('mouseenter', reveal);
    point.addEventListener('focus', reveal);
    point.addEventListener('click', reveal);
  });
}

function setChatEnabled(enabled) {
  $('#chat-input').disabled = !enabled;
  $('#send-btn').disabled = !enabled;
  $$('[data-question]').forEach((button) => { button.disabled = !enabled; });
}

function setMode(mode) {
  $('#mode-badge').textContent = mode === 'llm' ? 'LLM' : mode === 'configured' ? 'Ready' : 'Template fallback';
  $('#mode-badge').classList.toggle('llm', mode === 'llm');
  $('#assistant-footnote').textContent = mode === 'llm' ? 'Answered by the LLM · check the tool evidence'
    : mode === 'configured' ? 'Gateway configured · the actual mode shows after each answer'
      : 'Template fallback · answers are built from tool results only';
}

function resetChat() {
  state.chatController?.abort();
  state.chatController = null;
  state.history = [];
  state.sessionId = crypto.randomUUID ? crypto.randomUUID() : String(Date.now() + Math.random());
  $('#chat-input').value = '';
  $('#chat-context').textContent = currentLabel();
  setMode(state.health.llm_configured ? 'configured' : 'fallback');
  $('#chat-log').innerHTML = `<div class="welcome"><div class="welcome-emblem">${icon('spark')}</div><h3>See demand clearly,<br>replenish with confidence.</h3><p>${state.focus ? `Viewing ${esc(currentLabel())}.<br>Ask about the forecast, why it changed, how much to order, or a promotion what-if.` : 'Select a store and category with data first.'}</p><small>Answers use the v3 forecast and simulated inventory.<br>Open “View evidence” to check the actual tool results.</small></div>`;
  setChatEnabled(state.ready && !!state.focus);
}

function appendMessage(role, content) {
  const message = document.createElement('div');
  message.className = `message ${role}`;
  if (role === 'assistant') {
    const label = document.createElement('div');
    label.className = 'message-label';
    label.innerHTML = `${icon('spark')} FRESHFLOW`;
    message.append(label);
  }
  const text = document.createElement('div');
  text.className = 'message-content';
  text.textContent = content;
  message.append(text);
  $('#chat-log').append(message);
  $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  return message;
}

function appendTrace(message, traces) {
  if (!traces?.length) return;
  const detail = document.createElement('details');
  detail.className = 'trace';
  const summary = document.createElement('summary');
  summary.textContent = `View evidence · ${traces.length} tool call(s)`;
  detail.append(summary);
  const inner = document.createElement('div');
  inner.className = 'trace-inner';
  for (const entry of traces) {
    const item = document.createElement('div');
    item.className = 'trace-tool';
    const title = document.createElement('strong');
    title.textContent = toolNames[entry.tool] || entry.tool;
    const params = document.createElement('p');
    const code = document.createElement('code');
    code.textContent = `${entry.tool}(${JSON.stringify(entry.args)})`;
    params.append(code);
    const source = document.createElement('p');
    source.textContent = `Source: ${entry.result?.source || 'available data'}`;
    const raw = document.createElement('details');
    const toggle = document.createElement('summary');
    toggle.textContent = 'Show raw result';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(entry.result, null, 2);
    raw.append(toggle, pre);
    item.append(title, params, source, raw);
    inner.append(item);
  }
  detail.append(inner);
  message.append(detail);
}

async function sendQuestion(question) {
  const text = question.trim();
  if (!text || !state.ready || !state.focus || state.chatController) return;
  const controller = new AbortController();
  state.chatController = controller;
  const focusKey = key(state.focus);
  let timedOut = false;
  const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 55000);
  $('#chat-log .welcome')?.remove();
  appendMessage('user', text);
  $('#chat-input').value = '';
  setChatEnabled(false);
  const pending = appendMessage('pending', 'Looking up the data');
  try {
    const result = await request('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ message: text, session_id: state.sessionId, store_nbr: state.focus.store_nbr, family: state.focus.family }),
    });
    if (state.chatController !== controller || !state.focus || key(state.focus) !== focusKey) return;
    pending.remove();
    const message = appendMessage('assistant', result.reply);
    appendTrace(message, result.tool_trace);
    setMode(result.mode);
    state.history.push({ role: 'user', content: text }, { role: 'assistant', content: result.reply });
    $('#chat-log').scrollTop = $('#chat-log').scrollHeight;
  } catch (error) {
    if (state.chatController !== controller) return;
    pending.remove();
    const reason = timedOut ? 'The answer took too long. Please try again.' : error.message;
    appendMessage('error', `${reason} Your question was not saved; you can send it again.`);
    $('#chat-input').value = text;
  } finally {
    clearTimeout(timeout);
    if (state.chatController === controller) {
      state.chatController = null;
      setChatEnabled(state.ready && !!state.focus);
    }
  }
}

$('#store-select').addEventListener('change', (event) => { state.store = event.target.value; render(); });
$('#family-select').addEventListener('change', (event) => { state.family = event.target.value; render(); });
$$('[data-view]').forEach((button) => button.addEventListener('click', () => {
  state.view = button.dataset.view;
  $$('[data-view]').forEach((item) => {
    const active = item === button;
    item.classList.toggle('selected', active);
    item.setAttribute('aria-pressed', String(active));
  });
  render();
}));
$('#plan-body').addEventListener('click', (event) => {
  const row = event.target.closest('[data-key]');
  if (!row || !state.ready) return;
  const selected = state.plans.find((r) => key(r) === row.dataset.key);
  if (!selected || state.focus && key(selected) === key(state.focus)) return;
  state.focus = selected;
  renderTable(visiblePlans());
  renderChart();
  resetChat();
});
$('#chat-form').addEventListener('submit', (event) => { event.preventDefault(); sendQuestion($('#chat-input').value); });
$('#chat-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendQuestion(event.target.value); }
});
$$('[data-question]').forEach((button) => button.addEventListener('click', () => sendQuestion(button.dataset.question)));
$('#clear-chat').addEventListener('click', () => { resetChat(); toast('Chat cleared; current view kept'); });
$('#retry-btn').addEventListener('click', boot);

$('#export-btn').addEventListener('click', () => {
  const rows = visiblePlans();
  if (!state.ready || !rows.length) return;
  const header = ['Store', 'Category', '7-day forecast p50 (source units)', 'Simulated stock', 'Planned order qty', 'Action', 'Waste risk', 'Note'];
  const csvRows = [header, ...rows.map((r) => [r.store_nbr, familyName(r.family), r.forecast_7d, r.current_stock_simulated,
    r.recommended_order_qty, r.action === 'order_today' ? 'Order today' : 'Monitor', r.waste_risk === 'high' ? 'High' : 'Low',
    'Historical replay 2017-08-16 to 2017-08-22; inventory and operating parameters are simulated; not a purchase order'])];
  const csv = '\uFEFF' + csvRows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `FreshFlow_replenishment_${state.store ? `store${state.store}` : 'all_stores'}_2017-08-16.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  toast(`Exported ${rows.length} recommendation(s)`);
});

$$('[data-nav]').forEach((button) => button.addEventListener('click', () => {
  $$('[data-nav]').forEach((item) => item.classList.toggle('active', item === button));
  document.getElementById(button.dataset.nav).scrollIntoView({ behavior: 'smooth', block: 'start' });
}));
$$('[data-open-info]').forEach((button) => button.addEventListener('click', () => $('#info-dialog').showModal()));
$('#close-info').addEventListener('click', () => $('#info-dialog').close());
$('#info-done').addEventListener('click', () => $('#info-dialog').close());
$('#info-dialog').addEventListener('click', (event) => {
  if (event.target === $('#info-dialog')) {
    const r = event.target.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) event.target.close();
  }
});
boot();
