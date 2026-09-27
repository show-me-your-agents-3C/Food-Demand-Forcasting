'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const families = {
  BEVERAGES: ['饮料', '饮'], 'BREAD/BAKERY': ['面包烘焙', '麦'], DAIRY: ['乳制品', '乳'],
  'FROZEN FOODS': ['冷冻食品', '冻'], MEATS: ['肉类', '肉'], POULTRY: ['禽类', '禽'],
  PRODUCE: ['果蔬', '蔬'], SEAFOOD: ['海鲜', '鲜'],
};
const toolNames = {
  get_demand_forecast: '读取需求预测', get_inventory_status: '读取模拟库存',
  get_replenishment_plan: '读取补货建议', get_forecast_metrics: '读取回测指标',
  get_dashboard_summary: '读取计划概览', list_scope: '核对可用范围', simulate_promotion: '计算需求变化情景',
};
const state = { ready: false, store: '1', family: '', view: 'all', plans: [], forecasts: [],
  metrics: {}, health: {}, focus: null, history: [], chatController: null, bootController: null };
const number = (value, digits = 0) => Number(value).toLocaleString('zh-CN', { maximumFractionDigits: digits });
const familyName = (family) => families[family]?.[0] || family;
const key = (row) => `${row.store_nbr}:${row.family}`;
const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
const dateOnly = (value) => String(value).slice(0, 10);
const currentLabel = () => state.focus ? `门店 ${state.focus.store_nbr} · ${familyName(state.focus.family)}` : '请选择一个门店与品类';

async function request(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`服务暂时无法完成请求（${response.status}）。请稍后重试。`);
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
  $('#sidebar-status-text').textContent = '正在连接数据';
  $('#export-btn').disabled = true;
  $('#store-select').disabled = true;
  $('#family-select').disabled = true;
  setChatEnabled(false);
  try {
    const [health, scope, metrics, plans, forecasts] = await Promise.all([
      '/health', '/api/scope', '/api/metrics', '/api/replenishment', '/api/forecast',
    ].map((url) => request(url, { signal: controller.signal })));
    if (!health.data_available || !Array.isArray(plans) || !Array.isArray(forecasts) || !plans.length || !forecasts.length) {
      throw new Error('预测或补货数据尚未就绪，请检查后端结果文件后重试。');
    }
    Object.assign(state, { health, metrics, plans, forecasts, ready: true });
    $('#store-select').innerHTML = '<option value="">所有门店</option>' + scope.stores.map((store) => `<option value="${Number(store)}">门店 ${Number(store)}</option>`).join('');
    $('#family-select').innerHTML = '<option value="">全部品类</option>' + scope.families.map((family) => `<option value="${esc(family)}">${esc(familyName(family))}</option>`).join('');
    if (state.store && !scope.stores.includes(Number(state.store))) state.store = String(scope.stores[0]);
    if (!scope.families.includes(state.family)) state.family = '';
    $('#store-select').value = state.store;
    $('#family-select').value = state.family;
    $('#store-select').disabled = false;
    $('#family-select').disabled = false;
    $('#sidebar-status-text').textContent = '业务数据已连接';
    $('#sidebar-status').classList.remove('failed');
    const dates = forecasts.map((r) => dateOnly(r.date)).sort();
    $('#date-range').textContent = `${dates[0]} — ${dates.at(-1).slice(5)}`;
    render();
  } catch (error) {
    if (state.bootController !== controller) return;
    state.ready = false;
    $('#error-message').textContent = error.name === 'AbortError' ? '连接超时，请确认本地服务仍在运行，再点击重新连接。' : error.message;
    $('#error-banner').hidden = false;
    $('#sidebar-status').classList.add('failed');
    $('#sidebar-status-text').textContent = '连接失败';
    $('#mode-badge').textContent = '未连接';
    $('#assistant-footnote').textContent = '数据恢复后即可继续使用助手。';
    if (!state.plans.length) $('#chart').innerHTML = '<div class="loading-placeholder">等待连接业务数据</div>';
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
  $('#kpi-orders').innerHTML = `${scope.filter((r) => r.action === 'order_today').length}<small>项</small>`;
  $('#kpi-waste').innerHTML = `${scope.filter((r) => r.waste_risk === 'high').length}<small>项</small>`;
  $('#kpi-scope').innerHTML = `${scope.length}<small>组</small>`;
  $('#kpi-scope-note').textContent = `${new Set(scope.map((r) => r.store_nbr)).size} 家门店 · ${new Set(scope.map((r) => r.family)).size} 个食品品类`;
  $('#kpi-wape').innerHTML = state.metrics.wape == null ? '—' : `${number(state.metrics.wape * 100, 2)}<small>%</small>`;
  $('#nav-count').textContent = scope.filter((r) => r.action === 'order_today').length;
  $('#table-count').textContent = rows.length;
  $('#table-footer-count').textContent = `显示 ${rows.length} / ${scope.length} 个门店品类组合`;
  $('#export-btn').disabled = !state.ready || rows.length === 0;
  renderTable(rows);
  renderChart();
  if (previousKey !== (state.focus ? key(state.focus) : null) || !$('#chat-log').childElementCount) resetChat();
  setChatEnabled(state.ready && !!state.focus && !state.chatController);
}

function renderTable(rows) {
  $('#plan-body').innerHTML = rows.length ? rows.map((r) => `<tr class="plan-row ${state.focus && key(state.focus) === key(r) ? 'is-selected' : ''}" data-key="${esc(key(r))}">
    <td><div class="family-cell"><span class="family-initial" data-family="${esc(r.family)}">${esc(families[r.family]?.[1] || '食')}</span><div><strong>${esc(familyName(r.family))}</strong><small>门店 ${r.store_nbr} · ${esc(r.family)}</small></div></div></td>
    <td class="numeric">${number(r.forecast_7d, 1)}</td><td class="numeric">${number(r.current_stock_simulated, 1)}</td>
    <td class="numeric order-value">${number(r.recommended_order_qty)}</td>
    <td><span class="badge ${r.action === 'order_today' ? 'order' : 'monitor'}">${r.action === 'order_today' ? '需要补货' : '持续观察'}</span></td>
    <td><span class="badge ${r.waste_risk === 'high' ? 'high' : 'low'}">${r.waste_risk === 'high' ? '高风险' : '低风险'}</span></td>
    <td><button class="row-select" aria-label="查看门店${r.store_nbr}${esc(familyName(r.family))}" aria-pressed="${!!state.focus && key(state.focus) === key(r)}">${icon('arrow')}</button></td></tr>`).join('')
    : '<tr><td colspan="7" class="empty-state">当前范围没有符合条件的建议。可以切换到“全部”或选择其他品类。</td></tr>';
}

function renderChart() {
  if (!state.focus) {
    $('#chart-total').textContent = '—';
    $('#chart-subtitle').textContent = '当前筛选条件下没有可展示的组合';
    $('#chart').innerHTML = '<div class="loading-placeholder">调整筛选条件以查看预测</div>';
    $('#chart-point-detail').textContent = '销量使用原始数据单位';
    return;
  }
  const rows = state.forecasts.filter((r) => r.store_nbr === state.focus.store_nbr && r.family === state.focus.family)
    .sort((a, b) => String(a.date).localeCompare(String(b.date)));
  $('#chart-subtitle').textContent = `${currentLabel()} · 原始销量单位`;
  $('#chart-total').textContent = number(rows.reduce((sum, r) => sum + Number(r.forecast_sales), 0), 1);
  $('#chart-point-detail').textContent = '悬停或聚焦数据点，查看每日明细';
  if (!rows.length) {
    $('#chart').innerHTML = '<div class="loading-placeholder">这个组合暂无预测结果</div>';
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
  const points = rows.map((r, i) => `<g class="chart-target" tabindex="0" role="img" aria-label="${dateOnly(r.date)} 预测 ${number(r.forecast_sales, 1)}" data-index="${i}">
    <rect x="${x(i) - 16}" y="${top}" width="32" height="${bottom - top}" fill="transparent"/>
    <circle class="point-dot" cx="${x(i)}" cy="${y(r.forecast_sales)}" r="3.5"/>
    <text x="${x(i)}" y="${bottom + 24}" text-anchor="middle">${dateOnly(r.date).slice(5).replace('-', '/')}</text>
    <title>${dateOnly(r.date)} · 预测 ${number(r.forecast_sales, 1)}</title></g>`).join('');
  $('#chart').innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(currentLabel())}的7天需求预测折线图">
    <defs><linearGradient id="forecast-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="#c5d8ad" stop-opacity=".35"/><stop offset="100%" stop-color="#c5d8ad" stop-opacity=".02"/></linearGradient></defs>
    ${grid}<path d="${band}" fill="#e9f0df" opacity=".66"/><path d="${area}" fill="url(#forecast-fill)"/>
    <path d="${line}" fill="none" stroke="#4b7958" stroke-width="2.3" stroke-linejoin="round"/>${points}</svg>`;
  $$('.chart-target').forEach((point) => {
    const reveal = () => {
      const r = rows[Number(point.dataset.index)];
      $('#chart-point-detail').textContent = `${dateOnly(r.date)} · 预测 ${number(r.forecast_sales, 1)} · 参考区间 ${number(r.lower_bound, 1)}–${number(r.upper_bound, 1)}`;
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
  $('#mode-badge').textContent = mode === 'llm' ? '大模型' : mode === 'configured' ? '待问答' : '规则备用';
  $('#mode-badge').classList.toggle('llm', mode === 'llm');
  $('#assistant-footnote').textContent = mode === 'llm' ? '已通过大模型回答 · 请结合工具结果审阅建议'
    : mode === 'configured' ? '网关已配置 · 回答后显示实际运行模式'
      : '规则备用模式 · 支持查询和需求变化情景';
}

function resetChat() {
  state.chatController?.abort();
  state.chatController = null;
  state.history = [];
  $('#chat-input').value = '';
  $('#chat-context').textContent = currentLabel();
  setMode(state.health.llm_configured ? 'configured' : 'fallback');
  $('#chat-log').innerHTML = `<div class="welcome"><div class="welcome-emblem">${icon('spark')}</div><h3>一起看清需求，<br>把补货安排得更从容。</h3><p>${state.focus ? `正在查看 ${esc(currentLabel())}。<br>我可以解释补货依据，或比较需求变化后的计划。` : '请先选择一个有数据的门店和品类。'}</p><small>回答基于已有预测和模拟库存。<br>展开“查看依据”，可以核对实际工具结果。</small></div>`;
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
  summary.textContent = `查看依据 · ${traces.length} 次工具调用`;
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
    source.textContent = `来源：${entry.result?.source || '当前可用数据范围'}`;
    const raw = document.createElement('details');
    const toggle = document.createElement('summary');
    toggle.textContent = '展开原始结果';
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
  const pending = appendMessage('pending', '正在查询当前范围的数据');
  try {
    const context = { role: 'user', content: `当前页面选中：门店 ${state.focus.store_nbr}，品类 ${state.focus.family}。` };
    const result = await request('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: controller.signal,
      body: JSON.stringify({ message: text, history: [context, ...state.history.slice(-12)] }),
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
    const reason = timedOut ? '这次回答等待时间较长，请稍后重试。' : error.message;
    appendMessage('error', `${reason} 你的问题没有被保存为成功对话，可以重新发送。`);
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
$('#clear-chat').addEventListener('click', () => { resetChat(); toast('已清空对话，保留当前查看范围'); });
$('#retry-btn').addEventListener('click', boot);

$('#export-btn').addEventListener('click', () => {
  const rows = visiblePlans();
  if (!state.ready || !rows.length) return;
  const header = ['门店', '品类', '7天预测（原始销量单位）', '模拟库存', '计划采购量', '建议', '报废风险', '数据说明'];
  const csvRows = [header, ...rows.map((r) => [r.store_nbr, familyName(r.family), r.forecast_7d, r.current_stock_simulated,
    r.recommended_order_qty, r.action === 'order_today' ? '需要补货' : '持续观察', r.waste_risk === 'high' ? '高' : '低',
    '历史回放2017-08-16至2017-08-22；库存等运营参数为模拟；此文件不是采购订单'])];
  const csv = '\uFEFF' + csvRows.map((row) => row.map((value) => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `FreshFlow_补货建议_${state.store ? `门店${state.store}` : '所有门店'}_2017-08-16.csv`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
  toast(`已导出当前筛选的 ${rows.length} 条建议`);
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
