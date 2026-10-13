'use strict';

const $ = id => document.getElementById(id);
const valid = v => typeof v === 'number' && Number.isFinite(v);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, digits = 0) => valid(v) ? v.toLocaleString('ja-JP', {maximumFractionDigits:digits, minimumFractionDigits:digits}) : '—';
const pct = v => valid(v) ? `${fmt(v, 1)}%` : '—';
const unit = (v, label, digits = 1) => valid(v) ? `${fmt(v, digits)} ${label}` : '—';
const bytes = v => {
  if (!valid(v)) return '—';
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (Math.abs(v) >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${fmt(v, i ? 1 : 0)} ${units[i]}`;
};
const rate = v => valid(v) ? `${bytes(v)}/s` : '—';
const timeLabel = v => valid(v) ? new Date(v * 1000).toLocaleTimeString('ja-JP', {hour12:false}) : '—';
const duration = v => {
  if (!valid(v)) return '—';
  if (v >= 86400) return `${Math.floor(v / 86400)}日 ${Math.floor(v % 86400 / 3600)}時間`;
  return v >= 3600 ? `${Math.floor(v / 3600)}時間 ${Math.floor(v % 3600 / 60)}分` : `${Math.floor(v / 60)}分 ${Math.floor(v % 60)}秒`;
};
const states = {idle:'待機中', generating:'生成中', reading:'入力処理中', loading:'ロード中', unloaded:'未ロード', online:'接続済み', offline:'未接続', connecting:'接続確認中'};
const colors = {cpu:'#83bafd', gpu:'#76e2b4', ram:'#bda4fa', vram:'#edbc79', tok_s:'#76e2b4', prefill_tok_s:'#83bafd'};
let data = null, selectedTarget = null, selectedModel = '', view = 'overview', chartKey = 'tok_s', windowSeconds = 300;
let paused = false, pending = false, firstSnapshot = true, toastTimer = null, sidebarSignature = '', autoSelect = true;
try { selectedTarget = localStorage.getItem('observatory-target'); autoSelect = !selectedTarget; } catch (_) {}

function set(id, value) { $(id).textContent = value ?? '—'; }
function currentTarget() { return data?.targets.find(t => t.id === selectedTarget) || data?.targets[0]; }
function currentModel(target) {
  return target?.models.find(m => m.api_url === selectedModel) || target?.models.find(m => ['generating','reading'].includes(m.state)) || target?.models[0];
}
function chip(state) { return `<span class="chip ${esc(state)}">${esc(states[state] || state)}</span>`; }
function notify(message) {
  set('toast', message); $('toast').hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('toast').hidden = true; }, 3500);
}
function selectTarget(id) {
  autoSelect = false;
  if (selectedTarget !== id) selectedModel = '';
  selectedTarget = id;
  try { localStorage.setItem('observatory-target', id); } catch (_) {}
  sidebarSignature = ''; render();
}
function showView(value) {
  view = value;
  for (const key of ['overview','models','connections']) $(key + '-view').hidden = key !== value;
  document.querySelectorAll('[data-view]').forEach(b => b.classList.toggle('active', b.dataset.view === value));
  const titles = {overview:['RESOURCE OVERVIEW','リソース概要','ローカルとColab。推論の状態を、一つの画面で。'], models:['MODEL INVENTORY','モデル','稼働モデルと、ワークスペースに登録されたモデル。'], connections:['CONNECTIONS','接続管理','ローカルAPIとColab runtimeの接続状況。']};
  set('heading-eyebrow', titles[value][0]); set('page-title', titles[value][1]); set('breadcrumb', titles[value][1]); set('page-description', titles[value][2]);
  render();
}

function svgChart(points, key, color, small = false) {
  const series = points.map(p => p[key]);
  if (!series.some(valid)) return small ? '<div class="sparkline-empty">計測待ち</div>' : '<div class="chart-empty">データの計測を待っています</div>';
  const width = small ? 250 : 620, height = small ? 44 : 156, left = small ? 0 : 33, top = small ? 4 : 12, bottom = small ? 0 : 10;
  const plotWidth = width - left - 2, plotHeight = height - top - bottom;
  const values = series.filter(valid);
  const max = small ? 100 : Math.max(1, ...values) * 1.2;
  const coordinates = series.map((v, i) => valid(v) ? [left + i * plotWidth / Math.max(1, series.length - 1), top + (1 - Math.max(0, Math.min(max, v)) / max) * plotHeight] : null);
  const paths = []; let line = [];
  for (const point of [...coordinates, null]) {
    if (point) line.push(point);
    else if (line.length) { paths.push(line); line = []; }
  }
  const path = segment => segment.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
  const gradientId = `fill-${key}-${small ? 'small' : 'main'}`;
  const grid = small ? '' : [0, .5, 1].map(n => {
    const y = top + n * plotHeight;
    return `<line x1="${left}" y1="${y}" x2="${width}" y2="${y}" stroke="#2a3338" stroke-dasharray="3 4"/><text x="0" y="${y + 3}" fill="#6f7f89" font-family="monospace" font-size="9">${fmt(max * (1 - n))}</text>`;
  }).join('');
  const fills = paths.map(p => `<path d="${path(p)} L${p.at(-1)[0]},${height} L${p[0][0]},${height}Z" fill="url(#${gradientId})"/>`).join('');
  const lines = paths.map(p => `<path d="${path(p)}" fill="none" stroke="${color}" stroke-width="${small ? 1.5 : 2}" vector-effect="non-scaling-stroke"/>`).join('');
  return `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" aria-hidden="true"><defs><linearGradient id="${gradientId}" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="${color}" stop-opacity=".16"/><stop offset="100%" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>${grid}${fills}${lines}</svg>`;
}

function renderSidebar() {
  const signature = JSON.stringify(data.targets.map(t => [t.id, t.name, t.kind, t.available, t.state, t.models[0]?.name, selectedTarget === t.id]));
  if (sidebarSignature === signature) return;
  sidebarSignature = signature;
  const focused = document.activeElement?.dataset.target;
  $('target-list').innerHTML = data.targets.map(t => `<button class="target-button ${t.id === selectedTarget ? 'selected' : ''}" data-target="${esc(t.id)}" aria-pressed="${t.id === selectedTarget}"><div class="target-name"><span class="status-dot ${t.available ? '' : esc(t.state)}"></span><span>${esc(t.name)}</span><small class="target-kind">${t.kind === 'colab' ? 'CLOUD' : 'LOCAL'}</small></div><div class="target-meta">${esc(t.models[0]?.name || (t.available ? 'モデル未接続' : states[t.state]))}</div></button>`).join('');
  if (focused) document.querySelector(`[data-target="${CSS.escape(focused)}"]`)?.focus();
  set('target-count', String(data.targets.length).padStart(2, '0'));
}

function renderOverview(target) {
  const hw = target.hardware, cpu = hw?.cpu || {}, ram = hw?.ram || {}, gpu = hw?.gpus[0] || {};
  const model = currentModel(target), stats = model?.stats || {};
  const history = (target.history || []).filter(p => p.time >= (target.sampled_at || data.time) - windowSeconds);
  document.body.classList.toggle('stale', !target.available);
  set('host-name', target.name); set('host-icon', target.kind === 'colab' ? '⌁' : '▧');
  set('host-detail', hw ? `${hw.scope} · ${hw.hostname} · ${hw.cpu.threads || '—'} logical CPUs` : (target.kind === 'colab' ? '既存のColab SSH接続を自動検出' : 'ローカルマシンを監視'));
  const hostState = $('host-state'); hostState.className = 'chip ' + target.state; hostState.textContent = target.available ? 'CONNECTED' : target.state.toUpperCase();
  set('host-uptime', hw ? `HOST UPTIME ${duration(hw.uptime_s)}` : '');
  $('target-notice').hidden = target.available;
  set('target-notice', (target.error || '接続状況を確認しています。') + (hw && target.sampled_at ? ` 最終取得：${timeLabel(target.sampled_at)}。表示中の値はその時点のものです。` : ''));
  for (const [key, value] of [['cpu', cpu.util_percent], ['gpu', gpu.util_percent], ['ram', ram.percent], ['vram', gpu.memory_percent]]) {
    $(key + '-value').innerHTML = `${fmt(value, 1)}<span>%</span>`;
    $(key + '-spark').innerHTML = svgChart(history, key, colors[key], true);
  }
  set('cpu-name', cpu.name || 'CPU情報なし'); $('cpu-name').title = cpu.name || '';
  $('cpu-detail').innerHTML = `<span>${fmt(cpu.cores)} cores / ${fmt(cpu.threads)} threads</span><strong>${unit(cpu.frequency_mhz, 'MHz', 0)}</strong>`;
  set('gpu-name', gpu.name || 'GPU未検出'); $('gpu-name').title = gpu.name || '';
  $('gpu-detail').innerHTML = `<span>${valid(gpu.temperature_c) ? unit(gpu.temperature_c, '°C', 0) : '温度 —'}</span><strong>${unit(gpu.power_w, 'W')}</strong>`;
  set('ram-size', `${bytes(ram.used_bytes)} / ${bytes(ram.total_bytes)}`);
  $('ram-detail').innerHTML = `<span>利用可能</span><strong>${bytes(ram.available_bytes)}</strong>`;
  set('vram-size', `${bytes(gpu.used_bytes)} / ${bytes(gpu.total_bytes)}`);
  $('vram-detail').innerHTML = `<span>空きVRAM</span><strong>${valid(gpu.total_bytes) && valid(gpu.used_bytes) ? bytes(gpu.total_bytes - gpu.used_bytes) : '—'}</strong>`;

  const rolling = stats.speed_kind === 'rolling';
  const speed = rolling ? stats.tok_s : stats.average_tok_s;
  set('speed-label', rolling ? '生成速度 · 直近の実測' : '生成速度 · 累計平均');
  $('decode-speed').innerHTML = `${fmt(speed, 1)}<span>tok/s</span>`;
  set('speed-detail', !model ? 'モデルへの接続を待っています' : stats.speed_kind === 'unavailable' ? '速度取得には推論サーバーのメトリクスが必要です' : `${states[model.state] || model.state}${rolling ? ' · 直近ウィンドウ' : ' · 完了リクエストの平均'}${valid(stats.last_tok_s) ? ' · 前回 ' + fmt(stats.last_tok_s, 1) + ' tok/s' : ''}`);
  set('prefill-speed', fmt(rolling ? stats.prefill_tok_s : stats.average_prefill_tok_s, 1));
  const chartPoints = history.map(p => ({...p, tok_s:p.models?.[model?.api_url]?.tok_s ?? null, prefill_tok_s:p.models?.[model?.api_url]?.prefill_tok_s ?? null}));
  $('speed-chart').innerHTML = svgChart(chartPoints, chartKey, colors[chartKey]);
  $('speed-chart').onpointermove = event => {
    if (!chartPoints.length || !chartPoints.some(p => valid(p[chartKey]))) return;
    const rect = $('speed-chart').getBoundingClientRect();
    const fraction = Math.max(0, Math.min(1, ((event.clientX - rect.left) / rect.width * 620 - 33) / 585));
    const p = chartPoints[Math.round(fraction * (chartPoints.length - 1))];
    let tooltip = $('speed-chart').querySelector('.chart-tooltip');
    if (!tooltip) { tooltip = document.createElement('div'); tooltip.className = 'chart-tooltip'; $('speed-chart').append(tooltip); }
    tooltip.textContent = `${timeLabel(p.time)} · ${fmt(p[chartKey], 1)} tok/s`;
    tooltip.style.left = `${Math.min(rect.width - 185, Math.max(0, event.clientX - rect.left - 60))}px`;
  };
  $('speed-chart').onpointerleave = () => $('speed-chart').querySelector('.chart-tooltip')?.remove();
  set('chart-label', chartKey === 'tok_s' ? (rolling ? '生成 tok/s' : '累計平均 tok/s') : '入力処理 tok/s');
  set('chart-start', timeLabel(chartPoints[0]?.time)); set('chart-middle', timeLabel(chartPoints[Math.floor(chartPoints.length / 2)]?.time)); set('chart-end', timeLabel(chartPoints.at(-1)?.time));
  set('request-count', fmt(stats.requests_total)); set('output-total', fmt(stats.output_tokens_total)); set('cache-hit', pct(stats.cache_hit_percent));

  const modelState = $('model-state'); modelState.className = 'chip ' + (model?.state || 'offline'); modelState.textContent = states[model?.state] || '未接続';
  const select = $('model-select'); select.hidden = !target.models || target.models.length < 2;
  if (!select.hidden) {
    const options = target.models.map(m => `<option value="${esc(m.api_url)}">${esc(m.name)}</option>`).join('');
    if (select.dataset.options !== options) { select.innerHTML = options; select.dataset.options = options; }
    select.value = model.api_url;
  }
  set('model-name', model?.name || '稼働モデルなし');
  const tags = [model?.quantization, model?.kv ? `KV ${model.kv}` : null, model?.mtp ? `MTP ${model.mtp}` : null].filter(Boolean);
  $('model-tags').innerHTML = tags.map(tag => `<span>${esc(tag)}</span>`).join('');
  set('context-limit', valid(model?.context) ? `${fmt(model.context)} tokens` : '—');
  const used = valid(stats.prompt_tokens) && valid(stats.generated_tokens) ? stats.prompt_tokens + stats.generated_tokens : null;
  $('context-progress').style.width = valid(used) && model.context > 0 ? `${Math.min(100, used / model.context * 100)}%` : '0%';
  set('context-caption', valid(used) ? `${fmt(used)} 使用 / ${fmt(model.context)} 上限` : '使用量はリクエスト処理中に表示します');
  set('model-engine', model ? `${model.engine}${model.version ? ' · ' + model.version : ''}` : '—');
  set('last-speed', unit(stats.last_tok_s, 'tok/s')); set('last-prefill', unit(stats.last_prefill_tok_s, 'tok/s'));
  set('generated-tokens', fmt(stats.generated_tokens)); set('queued-count', fmt(stats.queued)); set('model-uptime', duration(model?.uptime_s));
  set('model-endpoint', model?.api_url || 'API 未接続');
  $('model-paths').innerHTML = model?.paths.length ? model.paths.map(p => `<div><span>${esc(p.role)}</span><code>${esc(p.path)}</code></div>`).join('') : '<span class="muted">このAPIは配置情報を公開していません。</span>';

  $('disk-list').innerHTML = hw?.disks.length ? hw.disks.map(d => `<div class="disk-entry"><div class="disk-heading"><span>${esc(d.label)}</span><strong>${pct(d.percent)}</strong></div><div class="disk-path">${esc(d.path)} · ${esc(d.filesystem)}</div><div class="progress-track"><i style="width:${valid(d.percent) ? d.percent : 0}%;background:${d.percent > 85 ? 'var(--amber)' : 'var(--green)'}"></i></div><div class="disk-caption"><span>${bytes(d.used_bytes)} / ${bytes(d.total_bytes)}</span><span>空き ${bytes(d.free_bytes)}</span></div>${d.virtual ? '<p class="disk-footnote">DriveFSの報告値。runtime SSDの容量とは別です。</p>' : ''}</div>`).join('') : '<div class="empty-state">ディスク情報なし</div>';
  const io = hw?.io_rates || {};
  set('disk-read', rate(io.disk_read_bps)); set('disk-write', rate(io.disk_write_bps)); set('network-rx', rate(io.network_rx_bps)); set('network-tx', rate(io.network_tx_bps));
  const sensors = [['CPU 温度', unit(cpu.temperature_c, '°C', 0), cpu.temperature_c > 80], ['GPU 温度', unit(gpu.temperature_c, '°C', 0), gpu.temperature_c > 80], ['GPU 消費電力', unit(gpu.power_w, 'W'), false], ['GPU 電力上限', unit(gpu.power_limit_w, 'W', 0), false], ['GPU クロック', unit(gpu.clock_mhz, 'MHz', 0), false], ['GPU ファン', pct(gpu.fan_percent), false]];
  $('sensor-list').innerHTML = sensors.map(([name, value, hot]) => `<div class="sensor-row"><span>${name}</span><strong class="${hot ? 'hot' : ''}">${esc(value)}</strong></div>`).join('');
  const cores = cpu.per_core || [];
  set('core-caption', `CPUコアごとの使用率${cpu.threads ? ' · ' + cpu.threads + ' threads' : ''}`);
  $('core-grid').innerHTML = cores.length ? cores.map(c => `<div class="core-cell ${c.util_percent > 85 ? 'hot' : ''}" style="background:rgba(118,226,180,${valid(c.util_percent) ? .12 + c.util_percent / 150 : .05})" title="${esc(c.name)} · ${pct(c.util_percent)}" aria-label="${esc(c.name)} ${pct(c.util_percent)}"></div>`).join('') : '<span class="core-empty">次のサンプルから使用率を表示します</span>';
  const scope = [hw?.scope, valid(cpu.quota_cores) ? `CPU割当 ${fmt(cpu.quota_cores, 1)} cores` : null, valid(ram.container_limit_bytes) ? `コンテナ上限 ${bytes(ram.container_limit_bytes)}` : null].filter(Boolean);
  set('host-scope', scope.join(' · '));
  $('gpu-list').hidden = !(hw?.gpus.length > 1);
  if (!$('gpu-list').hidden) $('gpu-list').innerHTML = hw.gpus.map(g => `<article class="panel"><div class="eyebrow">GPU ${fmt(g.index)}</div><h3>${esc(g.name)}</h3><div class="gpu-details"><span>負荷 ${pct(g.util_percent)}</span><span>VRAM ${bytes(g.used_bytes)} / ${bytes(g.total_bytes)}</span><span>${unit(g.temperature_c, '°C', 0)}</span><span>${unit(g.power_w, 'W')}</span></div></article>`).join('');
}

function renderModels() {
  const models = data.targets.flatMap(t => t.models.map(m => ({...m, host:t.name, available:t.available})));
  $('running-models').innerHTML = models.length ? models.map(m => `<tr><td>${esc(m.name)}<small>${esc(m.api_url)}</small></td><td>${esc(m.host)}</td><td>${chip(m.available ? m.state : 'offline')}</td><td class="mono">${fmt(m.stats.speed_kind === 'rolling' ? m.stats.tok_s : m.stats.average_tok_s, 1)}${m.stats.speed_kind === 'average' ? '<small>累計平均</small>' : ''}</td><td class="mono">${fmt(m.context)}</td></tr>`).join('') : '<tr><td colspan="5"><div class="empty-state">稼働中の推論APIが見つかりません。モデルを起動するか、接続管理からAPIを追加してください。</div></td></tr>';
  $('model-catalog').innerHTML = data.catalog.map(m => `<article class="panel catalog-card"><div class="eyebrow">${esc(m.id)}</div><h3>${esc(m.name)}</h3><p>${esc(m.description)}</p><div class="catalog-meta"><span>既定 ${fmt(m.default_context_length)}</span><span>上限 ${fmt(m.max_context_length)}</span></div></article>`).join('') || '<div class="empty-state">model.tomlの登録がありません。</div>';
}

function renderConnections() {
  $('connection-cards').innerHTML = data.targets.map(t => `<article class="panel connection-card"><div class="card-top"><span>${t.kind === 'colab' ? 'COLAB RUNTIME' : 'LOCAL HOST'}</span>${chip(t.state)}</div><h3>${esc(t.name)}</h3><p>${esc(t.available ? t.hardware?.hostname || '接続済み' : t.error || '接続状況を確認しています')}</p><code>${esc(t.models[0]?.api_url || t.api_urls?.[0] || (t.kind === 'colab' ? 'SSH / ' + t.session : '推論APIを自動検出'))}</code><div class="dialog-actions"><button class="secondary-button" data-select-target="${esc(t.id)}">リソースを見る</button>${!t.builtin ? `<button class="secondary-button delete-target" data-delete="${esc(t.id)}">削除</button>` : ''}</div></article>`).join('');
}

function render() {
  if (!data) return;
  if (!data.targets.some(t => t.id === selectedTarget)) selectedTarget = data.targets[0]?.id;
  renderSidebar();
  const target = currentTarget();
  if (target) renderOverview(target);
  if (view === 'models') renderModels();
  if (view === 'connections') renderConnections();
  set('update-time', paused ? '表示を一時停止' : timeLabel(data.time));
  set('footer-status', `${data.targets.filter(t => t.available).length} / ${data.targets.length} ホスト接続 · ${data.interval}秒間隔${paused ? ' · 表示停止中' : ''}`);
}

async function refresh(force = false) {
  if (pending || (paused && !force)) return;
  pending = true;
  try {
    const response = await fetch('/api/state', {cache:'no-store', signal:AbortSignal.timeout(6000)});
    if (!response.ok) throw new Error('Dashboard unavailable');
    data = await response.json();
    if (firstSnapshot) {
      const preferred = data.targets.find(t => t.available && t.kind === 'colab' && t.models.length);
      if (preferred && autoSelect) selectedTarget = preferred.id;
      firstSnapshot = data.targets.some(t => t.state === 'connecting');
    }
    $('global-error').hidden = true;
    render();
  } catch (_) {
    set('global-error', 'ダッシュボードサーバーに接続できません。表示中の値は最終取得時点のものです。自動で再接続します。');
    $('global-error').hidden = false;
  } finally { pending = false; }
}

function openDialog() { $('form-error').textContent = ''; $('connection-dialog').showModal(); }
$('add-connection').addEventListener('click', openDialog);
$('add-connection-main').addEventListener('click', openDialog);
for (const id of ['close-dialog', 'cancel-dialog']) $(id).addEventListener('click', () => $('connection-dialog').close());
$('connection-dialog').addEventListener('click', event => { if (event.target === $('connection-dialog')) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
$('connection-form').addEventListener('submit', async event => {
  event.preventDefault(); $('save-connection').disabled = true; set('form-error', '');
  try {
    const fields = new FormData(event.target);
    const response = await fetch('/api/targets', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:fields.get('name'), api_url:fields.get('api_url')}), signal:AbortSignal.timeout(6000)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error);
    $('connection-dialog').close(); event.target.reset(); selectedTarget = result.id; selectedModel = ''; sidebarSignature = '';
    await refresh(true); notify('接続を追加しました');
  } catch (error) { set('form-error', error.message || '接続を追加できませんでした'); }
  finally { $('save-connection').disabled = false; }
});
document.addEventListener('click', async event => {
  const button = event.target.closest('button'); if (!button) return;
  if (button.dataset.view) showView(button.dataset.view);
  if (button.dataset.target) { selectTarget(button.dataset.target); showView('overview'); }
  if (button.dataset.selectTarget) { selectTarget(button.dataset.selectTarget); showView('overview'); }
  if (button.dataset.chart) { chartKey = button.dataset.chart; document.querySelectorAll('[data-chart]').forEach(b => b.classList.toggle('active', b === button)); render(); }
  if (button.dataset.delete) {
    button.disabled = true;
    try {
      const response = await fetch('/api/targets', {method:'DELETE', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id:button.dataset.delete}), signal:AbortSignal.timeout(6000)});
      const result = await response.json(); if (!response.ok) throw new Error(result.error);
      await refresh(true); notify('接続を削除しました');
    } catch (error) { notify(error.message || '削除できませんでした'); button.disabled = false; }
  }
});
$('pause').addEventListener('click', () => {
  paused = !paused; $('pause').textContent = paused ? '▷' : 'Ⅱ';
  const label = paused ? '表示の更新を再開' : '表示の更新を一時停止'; $('pause').setAttribute('aria-label', label); $('pause').title = label;
  $('live-indicator').classList.toggle('paused', paused); $('live-indicator').innerHTML = `<span class="status-dot"></span>${paused ? 'PAUSED' : 'LIVE'}`;
  render(); if (!paused) refresh(true);
});
$('refresh').addEventListener('click', () => refresh(true));
$('history-window').addEventListener('change', event => { windowSeconds = Number(event.target.value); render(); });
$('model-select').addEventListener('change', event => { selectedModel = event.target.value; render(); });
$('export').addEventListener('click', () => {
  if (!data) { notify('計測データを取得してから保存できます'); return; }
  const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'}), url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = `llm-observatory-${new Date().toISOString().replace(/[:.]/g, '-')}.json`; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  notify('ローカルへスナップショットを保存しました');
});
async function poll() { await refresh(); setTimeout(poll, 2000); }
poll();
