'use strict';

const $ = id => document.getElementById(id);
const overlay = $('overlay');
const raw = $('raw');
const overview = $('overview');
const cropImage = document.createElement('canvas');
const overviewImage = document.createElement('canvas');

const defaults = {
  minScore: 0, blackOn: false, blackLevel: 118, rimOn: false, rimScore: .60,
  rimLevel: 2, sectorsOn: false, sectorsLevel: 25,
};
const presets = {
  none: defaults,
  black: { ...defaults, blackOn: true },
  probe: { ...defaults, blackOn: true, rimOn: true },
};
const state = {
  ...defaults, model: 'wide64', rois: [], marked: new Set(), reasons: [], selected: null,
  y: 0, x: 0, size: 96, loaded: false, saving: false, modelRequest: 0,
};

const rules = [
  { sliders: [['minScore', 'Минимальный score центра', 0, 1, .01]] },
  { id: 'blackOn', name: 'Почти чёрные ROI',
    sliders: [['blackLevel', 'Максимум P95 окрестности, ADU', 95, 260, 1]] },
  { id: 'rimOn', name: 'Слабая оболочка при низком score',
    sliders: [['rimScore', 'Score ниже', 0, 1, .01], ['rimLevel', 'Перепад оболочки ниже, ADU', -8, 25, .5]] },
  { id: 'sectorsOn', name: 'Доля ярких секторов оболочки',
    sliders: [['sectorsLevel', 'Минимальная доля, %', 0, 100, 1]] },
];

function makeControls() {
  for (const rule of rules) {
    const block = document.createElement('div');
    block.className = 'rule';
    const fields = document.createElement('fieldset');
    if (rule.id) {
      const label = document.createElement('label');
      const input = document.createElement('input');
      input.type = 'checkbox';
      input.id = rule.id;
      input.addEventListener('change', () => {
        state[rule.id] = input.checked;
        changed();
      });
      label.append(input, ' ' + rule.name);
      block.append(label);
      fields.id = rule.id + '-fields';
    }
    for (const [key, title, min, max, step] of rule.sliders) {
      const wrapper = document.createElement('div');
      wrapper.className = 'slider';
      const label = document.createElement('label');
      label.htmlFor = key;
      label.textContent = title;
      wrapper.append(label);
      for (const type of ['range', 'number']) {
        const input = document.createElement('input');
        Object.assign(input, { type, min, max, step, id: type === 'range' ? key : key + '-number' });
        if (type === 'number') input.setAttribute('aria-label', title);
        input.addEventListener(type === 'range' ? 'input' : 'change', () => {
          const value = input.valueAsNumber;
          if (Number.isFinite(value)) state[key] = Math.max(min, Math.min(max, value));
          changed();
        });
        wrapper.append(input);
      }
      fields.append(wrapper);
    }
    block.append(fields);
    $('thresholds').append(block);
  }
}

function syncControls() {
  for (const rule of rules) {
    if (rule.id) {
      $(rule.id).checked = state[rule.id];
      $(rule.id + '-fields').disabled = !state[rule.id];
    }
    for (const [key] of rule.sliders) {
      if ($(key).valueAsNumber !== state[key]) $(key).value = state[key];
      $(key + '-number').value = state[key];
    }
  }
}

function changed() {
  $('preset').value = 'manual';
  syncControls();
  schedule();
}

function rejected(roi) {
  const reasons = [];
  if (roi.score < state.minScore) reasons.push('низкий score');
  if (state.blackOn && roi.p95 <= state.blackLevel) reasons.push('почти чёрный фон');
  if (state.rimOn && roi.score < state.rimScore && roi.rim < state.rimLevel) reasons.push('слабая оболочка');
  if (state.sectorsOn && roi.sectors * 100 < state.sectorsLevel) reasons.push('мало ярких секторов');
  return reasons;
}

function asRoi([y, x, score, p95, rim, sectors, paths], id) {
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  const path = new Path2D();
  for (const item of paths) {
    path.moveTo(item[0], item[1]);
    for (let i = 0; i < item.length; i += 2) {
      const py = item[i], px = item[i + 1];
      path.lineTo(py, px);
      bounds[0] = Math.min(bounds[0], py);
      bounds[1] = Math.min(bounds[1], px);
      bounds[2] = Math.max(bounds[2], py);
      bounds[3] = Math.max(bounds[3], px);
    }
    path.closePath();
  }
  return { id, y, x, score, p95, rim, sectors, bounds, path };
}

async function request(path, options, binary = false) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error('HTTP ' + response.status);
  return binary ? response.arrayBuffer() : response.json();
}

async function loadModel(model) {
  const serial = ++state.modelRequest;
  $('status').textContent = 'Загрузка кандидатов…';
  state.loaded = false;
  state.selected = null;
  schedule();
  try {
    const [rows, marked] = await Promise.all([request('/api/roi/' + model), request('/api/marks/' + model)]);
    if (state.modelRequest !== serial) return;
    state.model = model;
    state.rois = rows.map(asRoi);
    state.marked = new Set(marked.map(Number));
    state.loaded = true;
    const info = state.meta.models[model];
    $('fixed').textContent = `Исходные пороги инференса: центр ${info.center_threshold.toFixed(2)}, маска ${info.mask_threshold.toFixed(2)}.`;
    $('status').textContent = 'Отметки сохраняются автоматически';
    schedule();
  } catch (error) {
    if (state.modelRequest === serial) $('status').textContent = 'Не удалось загрузить кандидатов. ' + error.message;
  }
}

function paintFrame(canvas, y0, x0, width, height, limits) {
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext('2d');
  const output = context.createImageData(width, height);
  const [lo, hi] = limits;
  const gain = 255 / Math.max(1e-5, hi - lo);
  const data = output.data;
  for (let x = 0; x < height; x++) {
    for (let y = 0; y < width; y++) {
      const k = 4 * (x * width + y);
      const light = (state.pixels[(y + y0) * state.width + x + x0] - lo) * gain;
      data[k] = data[k + 1] = data[k + 2] = light;
      data[k + 3] = 255;
    }
  }
  context.putImageData(output, 0, 0);
}

function paintCrop() {
  const { y, x, size } = state;
  const local = $('localContrast').checked;
  const stamp = [y, x, size, local].join(':');
  if (state.cropStamp === stamp) return;
  state.cropStamp = stamp;
  let limits = state.meta.display_range;
  if (local) {
    const samples = [];
    for (let py = y; py < y + size; py++) {
      for (let px = x; px < x + size; px++) samples.push(state.pixels[py * state.width + px]);
    }
    samples.sort((a, b) => a - b);
    const at = q => samples[Math.floor(q * (samples.length - 1))];
    limits = [at(.01), at(.995)];
  }
  paintFrame(cropImage, y, x, size, size, limits);
  const context = raw.getContext('2d');
  context.imageSmoothingEnabled = false;
  context.drawImage(cropImage, 0, 0, raw.width, raw.height);
}

function insideView(roi) {
  const [y0, x0, y1, x1] = roi.bounds;
  return y0 < state.y + state.size && y1 >= state.y && x0 < state.x + state.size && x1 >= state.x;
}

function drawROI(roi, context, selected, scale) {
  const marked = state.marked.has(roi.id);
  const inactive = state.reasons[roi.id].length > 0;
  context.strokeStyle = selected ? '#fbf2a2' : marked ? '#ffbd73' : inactive ? '#aab8d2' : '#6cecb1';
  context.globalAlpha = selected || marked ? 1 : inactive ? .58 : .95;
  context.lineWidth = (selected ? 2.5 : marked ? 2 : 1.2) / scale;
  context.setLineDash(inactive ? [3.5 / scale, 2.8 / scale] : []);
  context.stroke(roi.path);
}

function refreshSelection() {
  const roi = state.rois[state.selected];
  const button = $('mark');
  button.disabled = !state.loaded || !roi || state.saving;
  if (!roi || !state.loaded) {
    $('selected-title').textContent = 'Щёлкните по маске';
    $('selected-detail').textContent = 'Shift + щелчок отмечает ROI сразу';
    button.textContent = 'Отметить как неверный';
    return;
  }
  const reasons = state.reasons[roi.id];
  $('selected-title').textContent = `ROI ${roi.id + 1} · ${reasons.length ? 'отсечён' : 'активен'}`;
  $('selected-detail').textContent =
    `Score ${roi.score.toFixed(2)} · P95 ${roi.p95.toFixed(1)} ADU · оболочка ${roi.rim.toFixed(1)} ADU · яркие секторы ${Math.round(roi.sectors * 100)}%` +
    (reasons.length ? ' · ' + reasons.join(', ') : '');
  button.textContent = state.marked.has(roi.id) ? 'Снять отметку' : 'Отметить как неверный';
}

function refreshMarkList() {
  const stamp = [state.model, ...Array.from(state.marked, id => id + ':' + state.reasons[id].length)].join('|');
  if (state.markStamp === stamp) return;
  state.markStamp = stamp;
  let off = 0;
  const list = document.createDocumentFragment();
  for (const id of state.marked) {
    const roi = state.rois[id];
    if (!roi) continue;
    const rejected = state.reasons[id].length > 0;
    if (rejected) off++;
    const item = document.createElement('button');
    item.type = 'button';
    item.className = 'mark-row';
    const tag = document.createElement('span');
    tag.textContent = rejected ? 'Отсечён' : 'Активен';
    item.append(`ROI ${id + 1} · ${Math.round(roi.y)}, ${Math.round(roi.x)}`, tag);
    item.addEventListener('click', () => {
      state.selected = id;
      navigate(roi.y, roi.x);
    });
    list.append(item);
  }
  $('markList').replaceChildren(list);
  $('markedCount').textContent = state.marked.size;
  $('markedOff').textContent = off;
  $('markedOn').textContent = state.marked.size - off;
}

function render() {
  if (!state.pixels) return;
  paintCrop();
  const context = overview.getContext('2d');
  context.drawImage(overviewImage, 0, 0);
  context.strokeStyle = '#c9e1ff';
  context.lineWidth = 2;
  context.strokeRect(state.y, state.x, state.size, state.size);
  $('coordinates').textContent = `Y ${state.y} · X ${state.x} · ${state.size} × ${state.size} px`;

  const draw = overlay.getContext('2d');
  draw.imageSmoothingEnabled = false;
  draw.drawImage(cropImage, 0, 0, overlay.width, overlay.height);
  if (!state.loaded) {
    $('inside').textContent = '…';
    refreshSelection();
    return;
  }
  state.reasons = state.rois.map(rejected);
  const active = state.reasons.reduce((count, items) => count + Number(!items.length), 0);
  const visible = state.rois.filter(insideView);
  state.clickable = visible.filter(roi => !state.reasons[roi.id].length || state.marked.has(roi.id) ||
    $('showRejected').checked || state.selected === roi.id);

  const scale = overlay.width / state.size;
  draw.save();
  draw.scale(scale, scale);
  draw.translate(-state.y + .5, -state.x + .5);
  for (const roi of state.clickable) {
    if (roi.id !== state.selected) drawROI(roi, draw, false, scale);
  }
  const selected = state.rois[state.selected];
  if (selected && insideView(selected)) drawROI(selected, draw, true, scale);
  draw.restore();
  const inView = visible.filter(roi => !state.reasons[roi.id].length).length;
  $('inside').textContent = `${inView} в поле · ${active} / ${state.rois.length} на кадре`;
  refreshSelection();
  refreshMarkList();
}

let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => {
    pending = false;
    render();
  });
}

function navigate(y, x, fromPreset = false) {
  state.y = Math.max(0, Math.min(state.height - state.size, Math.round(y - state.size / 2)));
  state.x = Math.max(0, Math.min(state.width - state.size, Math.round(x - state.size / 2)));
  if (!fromPreset) $('crop').value = 'custom';
  schedule();
}

async function markSelected() {
  if (!state.loaded || state.selected === null || state.saving) return;
  const { model, selected: id } = state;
  const modelRequest = state.modelRequest;
  state.saving = true;
  $('status').textContent = 'Сохранение…';
  schedule();
  try {
    const marks = await request('/api/mark', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, id, marked: !state.marked.has(id) }),
    });
    if (modelRequest === state.modelRequest) {
      state.marked = new Set(marks.map(Number));
      $('status').textContent = 'Отметка сохранена на диск';
    }
  } catch (error) {
    $('status').textContent = 'Отметку сохранить не удалось. ' + error.message;
  } finally {
    state.saving = false;
    schedule();
  }
}

function position(canvas, event) {
  const bounds = canvas.getBoundingClientRect();
  return [(event.clientX - bounds.left) / bounds.width, (event.clientY - bounds.top) / bounds.height];
}

let lastClick;
overlay.addEventListener('click', event => {
  if (!state.loaded) return;
  const [u, v] = position(overlay, event);
  const y = state.y + u * state.size - .5;
  const x = state.x + v * state.size - .5;
  const context = overlay.getContext('2d');
  const candidates = state.clickable.map(roi => {
    const distance = Math.hypot(roi.y - y, roi.x - x);
    const contains = context.isPointInPath(roi.path, y, x, 'evenodd');
    return { roi, distance, contains };
  }).filter(item => item.contains || item.distance < 4)
    .sort((a, b) => Number(b.contains) - Number(a.contains) || a.distance - b.distance);
  const repeat = lastClick && Math.hypot(lastClick.y - y, lastClick.x - x) < 1;
  const current = candidates.findIndex(item => item.roi.id === state.selected);
  const index = repeat && !event.shiftKey ? (current + 1) % candidates.length : 0;
  state.selected = candidates[index]?.roi.id ?? null;
  lastClick = { y, x };
  schedule();
  if (event.shiftKey) markSelected();
});

overview.addEventListener('click', event => {
  if (!state.pixels) return;
  const [u, v] = position(overview, event);
  navigate(u * state.height, v * state.width);
});

for (const canvas of [overview, overlay]) {
  canvas.addEventListener('keydown', event => {
    const movement = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[event.key];
    if (movement && state.pixels) {
      event.preventDefault();
      navigate(state.y + state.size * (.5 + movement[0] / 3), state.x + state.size * (.5 + movement[1] / 3));
    } else if (event.key === ' ' && canvas === overlay && state.selected !== null) {
      event.preventDefault();
      markSelected();
    }
  });
}

$('size').addEventListener('change', () => {
  const y = state.y + state.size / 2, x = state.x + state.size / 2;
  state.size = Number($('size').value);
  navigate(y, x);
});
$('crop').addEventListener('change', () => {
  const crop = state.meta.crops[Number($('crop').value)];
  if (crop) navigate(crop[1] + 32, crop[2] + 32, true);
});
$('model').addEventListener('change', () => loadModel($('model').value));
$('localContrast').addEventListener('change', schedule);
$('showRejected').addEventListener('change', schedule);
$('mark').addEventListener('click', markSelected);
$('preset').addEventListener('change', () => {
  Object.assign(state, presets[$('preset').value]);
  syncControls();
  schedule();
});
$('reset').addEventListener('click', () => {
  $('preset').value = 'none';
  $('preset').dispatchEvent(new Event('change'));
});

async function init() {
  makeControls();
  syncControls();
  try {
    const [meta, buffer] = await Promise.all([request('/api/meta'), request('/api/frame', {}, true)]);
    state.meta = meta;
    [state.height, state.width] = meta.shape;
    state.pixels = new Float32Array(buffer);
    overview.width = state.height;
    overview.height = state.width;
    paintFrame(overviewImage, 0, 0, state.height, state.width, meta.display_range);
    for (const [id, info] of Object.entries(meta.models)) $('model').append(new Option(info.name, id));
    meta.crops.forEach(([name], i) => $('crop').append(new Option(name, i)));
    $('crop').append(new Option('Произвольный', 'custom'));
    $('crop').selectedIndex = 4;
    $('crop').dispatchEvent(new Event('change'));
    await loadModel($('model').value);
  } catch (error) {
    $('status').textContent = 'Не удалось открыть медиану. ' + error.message;
  }
}

init();
