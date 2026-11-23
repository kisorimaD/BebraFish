const element = (id) => document.getElementById(id);
const controls = ['model', 'model-filter', 'yes', 'no', 'bad-mask', 'uncertain', 'clear',
  'previous', 'next', 'pending'].map(element);
const statusNames = {accepted: 'Принято', rejected: 'Отклонено', no_soma: 'Нет сомы',
  bad_mask: 'Плохая маска', uncertain: 'Неясно'};

let proposals, allProposals, models, pixels, height, width, globalRange, reviewRois;
let decisions = {}, pointer = 0, candidateIndex = 0, saving = false;
let overviewImage;

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response;
}

function grayImage(y0, x0, spanY, spanX, range) {
  const image = document.createElement('canvas');
  image.width = spanY;
  image.height = spanX;
  const context = image.getContext('2d');
  const bitmap = context.createImageData(spanY, spanX);
  const scale = 255 / Math.max(range[1] - range[0], 1);
  for (let x = 0; x < spanX; x++) {
    for (let y = 0; y < spanY; y++) {
      const value = (pixels[(y + y0) * width + x + x0] - range[0]) * scale;
      const i = 4 * (x * spanY + y);
      bitmap.data[i] = bitmap.data[i + 1] = bitmap.data[i + 2] = value;
      bitmap.data[i + 3] = 255;
    }
  }
  context.putImageData(bitmap, 0, 0);
  return image;
}

function localRange(cy, cx) {
  const values = [];
  for (let y = Math.max(0, cy - 48); y < Math.min(height, cy + 48); y++) {
    for (let x = Math.max(0, cx - 48); x < Math.min(width, cx + 48); x++) {
      values.push(pixels[y * width + x]);
    }
  }
  values.sort((a, b) => a - b);
  return [values[Math.floor((values.length - 1) * .02)], values[Math.floor((values.length - 1) * .99)]];
}

function drawCrops(mask) {
  const span = Math.min(+element('field').value, width, height);
  const [cy, cx] = mask.center_yx.map(Math.round);
  const y0 = Math.max(0, Math.min(height - span, cy - Math.floor(span / 2)));
  const x0 = Math.max(0, Math.min(width - span, cx - Math.floor(span / 2)));
  const range = element('contrast').value === 'local' ? localRange(cy, cx) : globalRange;
  const source = grayImage(y0, x0, span, span, range);
  const step = 10;
  for (const id of ['raw', 'overlay']) {
    const canvas = element(id);
    canvas.width = canvas.height = span * step;
    const context = canvas.getContext('2d');
    context.imageSmoothingEnabled = false;
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
  }
  // Display is transposed. Draw exposed pixel edges of the exact model mask.
  const context = element('overlay').getContext('2d');
  context.scale(step, step);
  const footprint = new Set();
  for (const [y, start, stop] of mask.mask_runs_yx) {
    for (let x = start; x < stop; x++) footprint.add(y * width + x);
  }
  context.fillStyle = '#64e8c52c';
  context.strokeStyle = '#87ffe0';
  context.lineWidth = .22;
  context.beginPath();
  for (const p of footprint) {
    const y = Math.floor(p / width), x = p % width;
    const u = y - y0, v = x - x0;
    context.fillRect(u, v, 1, 1);
    if (!footprint.has(p - width)) { context.moveTo(u, v); context.lineTo(u, v + 1); }
    if (!footprint.has(p + width)) { context.moveTo(u + 1, v); context.lineTo(u + 1, v + 1); }
    if (x === 0 || !footprint.has(p - 1)) { context.moveTo(u, v); context.lineTo(u + 1, v); }
    if (x === width - 1 || !footprint.has(p + 1)) { context.moveTo(u, v + 1); context.lineTo(u + 1, v + 1); }
  }
  context.stroke();

  const overview = element('overview').getContext('2d');
  overview.drawImage(overviewImage, 0, 0);
  for (const group of proposals) {
    if (decisions[group.id]?.status !== 'accepted') continue;
    const selected = group.candidates.find((c) => c.id === decisions[group.id].candidate_id);
    const [y, x] = selected.center_yx;
    overview.fillStyle = '#85eac9';
    overview.fillRect(y + .5 - 1.3, x + .5 - 1.3, 2.6, 2.6);
  }
  overview.lineWidth = 2;
  overview.strokeStyle = '#d5fff3';
  overview.strokeRect(y0, x0, span, span);
}

function render() {
  const group = proposals[pointer];
  const decision = decisions[group.id];
  const mask = group.candidates[candidateIndex];
  const done = proposals.map((item) => decisions[item.id]).filter(Boolean);
  const reviewed = done.length;
  element('accepted').textContent = done.filter((item) => item.status === 'accepted').length;
  element('rejected').textContent = done.filter((item) => item.status === 'rejected' &&
    (!reviewRois || item.reason === 'no_soma')).length;
  element('bad-count').textContent = done.filter((item) => item.reason === 'bad_mask').length;
  element('uncertain-count').textContent = done.filter((item) => item.status === 'uncertain').length;
  element('remaining').textContent = proposals.length - reviewed;
  element('current-id').textContent = group.id;
  element('current-state').textContent = statusNames[decision?.reason ?? decision?.status] ?? 'Без решения';
  element('current-state').dataset.status = decision?.status ?? '';
  element('prior').hidden = !group.nearby_manual_annotation;
  element('position').textContent = `${pointer + 1} / ${proposals.length}`;
  element('coordinates').textContent = `Y ${mask.center_yx[0].toFixed(1)} · X ${mask.center_yx[1].toFixed(1)}`;
  element('details').textContent = `Уверенность центра ${mask.score.toFixed(2)} · ${mask.area_px} px²` +
    (mask.source_instance_index === undefined ? '' : ` · ROI ${mask.source_instance_index + 1}`);
  const select = element('model');
  select.replaceChildren(...group.candidates.map((c, i) => new Option(models.get(c.model_id), i)));
  select.value = candidateIndex;
  for (const control of controls) control.disabled = saving;
  select.disabled ||= group.candidates.length < 2;
  element('previous').disabled ||= pointer === 0;
  element('next').disabled ||= pointer === proposals.length - 1;
  element('pending').disabled ||= reviewed === proposals.length;
  element('clear').hidden = !decision;
  drawCrops(mask);
}

function selectCard(i) {
  if (i < 0 || i >= proposals.length) return;
  pointer = i;
  const group = proposals[pointer];
  candidateIndex = Math.max(0, group.candidates.findIndex((c) => c.id === decisions[group.id]?.candidate_id));
  render();
}

function nextPending(start) {
  for (let step = 0; step < proposals.length; step++) {
    const i = (start + step) % proposals.length;
    if (!decisions[proposals[i].id]) return i;
  }
  return -1;
}

async function decide(status, reason = null) {
  if (saving) return;
  const group = proposals[pointer];
  const indicator = element('save-state');
  saving = true;
  indicator.removeAttribute('data-error');
  indicator.textContent = 'Сохраняю…';
  render();
  try {
    const response = await request('/api/decision', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: group.id, candidate_id: group.candidates[candidateIndex].id, status, reason})
    });
    decisions = (await response.json()).decisions;
    indicator.textContent = 'Сохранено на диск';
    if (status !== null) selectCard(nextPending(pointer + 1));
  } catch (error) {
    indicator.dataset.error = '';
    indicator.textContent = `Ошибка сохранения (${error.message}). Повторите.`;
  } finally {
    saving = false;
    render();
  }
}

async function start() {
  const [raw, queue, state] = await Promise.all([
    request('/api/frame').then((response) => response.arrayBuffer()),
    request('/api/proposals').then((response) => response.json()),
    request('/api/decisions').then((response) => response.json())
  ]);
  allProposals = proposals = queue.proposals;
  reviewRois = queue.review_rois === true;
  models = new Map(queue.models.map((model) => [model.id, model.name]));
  decisions = state.decisions;
  [height, width] = queue.dataset.shape_yx;
  globalRange = queue.dataset.display_range;
  const float = queue.dataset.dtype === 'float32';
  const stride = float ? 4 : 2;
  if (raw.byteLength !== stride * height * width) throw new Error('Размер кадра не совпадает');
  const view = new DataView(raw);
  pixels = float ? Float32Array.from({length: height * width}, (_, i) => view.getFloat32(i * 4, true))
    : Uint16Array.from({length: height * width}, (_, i) => view.getUint16(i * 2, true));
  element('source').textContent = `${queue.dataset.source_name} · кадр ${queue.dataset.frame_index}`;
  overviewImage = grayImage(0, 0, height, width, globalRange);
  element('overview').width = height;
  element('overview').height = width;
  element('review').hidden = element('download').hidden = false;
  if (reviewRois) {
    element('source').textContent = 'Plane20 · зарегистрированная медиана';
    element('original').textContent = 'Медиана без маски';
    element('prompt').textContent = 'Маска охватывает всю сому по её видимому внешнему контуру?';
    element('no-label').textContent = 'Нет сомы';
    element('rejected-label').textContent = 'Нет сомы';
    for (const id of ['bad-mask', 'uncertain', 'bad-progress', 'uncertain-progress', 'model-filter-control']) {
      element(id).hidden = false;
    }
    const filter = element('model-filter');
    filter.replaceChildren(new Option('Обе модели', 'all'),
      ...[...models].map(([id, name]) => new Option(name, id)));
    filter.onchange = () => {
      const value = filter.value;
      proposals = value === 'all' ? allProposals : allProposals.filter((group) => group.candidates[0].model_id === value);
      selectCard(Math.max(0, nextPending(0)));
      filter.blur();
    };
  }
  element('save-state').textContent = 'Готово к разметке';
  selectCard(Math.max(0, nextPending(0)));

  element('yes').onclick = () => decide('accepted');
  element('no').onclick = () => decide('rejected', reviewRois ? 'no_soma' : null);
  element('bad-mask').onclick = () => decide('rejected', 'bad_mask');
  element('uncertain').onclick = () => decide('uncertain');
  element('clear').onclick = () => decide(null);
  element('previous').onclick = () => selectCard(pointer - 1);
  element('next').onclick = () => selectCard(pointer + 1);
  element('pending').onclick = () => selectCard(nextPending(pointer + 1));
  element('model').onchange = (event) => {
    candidateIndex = +event.target.value;
    event.target.blur();
    render();
  };
  element('field').onchange = element('contrast').onchange = render;
  element('overview').onclick = (event) => {
    if (saving) return;
    const box = element('overview').getBoundingClientRect();
    const y = (event.clientX - box.left) * height / box.width - .5;
    const x = (event.clientY - box.top) * width / box.height - .5;
    let best = -1, min = Infinity;
    proposals.forEach((group, i) => {
      const [cy, cx] = group.candidates[0].center_yx;
      const distance = (cy - y) ** 2 + (cx - x) ** 2;
      if (distance < min) { min = distance; best = i; }
    });
    selectCard(best);
  };
  document.addEventListener('keydown', (event) => {
    if (saving || event.repeat || event.altKey || event.ctrlKey || event.metaKey
        || event.target.matches('select, input, textarea')) return;
    const action = {KeyY: 'yes', KeyN: 'no', KeyM: 'bad-mask', KeyU: 'uncertain',
      ArrowLeft: 'previous', ArrowRight: 'next'}[event.code];
    if (action && !element(action).hidden) { event.preventDefault(); element(action).click(); }
  });
}

start().catch((error) => {
  element('error').hidden = false;
  element('error').textContent = `Не удалось открыть сессию. ${error.message}`;
  element('save-state').textContent = 'Ошибка загрузки';
});
