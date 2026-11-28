const byId = id => document.getElementById(id);
const steps = [...document.querySelectorAll('[data-step]')];
const canvas = byId('image');
const context = canvas.getContext('2d');
let base, image, crops, previewBase, activeStep, reviewPoll;
let saveQueue = Promise.resolve();

async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Ошибка запроса');
  return value;
}

function status(text, error = false) {
  byId('status').textContent = text;
  byId('status').dataset.error = String(error);
}

function showStep(active) {
  activeStep = active;
  for (const button of steps) {
    const step = button.dataset.step;
    byId(step).hidden = step !== active;
    button.setAttribute('aria-current', step === active ? 'step' : 'false');
  }
  const frame = byId(active).querySelector('iframe');
  if (active === 'refinement') refreshReview();
  if (frame && active !== 'refinement' && !frame.hasAttribute('src')) frame.src = base + active + '/';
  if (active === 'centers') drawCrop();
}

for (const button of steps) button.onclick = () => {
  showStep(button.dataset.step);
  saveDraft();
};

async function waitFor(jobBase, pending, onStatus = () => {}) {
  while (true) {
    const state = await request(jobBase + 'status');
    if (state.stage !== pending) {
      if (state.stage === 'prepare_error' || (pending === 'generating' && state.stage === 'fit_error')) throw new Error(state.error);
      return state;
    }
    onStatus(state);
    await new Promise(resolve => setTimeout(resolve, 600));
  }
}

async function loadImage(url) {
  const loaded = new Image();
  loaded.src = url;
  await loaded.decode();
  return loaded;
}

function draftKey(project) {
  return 'bebrafish-draft:' + project;
}

function saveDraft() {
  if (!base || !crops) return;
  const project = base;
  const draft = {
    updated: Date.now(), crops, step: activeStep, cropIndex: byId('crop').selectedIndex,
    count: byId('count').value, seed: byId('seed').value, grow_somas: byId('grow-somas').checked, normalize: byId('p5-p95').checked
  };
  const snapshot = JSON.stringify(draft);
  try { localStorage.setItem(draftKey(project), snapshot); } catch {}
  saveQueue = saveQueue.catch(() => {}).then(() => request(project + 'draft', JSON.parse(snapshot)));
  saveQueue.catch(error => {
    if (base === project) status('Не удалось сохранить прогресс: ' + error.message, true);
  });
}

async function listProjects() {
  const projects = await request('/projects');
  const select = byId('projects');
  select.replaceChildren(new Option('Выберите проект', ''));
  for (const project of projects.filter(project => project.stage !== 'prepare_error').reverse()) {
    select.add(new Option(project.name + ' · ' + project.job.slice(0, 8), project.job));
  }
}

function busy(value) {
  for (const id of ['projects', 'choose']) byId(id).disabled = value;
}

function initialCrops(reference, soma) {
  const size = Math.min(64, reference.width, reference.height);
  const result = [];
  for (const crop of soma.crops) {
    const x = crop.x + Math.floor((crop.width-size)/2);
    const y = crop.y + Math.floor((crop.height-size)/2);
    if (crop.width >= size && crop.height >= size &&
        !result.some(other => Math.abs(other.x-x) < size && Math.abs(other.y-y) < size)) {
      result.push({x, y, width: size, height: size, points: [], reviewed: false});
    }
  }
  return result.length ? result.slice(0, 30) : [{x: 0, y: 0, width: size, height: size, points: [], reviewed: false}];
}

async function openProject(token, requestedStep) {
  busy(true);
  clearTimeout(reviewPoll);
  byId('baseline-links').hidden = true;
  byId('workflow').hidden = byId('result').hidden = byId('comparison').hidden = byId('metrics').hidden = true;
  status('Открываем проект…');
  base = '/jobs/' + token + '/';
  crops = null;
  history.replaceState(null, '', '/#' + token);
  try {
    const state = await waitFor(base, 'preparing');
    const [reference, soma, stored] = await Promise.all([
      loadImage(base + 'reference.png'), request(base + 'soma/api/annotations'), request(base + 'draft')
    ]);
    let draft = stored;
    try {
      const local = JSON.parse(localStorage.getItem(draftKey(base)));
      if (local && local.updated > (stored.updated || 0)) draft = local;
    } catch {}
    image = reference;
    crops = draft.crops || initialCrops(reference, soma);
    if (!draft.crops && draft.reviewed) {
      for (const saved of draft.reviewed) {
        const index = crops.findIndex(crop => ['x', 'y', 'width', 'height'].every(key => crop[key] === saved[key]));
        const reviewed = {...saved, reviewed: true};
        if (index < 0) crops.push(reviewed);
        else crops[index] = reviewed;
      }
      crops = [...crops.filter(crop => crop.reviewed), ...crops.filter(crop => !crop.reviewed)].slice(0, 30);
    }
    byId('count').value = draft.count ?? 8;
    byId('seed').value = draft.seed ?? 20261124;
    byId('grow-somas').checked = draft.grow_somas === true;
    byId('p5-p95').checked = Boolean(draft.normalize);
    byId('crop').replaceChildren(...crops.map((_, i) => new Option('Кроп ' + (i+1), i)));
    byId('crop').selectedIndex = Math.min(crops.length-1, Math.max(0, draft.cropIndex || 0));
    for (const frame of document.querySelectorAll('iframe')) frame.removeAttribute('src');
    byId('projects').value = token;
    byId('workflow').hidden = false;
    const fallback = ['done', 'generating'].includes(state.stage) ? 'generate' : 'soma';
    showStep(requestedStep === 'generate' ? requestedStep :
      steps.some(button => button.dataset.step === draft.step) ? draft.step : fallback);
    if (draft !== stored) saveDraft();
    if (state.stage === 'done') showResult(state);
    else if (state.stage === 'generating') await trackGeneration();
    else status('Проект открыт.');
  } catch (error) {
    status(error.message, true);
  } finally {
    busy(false);
  }
}

byId('projects').onchange = event => {
  if (event.target.value) openProject(event.target.value);
};
byId('choose').onclick = () => byId('file').click();
byId('file').onchange = async event => {
  const file = event.target.files[0];
  if (!file) return;
  busy(true);
  status('Загружаем TIFF в локальное приложение…');
  try {
    const response = await fetch('/upload', {
      method: 'POST', headers: {'X-Filename': encodeURIComponent(file.name)}, body: file
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    await listProjects();
    await openProject(data.job);
  } catch (error) {
    status(error.message, true);
  } finally {
    event.target.value = '';
    busy(false);
  }
};

function selectedCrop() {
  return crops[Number(byId('crop').value)];
}

function drawCrop() {
  const crop = selectedCrop();
  const scale = 6;
  canvas.width = crop.width * scale;
  canvas.height = crop.height * scale;
  context.imageSmoothingEnabled = false;
  context.drawImage(image, crop.x, crop.y, crop.width, crop.height, 0, 0, canvas.width, canvas.height);
  context.strokeStyle = '#6affcd';
  context.lineWidth = 2;
  for (const [x, y] of crop.points) {
    context.beginPath();
    context.arc((x-crop.x+.5)*scale, (y-crop.y+.5)*scale, 5, 0, 2*Math.PI);
    context.stroke();
  }
  byId('holdout').checked = Boolean(crop.holdout);
  byId('reviewed').checked = crop.reviewed;
  byId('marked').textContent = 'Центров в кропе: ' + crop.points.length +
    ' · Просмотрено кропов: ' + crops.filter(item => item.reviewed).length;
}

byId('crop').onchange = () => { drawCrop(); saveDraft(); };
byId('holdout').onchange = event => {
  selectedCrop().holdout = event.currentTarget.checked;
  saveDraft();
};
byId('reviewed').onchange = event => {
  selectedCrop().reviewed = event.currentTarget.checked;
  drawCrop();
  saveDraft();
};
canvas.onclick = event => {
  const crop = selectedCrop();
  const bounds = canvas.getBoundingClientRect();
  const x = Math.min(crop.width-1, Math.floor((event.clientX-bounds.left)*crop.width/bounds.width)) + crop.x;
  const y = Math.min(crop.height-1, Math.floor((event.clientY-bounds.top)*crop.height/bounds.height)) + crop.y;
  const index = crop.points.findIndex(([px, py]) => Math.hypot(px-x, py-y) <= 1.5);
  if (index < 0) crop.points.push([x, y]);
  else crop.points.splice(index, 1);
  crop.reviewed = false;
  drawCrop();
  saveDraft();
};

function showResult(state) {
  const output = byId('result');
  const {cells, target} = state.result;
  status('Готово. Сцен: ' + cells.length + ' · Сом на сцену: ' +
    Math.min(...cells) + '–' + Math.max(...cells) + ' · Цель по плотности: ' + target);
  output.replaceChildren();
  previewBase = base + state.attempt + '/';
  for (const [name, file] of [['Скачать датасет ZIP', 'dataset.zip'],
                             ['Пример TIFF', 'example.tiff'], ['Отчёт JSON', 'report.json']]) {
    if (output.childNodes.length) output.append(' · ');
    const anchor = document.createElement('a');
    anchor.href = previewBase + file;
    anchor.download = file;
    anchor.textContent = name;
    output.append(anchor);
  }
  const light = document.createElement('a');
  light.href = previewBase + 'light';
  light.textContent = 'Настроить свечение';
  output.append(' · ', light);
  output.hidden = false;
  updatePreviews();
  byId('metrics').open = false;
  byId('metrics-image').removeAttribute('src');
  byId('metrics-link').href = previewBase + 'metrics.png';
  byId('metrics-status').textContent = '';
  byId('metrics').hidden = false;
  byId('comparison').hidden = false;
  const project = base;
  request(previewBase + 'report.json').then(report => {
    if (project !== base || previewBase !== project + state.attempt + '/') return;
    const links = byId('baseline-links');
    links.replaceChildren();
    if (report.refinement) {
      const original = base + report.refinement.source_run + '/';
      const files = [['Скачать итоговые контуры', previewBase + 'soma.json'],
                     ['Графики исходного прогона', original + 'metrics.png'],
                     ['Исходный датасет', original + 'dataset.zip']];
      if (report.heldout_crops) files.unshift(['Сравнение на контрольных кропах', previewBase + 'heldout-metrics.png']);
      for (const [label, url] of files) {
        if (links.childNodes.length) links.append(' · ');
        const link = document.createElement('a');
        link.href = url;
        if (url.endsWith('/soma.json')) link.download = 'soma.json';
        link.textContent = label;
        links.append(link);
      }
    }
    if (report.manual_light && report.manual_light.source_run !== report.refinement?.source_run) {
      const link = document.createElement('a');
      link.href = base + report.manual_light.source_run + '/dataset.zip';
      link.textContent = 'Датасет до ручной настройки';
      if (links.childNodes.length) links.append(' · ');
      links.append(link);
    }
    links.hidden = !links.childNodes.length;
  }).catch(error => status(error.message, true));
}

async function trackGeneration() {
  const button = byId('run').querySelector('button');
  button.disabled = true;
  status('Подгоняем параметры генератора…');
  try {
    const state = await waitFor(base, 'generating', progress => {
      if (progress.completed) status('Сгенерировано сцен: ' + progress.completed + ' / ' + progress.count);
    });
    showResult(state);
  } finally {
    button.disabled = false;
  }
}

byId('run').onsubmit = async event => {
  event.preventDefault();
  const button = event.submitter;
  button.disabled = true;
  busy(true);
  try {
    saveDraft();
    await request(base + 'generate', {
      crops: crops.filter(crop => crop.reviewed && !crop.holdout),
      heldout: crops.filter(crop => crop.holdout && crop.reviewed),
      count: Number(byId('count').value),
      seed: Number(byId('seed').value),
      grow_somas: byId('grow-somas').checked
    });
    byId('result').hidden = byId('comparison').hidden = byId('metrics').hidden = true;
    await trackGeneration();
  } catch (error) {
    status(error.message, true);
  } finally {
    button.disabled = false;
    busy(false);
  }
};

async function refreshReview() {
  clearTimeout(reviewPoll);
  const project = base;
  try {
    const [job, state] = await Promise.all([request(project + 'status'), request(project + 'refinement')]);
    if (base !== project) return;
    const working = state.stage === 'working';
    const hasRun = job.stage === 'done' || (job.stage === 'fit_error' && job.previous_attempt);
    const disabled = working || !hasRun;
    byId('train-review').disabled = byId('import-review').querySelector('button').disabled = disabled;
    byId('review-status').textContent = state.message && state.stage !== 'done' ? state.message
      : !hasRun ? 'Сначала завершите генерацию.' : '';
    const log = byId('training-log');
    log.hidden = !state.training;
    if (state.training) log.href = project + state.session + '/training.log';
    const session = state.stage === 'done' ? state.session : state.previous;
    const frame = byId('refinement').querySelector('iframe');
    frame.hidden = byId('review-help').hidden = !session;
    if (session && frame.getAttribute('src') !== project + session + '/') frame.src = project + session + '/';
    if (working) reviewPoll = setTimeout(refreshReview, 1200);
  } catch (error) {
    if (base === project) byId('review-status').textContent = error.message;
  }
}

async function startReview(route, body) {
  const project = base;
  byId('train-review').disabled = byId('import-review').querySelector('button').disabled = true;
  try {
    await request(project + route, body);
    if (base === project) await refreshReview();
  } catch (error) {
    if (base !== project) return;
    await refreshReview();
    if (base === project) byId('review-status').textContent = error.message;
  }
}

byId('train-review').onclick = () => startReview('train-review', {});
byId('import-review').onsubmit = event => {
  event.preventDefault();
  startReview('import-review', {path: byId('prediction-path').value});
};
window.addEventListener('message', async event => {
  const frame = byId('refinement').querySelector('iframe');
  if (event.origin !== location.origin || event.source !== frame.contentWindow ||
      event.data?.type !== 'bebrafish-refit') return;
  showStep('generate');
  saveDraft();
  busy(true);
  try { await trackGeneration(); }
  catch (error) { status(error.message, true); }
  finally { busy(false); }
});

for (const id of ['count', 'seed', 'grow-somas']) byId(id).onchange = saveDraft;

function updatePreviews() {
  const normalize = byId('p5-p95').checked;
  const suffix = normalize ? '-p5-p95' : '';
  for (const type of ['scene', 'crops']) {
    const preview = byId(type + '-preview');
    preview.src = previewBase + type + suffix + '.png';
  }
  byId('contrast-description').textContent = normalize
    ? 'Каждая картинка и каждый кроп нормируются отдельно по своим P5 и P95.'
    : 'Общая шкала яркости для всех изображений, от P1 до P99,5 реальной медианы.';
}

byId('p5-p95').onchange = () => { updatePreviews(); saveDraft(); };

for (const type of ['scene', 'crops']) {
  const preview = byId(type + '-preview');
  preview.onerror = () => {
    if (!byId('comparison').hidden) {
      status('Не удалось загрузить предпросмотр. Перезапустите сервер приложения и обновите страницу.', true);
    }
  };
}


async function init() {
  try {
    await listProjects();
    const token = location.hash.slice(1);
    if (/^[0-9a-f]{32}$/.test(token)) await openProject(token, new URLSearchParams(location.search).get('step'));
  } catch (error) {
    status(error.message, true);
  }
}
init();


byId('metrics').ontoggle = () => {
  const image = byId('metrics-image');
  if (byId('metrics').open && !image.hasAttribute('src')) {
    byId('metrics-status').textContent = 'Строим графики…';
    image.src = previewBase + 'metrics.png';
  }
};
byId('metrics-image').onload = () => { byId('metrics-status').textContent = ''; };
byId('metrics-image').onerror = () => {
  byId('metrics-status').textContent = 'Графики не загрузились. Закройте и откройте раздел ещё раз.';
  byId('metrics-image').removeAttribute('src');
};
