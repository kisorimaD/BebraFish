const byId = id => document.getElementById(id);
const endpoint = location.pathname + '/data';
const controls = new Map();
const colors = ['#77dded', '#cabffa', '#ffb38a'];
const storageKey = 'bebrafish-light:' + location.pathname;
let scene, crop, pending = false, revision = 0, timer;

async function request(body) {
  const response = await fetch(endpoint, body ? {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body),
  } : {});
  if (!response.ok && !response.headers.get('Content-Type')?.includes('json')) {
    throw new Error('Сцена недоступна.');
  }
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Ошибка предпросмотра.');
  return result;
}

function status(message, error = false) {
  const node = byId('status');
  node.textContent = message;
  node.classList.toggle('error', error);
}

function params() {
  return Object.fromEntries([...controls].map(([key, {value}]) => [key, value.valueAsNumber]));
}

function preset() {
  return {crop, settings: params(), noise: byId('noise').checked};
}

function setCrop(y, x, size) {
  const [height, width] = scene.shape;
  const h = Math.min(height, size);
  const w = Math.min(width, size);
  y = Math.max(0, Math.min(height - h, Math.round(y)));
  x = Math.max(0, Math.min(width - w, Math.round(x)));
  crop = [y, x, h, w];
  byId('y').value = y;
  byId('x').value = x;
  Object.assign(byId('rectangle').style, {
    top: 100*y/height + '%', left: 100*x/width + '%',
    height: 100*h/height + '%', width: 100*w/width + '%',
  });
}

function selectCrop() {
  const saved = scene.crops[Number(byId('crop').value)];
  const size = Number(byId('size').value);
  if (saved) setCrop(saved.y + (saved.height-size)/2, saved.x + (saved.width-size)/2, size);
  else setCrop((scene.shape[0]-size)/2, (scene.shape[1]-size)/2, size);
}

function buildControls() {
  for (const item of scene.controls) {
    const container = document.createElement('div');
    container.className = 'setting';
    const label = document.createElement('label');
    label.htmlFor = 'slider-' + item.name;
    label.textContent = item.label + (item.unit ? ' · ' + item.unit : '');
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.id = label.htmlFor;
    const value = document.createElement('input');
    value.type = 'number';
    value.setAttribute('aria-label', item.label);
    for (const element of [slider, value]) {
      element.min = Math.min(item.min, scene.defaults[item.name]);
      element.max = Math.max(item.max, scene.defaults[item.name]);
    }
    slider.step = item.step;
    value.step = 'any';
    slider.value = value.value = scene.defaults[item.name];
    slider.oninput = () => {
      value.value = slider.value;
      queueRender();
    };
    value.onchange = () => {
      value.value = Math.min(Number(value.max), Math.max(Number(value.min),
        Number.isFinite(value.valueAsNumber) ? value.valueAsNumber : scene.defaults[item.name]));
      slider.value = value.value;
      queueRender();
    };
    controls.set(item.name, {slider, value});
    container.append(label, slider, value);
    byId('controls').append(container);
  }
}

function apply(settings) {
  for (const [key, {slider, value}] of controls) {
    const number = settings[key];
    const use = Number.isFinite(number) && number >= Number(value.min) && number <= Number(value.max);
    slider.value = value.value = use ? number : scene.defaults[key];
  }
}

function metrics(stats) {
  const rows = [
    ['Сомы с полным профилем', 'n', 0],
    ['Пик оболочки минус центр, ADU', 'shell_contrast', 1],
    ['Ошибка радиального профиля, ADU', 'profile_error', 1],
    ['Контраст кропа P95−P5, ADU', 'contrast', 1],
    ['Автокорреляция на сдвиге 4 px', 'acf4', 3],
  ];
  byId('metrics').replaceChildren(...rows.map(([label, key, digits]) => {
    const row = document.createElement('tr');
    const name = document.createElement('th');
    name.scope = 'row';
    name.textContent = label;
    row.append(name);
    for (const result of stats) {
      const value = document.createElement('td');
      value.textContent = result[key] === null ? '—' : result[key].toLocaleString('ru', {maximumFractionDigits: digits});
      row.append(value);
    }
    return row;
  }));
}

function profile(stats) {
  const canvas = byId('chart');
  const c = canvas.getContext('2d');
  const {width, height} = canvas;
  c.clearRect(0, 0, width, height);
  const all = stats.flatMap(row => row.profile?.adu ?? []);
  byId('profile-note').textContent = stats[0].n ? 'Медиана только по сомам, чей внешний профиль помещается в кроп.'
    : 'В кропе нет полных размеченных сом для профиля референса.';
  if (!all.length) return;
  const min = Math.min(0, ...all);
  const max = Math.max(1, ...all);
  const bounds = {left: 55, right: width-20, top: 18, bottom: height-42};
  const x = radius => bounds.left + radius/1.8*(bounds.right-bounds.left);
  const y = adu => bounds.bottom - (adu-min)/(max-min)*(bounds.bottom-bounds.top);
  c.font = '14px system-ui';
  c.fillStyle = '#bfd4d7';
  c.strokeStyle = '#496069';
  c.lineWidth = 1;
  for (let index=0; index<=4; index++) {
    const tick = min+(max-min)*index/4;
    c.beginPath();
    c.moveTo(bounds.left, y(tick));
    c.lineTo(bounds.right, y(tick));
    c.stroke();
    c.fillText(tick.toFixed(0), 10, y(tick)+5);
  }
  for (const tick of [0, .5, 1, 1.5, 1.8]) {
    c.fillText(tick.toLocaleString('ru'), x(tick)-5, bounds.bottom+18);
  }
  c.fillText('ADU', 10, 12);
  c.fillText('Доли радиуса маски', width/2-56, height-5);
  stats.forEach(({profile}, index) => {
    if (!profile) return;
    c.beginPath();
    c.strokeStyle = colors[index];
    c.lineWidth = 2.5;
    if (index === 1) c.setLineDash([6, 5]);
    profile.radius.forEach((r, i) => {
      if (i === 0) c.moveTo(x(r), y(profile.adu[i]));
      else c.lineTo(x(r), y(profile.adu[i]));
    });
    c.stroke();
    c.setLineDash([]);
  });
}

function queueRender() {
  revision++;
  clearTimeout(timer);
  timer = setTimeout(render, 150);
}

async function render() {
  if (pending) return;
  const current = revision;
  const state = preset();
  pending = true;
  status('Обновляем кроп…');
  try {
    const result = await request(state);
    if (current !== revision) return;
    ['real', 'baseline', 'changed'].forEach((id, i) => { byId(id).src = result.images[i]; });
    metrics(result.stats);
    profile(result.stats);
    try { localStorage.setItem(storageKey, JSON.stringify(state)); } catch {}
    status('');
  } catch (error) {
    if (current === revision) status(error.message, true);
  } finally {
    pending = false;
    if (current !== revision) {
      clearTimeout(timer);
      timer = setTimeout(render, 0);
    }
  }
}

async function init() {
  scene = await request();
  buildControls();
  byId('overview').src = scene.overview;
  byId('noise').checked = scene.noise;
  byId('noise-label').hidden = !scene.noise;
  const dropdown = byId('crop');
  scene.crops.forEach((item, i) => dropdown.add(new Option(
    (item.label || 'Калибровка') + ' · X ' + item.x + ', Y ' + item.y, String(i))));
  dropdown.add(new Option('Произвольный', 'custom'));
  dropdown.value = scene.crops.length ? String(scene.default_crop ?? 0) : 'custom';
  selectCrop();
  let saved;
  try { saved = JSON.parse(localStorage.getItem(storageKey)); } catch {}
  if (saved && Array.isArray(saved.crop) && saved.crop.length === 4) {
    apply(saved.settings || {});
    if (scene.noise) byId('noise').checked = Boolean(saved.noise);
    const [y, x, h, w] = saved.crop;
    const size = Math.max(h, w);
    if ([64, 96, 128, 192].includes(size)) byId('size').value = size;
    setCrop(y, x, Number(byId('size').value));
    dropdown.value = 'custom';
  }
  dropdown.onchange = () => {
    if (dropdown.value !== 'custom') selectCrop();
    queueRender();
  };
  byId('map').onclick = event => {
    const bounds = byId('overview').getBoundingClientRect();
    const size = Number(byId('size').value);
    dropdown.value = 'custom';
    setCrop((event.clientY-bounds.top)*scene.shape[0]/bounds.height-size/2,
            (event.clientX-bounds.left)*scene.shape[1]/bounds.width-size/2, size);
    queueRender();
  };
  for (const id of ['x','y','size']) byId(id).onchange = () => {
    dropdown.value = 'custom';
    setCrop(Number(byId('y').value), Number(byId('x').value), Number(byId('size').value));
    queueRender();
  };
  byId('noise').onchange = queueRender;
  byId('reset').onclick = () => {
    apply(scene.defaults);
    byId('noise').checked = scene.noise;
    queueRender();
  };
  const applyButton = byId('apply');
  applyButton.hidden = !scene.can_apply;
  applyButton.onclick = async () => {
    applyButton.disabled = true;
    status('Запускаем новый прогон…');
    try {
      const result = await request({apply: true, settings: params()});
      location.assign('/?step=generate#' + result.project);
    } catch (error) {
      status(error.message, true);
      applyButton.disabled = false;
    }
  };
  byId('save').onclick = () => {
    const blob = new Blob([JSON.stringify({seed: scene.seed, ...preset()}, null, 2) + '\n'],
                          {type:'application/json'});
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'soma-light-settings.json';
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  byId('workspace').hidden = false;
  await render();
}

init().catch(error => status(error.message, true));
