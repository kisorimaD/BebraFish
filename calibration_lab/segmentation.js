import {nearestMaskOwners, selectMasks} from './segmentation.mjs';

const byId = id => document.getElementById(id);
const canvas = document.querySelector('canvas');
const fieldset = document.querySelector('fieldset');
const status = byId('status');
const controls = ['threshold', 'radius', 'bonus', 'penalty', 'maxCover', 'zoom'].map(byId);
const colors = {original: '#75edb1', added: '#ffce6c', removed: '#ffa0c3'};

function resolvedContours(groups, width, height) {
  const accepted = [...groups.original, ...groups.added];
  const owners = nearestMaskOwners(accepted, [width, height]);
  const paths = {original: new Path2D(), added: new Path2D()};
  for (const [index, {candidate}] of accepted.entries()) {
    const path = index < groups.original.length ? paths.original : paths.added;
    const edge = (x, y, endX, endY) => {
      path.moveTo(x, y);
      path.lineTo(endX, endY);
    };
    for (const [start, length] of candidate.runs) {
      const y = Math.floor(start / width);
      for (let pixel = start; pixel < start + length; pixel++) {
        if (owners[pixel] !== index) continue;
        const x = pixel % width;
        if (x === 0 || owners[pixel - 1] !== index) edge(x, y, x, y + 1);
        if (x === width - 1 || owners[pixel + 1] !== index) edge(x + 1, y, x + 1, y + 1);
        if (owners[pixel - width] !== index) edge(x, y, x + 1, y);
        if (owners[pixel + width] !== index) edge(x, y + 1, x + 1, y + 1);
      }
    }
  }
  return paths;
}

async function start() {
  const response = await fetch('/segmentation/candidates.json');
  if (!response.ok) throw new Error('Нет подготовленных кандидатов для просмотра.');
  const data = await response.json();
  const scene = new Image();
  scene.src = '/segmentation/scene.png';
  await scene.decode();
  const {width, height, candidates} = data;
  byId('source').textContent = `Чекпойнт: ${data.checkpoint}\nSHA-256: ${data.checkpoint_sha256}\nСцена: ${data.image}, страница ${data.page}\nПорог маски: ${data.mask_threshold}`
    + (data.denoiser ? `\nДенойзер: ${data.denoiser}` : '');
  for (const candidate of candidates) {
    const path = new Path2D();
    for (const loop of candidate.contours) {
      const [[x, y], ...points] = loop;
      path.moveTo(x, y);
      for (const [px, py] of points) path.lineTo(px, py);
      path.closePath();
    }
    candidate.path = path;
    delete candidate.contours;
  }

  const threshold = byId('threshold');
  threshold.min = data.candidate_floor;
  threshold.value = Math.max(data.candidate_floor, Number(threshold.defaultValue));
  const defaults = controls.map(control => control.value);
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  let scheduled = false;

  function draw() {
    scheduled = false;
    const settings = {};
    for (const control of controls) {
      settings[control.id] = control.valueAsNumber;
      control.previousElementSibling.value = control.id === 'maxCover'
        ? `${Math.round(control.valueAsNumber * 100)}%` : control.value;
    }
    const groups = selectMasks(candidates, [width, height], settings);
    const contours = resolvedContours(groups, width, height);
    canvas.style.width = `${width * settings.zoom}px`;
    ctx.drawImage(scene, 0, 0);
    ctx.lineWidth = Math.max(0.5, 1 / settings.zoom);
    for (const [name, color] of Object.entries(colors)) {
      const toggle = byId(name);
      toggle.parentElement.querySelector('span').textContent = groups[name].length;
      if (!toggle.checked) continue;
      ctx.strokeStyle = color;
      ctx.setLineDash(name === 'removed' ? [2, 2] : []);
      if (name === 'removed') {
        for (const {candidate} of groups.removed) ctx.stroke(candidate.path);
      } else {
        ctx.stroke(contours[name]);
      }
    }
  }

  function scheduleDraw() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(draw);
  }

  fieldset.addEventListener('input', scheduleDraw);
  byId('reset').addEventListener('click', () => {
    controls.forEach((control, i) => { control.value = defaults[i]; });
    for (const name of Object.keys(colors)) byId(name).checked = byId(name).defaultChecked;
    scheduleDraw();
  });
  fieldset.disabled = false;
  canvas.hidden = false;
  status.textContent = `${candidates.length.toLocaleString('ru')} кандидатов от порога ${data.candidate_floor}`;
  draw();
}

start().catch(error => { status.textContent = error.message; });
