function maskContacts(candidates, width, height) {
  const labels = new Int32Array(width * height);
  for (const [index, candidate] of candidates.entries()) {
    for (const [start, length] of candidate.runs) {
      for (let pixel = start; pixel < start + length; pixel++) {
        labels[pixel] = labels[pixel] === 0 ? index + 1 : -1;
      }
    }
  }

  return (candidate, index) => {
    const own = index + 1;
    for (const [start, length] of candidate.runs) {
      const y = Math.floor(start / width);
      const x = start % width;
      const left = Math.max(0, x - 1);
      const right = Math.min(width, x + length + 1);
      for (let row = Math.max(0, y - 1); row < Math.min(height, y + 2); row++) {
        for (let pixel = row * width + left; pixel < row * width + right; pixel++) {
          if (labels[pixel] !== 0 && labels[pixel] !== own) return true;
        }
      }
    }
    return false;
  };
}

export function selectMasks(candidates, shape, {threshold, radius, bonus, penalty, maxCover}) {
  const [width, height] = shape;
  const cutoff = Math.fround(threshold);
  const baseline = candidates.filter(candidate => candidate.score >= cutoff);
  const bins = new Map();
  const dropped = new Set();
  const key = (x, y) => `${x},${y}`;
  if (radius > 0 && (bonus > 0 || penalty > 0)) {
    for (const candidate of baseline) {
      const [x, y] = candidate.center.map(value => Math.floor(value / radius));
      const cell = key(x, y);
      if (!bins.has(cell)) bins.set(cell, []);
      bins.get(cell).push(candidate);
    }
  }

  function neighborCount(candidate) {
    if (!bins.size) return 0;
    const [x, y] = candidate.center;
    const col = Math.floor(x / radius);
    const row = Math.floor(y / radius);
    let count = 0;
    for (let j = row - 1; j <= row + 1; j++) {
      for (let i = col - 1; i <= col + 1; i++) {
        for (const neighbor of bins.get(key(i, j)) ?? []) {
          const [nx, ny] = neighbor.center;
          if (neighbor !== candidate && !dropped.has(neighbor) &&
              (nx - x) ** 2 + (ny - y) ** 2 <= radius ** 2) {
            count++;
            if (count === 2) return count;
          }
        }
      }
    }
    return count;
  }

  const original = [];
  const added = [];
  const removed = [];
  const touchesAnother = penalty > 0 ? maskContacts(baseline, width, height) : null;
  for (const [index, candidate] of baseline.entries()) {
    const isolated = touchesAnother && !neighborCount(candidate) && !touchesAnother(candidate, index);
    const adjusted = candidate.score - (isolated ? penalty : 0);
    (adjusted >= cutoff ? original : removed).push({candidate, adjusted});
  }
  if (!bonus || !bins.size) return {original, added, removed};
  for (const {candidate} of removed) dropped.add(candidate);

  const occupied = new Uint8Array(width * height);
  const mark = candidate => {
    for (const [start, length] of candidate.runs) occupied.fill(1, start, start + length);
  };
  for (const {candidate} of original) mark(candidate);
  const borderline = [];
  for (const candidate of candidates) {
    if (candidate.score >= cutoff || candidate.score + 2 * bonus < cutoff) continue;
    const adjusted = candidate.score + bonus * neighborCount(candidate);
    if (adjusted >= cutoff) borderline.push({candidate, adjusted});
  }
  borderline.sort((a, b) => b.adjusted - a.adjusted || b.candidate.score - a.candidate.score);
  for (const result of borderline) {
    const [x, y] = result.candidate.center.map(Math.floor);
    if (occupied[y * width + x]) continue;
    let covered = 0;
    let area = 0;
    for (const [start, length] of result.candidate.runs) {
      area += length;
      for (let i = start; i < start + length; i++) covered += occupied[i];
    }
    if (area && covered / area < maxCover) {
      added.push({...result, coverage: covered / area});
      mark(result.candidate);
    }
  }
  return {original, added, removed};
}

// Only a mask covering the pixel can claim it. On ties the first mask wins.
export function nearestMaskOwners(masks, [width, height]) {
  const owners = new Int32Array(width * height).fill(-1);
  for (const [index, {candidate}] of masks.entries()) {
    const [cx, cy] = candidate.center;
    for (const [start, length] of candidate.runs) {
      const y = Math.floor(start / width) + 0.5;
      const dy = (y - cy) ** 2;
      for (let pixel = start; pixel < start + length; pixel++) {
        const previous = owners[pixel];
        if (previous >= 0) {
          const x = pixel % width + 0.5;
          const [px, py] = masks[previous].candidate.center;
          if ((x - cx) ** 2 + dy >= (x - px) ** 2 + (y - py) ** 2) continue;
        }
        owners[pixel] = index;
      }
    }
  }
  return owners;
}
