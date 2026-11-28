import assert from 'node:assert/strict';
import {test} from 'node:test';
import {nearestMaskOwners, selectMasks} from '../calibration_lab/segmentation.mjs';

const mask = (x, score, runs) => ({center: [x, 0], score, runs});
const options = {threshold: 0.3, radius: 5, maxCover: 0.85, bonus: 0.1, penalty: 0};
const shape = [50, 10];

test('nearby candidates need free area and cannot promote another candidate', () => {
  const anchor = mask(5, 0.5, [[4, 4]]);
  const duplicate = mask(6, 0.29, [[5, 3]]);
  const partial = mask(9, 0.25, [[6, 4]]);
  const chain = mask(13, 0.25, [[13, 4]]);
  const result = selectMasks([anchor, duplicate, partial, chain], shape, options);
  assert.deepEqual(result.original.map(row => row.candidate), [anchor]);
  assert.deepEqual(result.added.map(row => row.candidate), [partial]);
  assert.equal(result.added[0].coverage, 0.5);
});

test('coverage combines masks, including previously promoted candidates', () => {
  const a = mask(5, 0.5, [[0, 2]]);
  const b = mask(6, 0.5, [[2, 2]]);
  const covered = mask(7, 0.29, [[0, 4]]);
  const promoted = mask(7, 0.28, [[4, 3]]);
  const duplicate = mask(8, 0.27, [[3, 4]]);
  const result = selectMasks([a, b, covered, promoted, duplicate], shape, options);
  assert.deepEqual(result.added.map(row => row.candidate), [promoted]);
});

test('a removed isolated detection cannot provide a bonus', () => {
  const isolated = mask(5, 0.32, [[5, 1]]);
  const weak = mask(6, 0.29, [[6, 1]]);
  const result = selectMasks([isolated, weak], shape, {...options, penalty: 0.05});
  assert.deepEqual(result.removed.map(row => row.candidate), [isolated]);
  assert.deepEqual(result.added, []);
});


test('shared pixels go to the nearest covering center and ties stay stable', () => {
  const left = mask(1.5, 0.5, [[0, 5]]);
  const right = mask(5.5, 0.4, [[2, 4]]);
  const outside = mask(4.5, 0.7, [[6, 1]]);
  const rows = [left, right, outside].map(candidate => ({candidate}));
  assert.deepEqual([...nearestMaskOwners(rows, [8, 1])], [0, 0, 0, 0, 1, 1, 2, -1]);
});

test('a weak center inside an accepted mask cannot be promoted by an uncovered fringe', () => {
  const anchor = mask(5, 0.5, [[4, 4]]);
  const duplicate = mask(6, 0.29, [[6, 4]]);
  const result = selectMasks([anchor, duplicate], shape, options);
  assert.deepEqual(result.added, []);
});

test('overlap and diagonal contact protect against isolation penalty beyond the center radius', () => {
  const anchor = mask(4, 0.5, [[2, 8]]);
  const overlapping = mask(12, 0.32, [[9, 5]]);
  const diagonal = {...mask(15, 0.32, [[64, 2]]), center: [15, 1]};
  const isolated = mask(25, 0.32, [[25, 2]]);
  const result = selectMasks([anchor, overlapping, diagonal, isolated], shape,
    {...options, radius: 1, penalty: 0.05});
  assert.deepEqual(result.original.map(row => row.candidate), [anchor, overlapping, diagonal]);
  assert.deepEqual(result.removed.map(row => row.candidate), [isolated]);
});

test('contact checks do not wrap between rows', () => {
  const left = {...mask(0, 0.32, [[50, 1]]), center: [0, 1]};
  const right = mask(49, 0.5, [[49, 1]]);
  const result = selectMasks([left, right], shape, {...options, radius: 1, penalty: 0.05});
  assert.deepEqual(result.removed.map(row => row.candidate), [left]);
});
