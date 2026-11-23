import test from 'node:test';
import assert from 'node:assert/strict';
import {makeGrid,cropAt,regrid,upgradeDocument} from '../region_sandbox/crops.mjs';
test('default crop partition covers each pixel once, including partial borders',()=>{
 const crops=makeGrid([386,320]),cover=new Uint8Array(386*320);
 for(const c of crops)for(let y=c.y;y<c.y+c.height;y++)for(let x=c.x;x<c.x+c.width;x++)cover[y*320+x]++;
 assert.ok(cover.every(v=>v===1));assert.equal(crops.length,6);
 assert.equal(cropAt(crops,319,385).id,'tile-384-192');assert.equal(crops.at(-1).height,2);assert.throws(()=>makeGrid([1376,320],[1,1]),/много/);
});
test('upgrade and regrid preserve thresholds and positions without mutating input',()=>{
 const old={format:'region-flood-v1',dataset:{shape_yx:[386,320]},points:[{id:'a',x:319,y:385,threshold:180.3,enabled:true}]};
 const next=upgradeDocument(old);assert.equal(next.format,'region-flood-v2');assert.equal(old.format,'region-flood-v1');assert.equal(old.points[0].crop_id,undefined);
 const other=regrid(next,[128,128]);assert.equal(other.points[0].threshold,180.3);assert.equal(other.points[0].x,319);assert.equal(other.points[0].y,385);assert.equal(other.points[0].crop_id,'tile-384-256');assert.equal(next.points[0].crop_id,'tile-384-192');assert.equal(upgradeDocument(next),next);
});
