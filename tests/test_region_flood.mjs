import test from 'node:test';
import assert from 'node:assert/strict';
import {flood,unionMasks} from '../region_sandbox/flood.mjs';
const p=(x,y,threshold,enabled=true)=>({x,y,threshold,enabled});
test('each point has its own absolute threshold and cannot drift across a gradient',()=>{
 const image=Float32Array.from([1,2,3,4,5,6]);
 const a=flood(image,6,1,p(0,0,2)),b=flood(image,6,1,p(4,0,5));
 assert.equal(a.count,2);assert.equal(b.count,5);
 assert.deepEqual([...a.mask],[1,1,0,0,0,0]);
 assert.deepEqual([...unionMasks([a.mask,b.mask],6)],[1,1,1,1,1,0]);
});
test('flood stops at bright barriers and disabled points contribute nothing',()=>{
 const image=Float32Array.from([1,1,9,1,1,1,1,9,1,1]);
 assert.equal(flood(image,5,2,p(0,0,2)).count,4);
 assert.equal(flood(image,5,2,p(4,0,2)).count,4);
 assert.equal(flood(image,5,2,p(0,0,2,false)).count,0);
 assert.equal(flood(image,5,2,p(0,0,.5)).count,0);
 assert.equal(flood(image,5,2,p(0,0,10)).count,10);
});
test('4 versus 8 connectivity and float32 ADU precision',()=>{
 const image=Float32Array.from([1,9,9,1]);
 assert.equal(flood(image,2,2,p(0,0,2),4).count,1);
 assert.equal(flood(image,2,2,p(0,0,2),8).count,2);
 assert.equal(flood(Float32Array.from([150.1,150.2]),2,1,p(0,0,150.1)).count,1);
});
test('a crop is a hard graph boundary, even when a dark path goes around it',()=>{
 const image=new Float32Array(25).fill(1);for(let y=1;y<=3;y++)image[y*5+2]=9;
 const crop={x:1,y:1,width:3,height:3};
 for(const connectivity of [4,8]){
  const fill=flood(image,5,5,p(1,2,2),connectivity,crop);
  assert.equal(fill.count,3);assert.equal(fill.mask[2*5+3],0);assert.equal(fill.mask[0],0);
  const all=flood(image,5,5,p(1,2,100),connectivity,crop);assert.equal(all.count,9);assert.equal(all.mask[0],0);
 }
});
