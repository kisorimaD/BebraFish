import test from 'node:test';
import assert from 'node:assert/strict';
const messages=[];
globalThis.self={postMessage(data){messages.push(data)}};
await import('../region_sandbox/fill-worker.js');
test('worker invalidates only changed seed cache and returns current union/highlight',()=>{
 self.onmessage({data:{type:'init',image:Float32Array.from([1,2,9,3,4]).buffer,width:5,height:1}});
 const points=[{id:'a',x:0,y:0,threshold:2,enabled:true},{id:'b',x:4,y:0,threshold:4,enabled:true}];
 self.onmessage({data:{revision:1,points,selected:'a',connectivity:4}});
 let r=messages.at(-1);assert.equal(r.count,4);assert.equal(r.counts.a.count,2);assert.equal(r.counts.b.count,2);
 assert.deepEqual([...new Uint8Array(r.selectedMask)],[1,1,0,0,0]);
 points[0].threshold=1;
 self.onmessage({data:{revision:2,points,selected:'b',connectivity:4}});
 r=messages.at(-1);assert.equal(r.revision,2);assert.equal(r.count,3);assert.equal(r.counts.a.count,1);assert.equal(r.counts.b.count,2);
 assert.deepEqual([...new Uint8Array(r.selectedMask)],[0,0,0,1,1]);
 points.splice(0,1);points[0].enabled=false;
 self.onmessage({data:{revision:3,points,selected:'b',connectivity:4}});
 r=messages.at(-1);assert.equal(r.count,0);assert.equal('a' in r.counts,false);assert.equal(r.counts.b.count,0);
});
test('worker cache is invalidated when crop bounds change',()=>{
 self.onmessage({data:{type:'init',image:new Float32Array(25).fill(1).buffer,width:5,height:5}});
 const points=[{id:'a',x:1,y:1,threshold:100,enabled:true,crop_id:'c'}];
 self.onmessage({data:{revision:10,points,selected:'a',connectivity:8,crops:[{id:'c',x:0,y:0,width:2,height:2}]}});
 assert.equal(messages.at(-1).count,4);
 self.onmessage({data:{revision:11,points,selected:'a',connectivity:8,crops:[{id:'c',x:1,y:1,width:3,height:3}]}});
 assert.equal(messages.at(-1).count,9);assert.equal(new Uint8Array(messages.at(-1).mask)[0],0);
});

test('polygon edits invalidate cache and restrict only their own point',()=>{
 self.onmessage({data:{type:'init',image:new Float32Array(49).fill(1).buffer,width:7,height:7}});
 const points=[{id:'a',x:1,y:1,threshold:100,enabled:true,polygon:[[1,1],[3,1],[3,3],[1,3]]}];
 const run=()=>{self.onmessage({data:{revision:20,points,selected:'a',connectivity:8}});return messages.at(-1)};
 assert.equal(run().count,9);
 points[0].polygon=[[0,0],[2,0],[2,2],[0,2]];
 assert.equal(run().count,9);assert.equal(new Uint8Array(messages.at(-1).mask)[0],1);
 points.push({id:'b',x:6,y:6,threshold:100,enabled:true});assert.equal(run().count,49);
 points.pop();delete points[0].polygon;assert.equal(run().count,49);
});
