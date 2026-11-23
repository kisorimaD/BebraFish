import test from 'node:test';
import assert from 'node:assert/strict';
import {measure, validPolygon, contains} from '../annotation_sandbox/geometry.mjs';
const close=(a,b,tol=1e-8)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('filled rectangle uses area moments, independent of contour direction',()=>{
 const p=[[10,20],[20,20],[20,24],[10,24]];
 for(const points of [p,[...p].reverse()]){
  const m=measure(points);close(m.area_px2,40);close(m.centroid_x,15);close(m.centroid_y,22);close(m.perimeter_px,28);
  close(m.major_axis_px,40/Math.sqrt(12));close(m.minor_axis_px,16/Math.sqrt(12));close(m.aspect_ratio,2.5);
 }
});
test('circle and rotated ellipse recover diameters and orientation',()=>{
 const angle=.37,points=Array.from({length:360},(_,i)=>{const t=2*Math.PI*i/360,x=8*Math.cos(t),y=3*Math.sin(t);return[120+x*Math.cos(angle)-y*Math.sin(angle),900+x*Math.sin(angle)+y*Math.cos(angle)]});
 const m=measure(points);close(m.major_axis_px,16,.002);close(m.minor_axis_px,6,.002);close(m.orientation_deg,angle*180/Math.PI,1e-5);close(m.area_px2,Math.PI*24,.01);
});
test('reject self-crossings, degeneracy and nonfinite vertices',()=>{
 assert.equal(validPolygon([[0,0],[10,0],[10,10],[0,10]]),true);
 for(const p of [[[0,0],[10,10],[0,10],[10,0]],[[0,0],[1,0],[2,0]],[[0,0],[NaN,2],[4,4]],[[0,0],[10,0],[10,10],[0,10],[8,-1]]])assert.equal(validPolygon(p),false);
 assert.equal(contains([[0,0],[10,0],[10,10],[0,10]],[5,5]),true);
 assert.equal(contains([[0,0],[10,0],[10,10],[0,10]],[15,5]),false);
});
