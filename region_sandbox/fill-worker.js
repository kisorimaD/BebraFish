import {flood,unionMasks} from './flood.mjs';
let image,width,height,cache=new Map();
self.onmessage=({data})=>{try{
 if(data.type==='init'){image=new Float32Array(data.image);width=data.width;height=data.height;cache.clear();return}
 const present=new Set(data.points.map(p=>p.id));for(const id of cache.keys())if(!present.has(id))cache.delete(id);
 const counts={},masks=[],cropMap=new Map((data.crops||[]).map(c=>[c.id,c]));
 for(const point of data.points){const bounds=data.crops?cropMap.get(point.crop_id):null;if(data.crops&&!bounds)throw Error('Не найден кроп точки');const key=JSON.stringify([point.x,point.y,point.threshold,point.enabled,data.connectivity,bounds,point.polygon]);let entry=cache.get(point.id);
  if(!entry||entry.key!==key){entry={key,...flood(image,width,height,point,data.connectivity,bounds)};cache.set(point.id,entry)}
  counts[point.id]={count:entry.count,touchesEdge:entry.touchesEdge};if(point.enabled)masks.push(entry.mask);
 }
 const union=unionMasks(masks,image.length);let count=0;for(const value of union)count+=value;
 const selected=cache.get(data.selected)?.mask?.slice()||new Uint8Array(image.length);
 self.postMessage({revision:data.revision,selectedId:data.selected,mask:union.buffer,selectedMask:selected.buffer,counts,count},[union.buffer,selected.buffer]);
 }catch(e){self.postMessage({revision:data.revision,error:e.message})}};
