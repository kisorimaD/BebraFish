export function makeGrid(shape,size=[192,192]){
 const[h,w]=shape,[cw,ch]=size;if(![cw,ch].every(n=>Number.isInteger(n)&&n>0&&n<=16384))throw Error('Размеры кропа должны быть целыми от 1 до 16384');
 if(Math.ceil(h/ch)*Math.ceil(w/cw)>10000)throw Error('Слишком много кропов');
 const crops=[];for(let y=0;y<h;y+=ch)for(let x=0;x<w;x+=cw)crops.push({id:`tile-${y}-${x}`,x,y,width:Math.min(cw,w-x),height:Math.min(ch,h-y),name:`Кроп ${String(crops.length+1).padStart(2,'0')}`});
 if(crops.length>10000)throw Error('Слишком много кропов');return crops;
}
export function cropAt(crops,x,y){return crops.find(c=>x>=c.x&&x<c.x+c.width&&y>=c.y&&y<c.y+c.height)}
export function regrid(doc,size){const crops=makeGrid(doc.dataset.shape_yx,size);return{...doc,format:'region-flood-v2',crop_size_xy:[...size],crops,points:doc.points.map(p=>{const c=cropAt(crops,p.x,p.y);if(!c)throw Error('Точка вне изображения');return{...p,crop_id:c.id}})}}
export function upgradeDocument(doc){return doc.format==='region-flood-v1'?regrid(doc,[192,192]):doc}
