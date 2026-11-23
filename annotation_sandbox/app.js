import {validPolygon, contains, measure} from './geometry.mjs';
const $=id=>document.getElementById(id), canvas=$('canvas'), ctx=canvas.getContext('2d'), overview=$('overview'), ov=overview.getContext('2d');
let meta,raw,doc,current,selected=null,tool='freehand',draft=[],showMasks=true,space=false,gesture=null;
let zoom=1,pan=[0,0],W=0,H=0,undoStack=[],redoStack=[],saveTimer,saving=false,pending=false,revision=0,storageKey;
let black=90,white=400,fullCanvas=document.createElement('canvas'),cropCanvas=document.createElement('canvas');
const colors=['#b8edc9','#94caf5','#e8b1fb','#ffd59a','#a7ebe8'];
function status(t){$('status').textContent=t}
function error(t){$('error').textContent=t;$('error').hidden=!t}
function crop(){return doc.crops.find(c=>c.id===current)}
function cell(){return doc.annotations.find(a=>a.id===selected)}
function clone(x){return JSON.parse(JSON.stringify(x))}
function checkpoint(){undoStack.push(JSON.stringify(doc));if(undoStack.length>70)undoStack.shift();redoStack=[]}
function changed(){doc.updated_at=new Date().toISOString();revision++;try{localStorage.setItem(storageKey,JSON.stringify(doc))}catch{error('Копия в браузере не сохранена. Проверьте автосохранение на диск или экспортируйте JSON.')}pending=true;$('saveState').textContent='Сохранение…';clearTimeout(saveTimer);saveTimer=setTimeout(save,350);renderUI()}
async function save(){
  if(saving)return;
  saving=true;pending=false;const sent=revision;
  try{
    const res=await fetch('/api/annotations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(exportDoc())});
    if(!res.ok)throw Error(await res.text());const result=await res.json();
    if(revision===sent)$('saveState').textContent='✓ Сохранено на диск';
    $('saveState').title=result.path;error('');
  }catch(e){$('saveState').textContent='⚠ Только в браузере';error('Автосохранение на диск недоступно: '+e.message+'. Экспортируйте JSON.');}
  finally{saving=false;if(pending||revision!==sent)saveTimer=setTimeout(save,350)}
}
function exportDoc(){return {...doc,exported_at:new Date().toISOString(),coordinate_system:'TIFF pixel-edge coordinates; points_xy=[x,y]; native shape_yx',annotations:doc.annotations.map(a=>({...a,measurements:measure(a.points_xy)}))}}
function download(name,text,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000)}
function inCrop(a){const c=crop();return a.points_xy.some(([x,y])=>x>=c.x&&x<=c.x+c.width&&y>=c.y&&y<=c.y+c.height)||contains(a.points_xy,[c.x+c.width/2,c.y+c.height/2])}
function visible(){return doc.annotations.filter(inCrop)}
function isReliable(a){return a.confidence==='confident'&&!a.truncated}
function transform(){const c=crop();const scale=Math.min(W/c.height,H/c.width)*zoom;return {scale,ox:(W-c.height*scale)/2+pan[0],oy:(H-c.width*scale)/2+pan[1]}}
function screen([x,y]){const c=crop(),{scale,ox,oy}=transform();return [(y-c.y)*scale+ox,(x-c.x)*scale+oy]}
function point(e,clamp=true){const rect=canvas.getBoundingClientRect(),sx=e.clientX-rect.left,sy=e.clientY-rect.top;const c=crop(),{scale,ox,oy}=transform();let x=(sy-oy)/scale+c.x,y=(sx-ox)/scale+c.y;
  if(clamp){x=Math.max(c.x,Math.min(c.x+c.width,x));y=Math.max(c.y,Math.min(c.y+c.height,y))}return [x,y]}
function onImage(e){const p=point(e,false),c=crop();return p[0]>=c.x&&p[0]<=c.x+c.width&&p[1]>=c.y&&p[1]<=c.y+c.height}
function setTool(t){if(draft.length){status('Сначала замкните контур или нажмите Esc.');return}tool=t;['freehand','polygon','select','move'].forEach(k=>$(k).classList.toggle('active',k===t));$('stageHint').textContent={freehand:'Обведите всю сому с зажатой кнопкой мыши',polygon:'Точки по клику · Enter — замкнуть · Esc — отменить',select:'Выберите сому · тяните вершины · Shift + клик по ребру добавляет вершину',move:'Перетаскивайте изображение мышью по X/Y · колесо меняет масштаб'}[t];draw()}
function switchCrop(id){if(draft.length){status('Незавершённый контур отменён при смене кропа.');draft=[]}current=id;selected=null;zoom=1;pan=[0,0];renderCrop();renderUI()}
function imageData(c){const data=new ImageData(c.height,c.width);for(let v=0;v<c.width;v++)for(let u=0;u<c.height;u++){
 const intensity=raw[(c.y+u)*meta.shape_yx[1]+c.x+v],g=Math.max(0,Math.min(255,Math.round(255*(intensity-black)/(white-black))));const i=(v*c.height+u)*4;data.data[i]=g;data.data[i+1]=g;data.data[i+2]=g;data.data[i+3]=255;
}return data}
function renderCrop(){const c=crop();cropCanvas.width=c.height;cropCanvas.height=c.width;cropCanvas.getContext('2d').putImageData(imageData(c),0,0);draw()}
function renderAllImages(){fullCanvas.width=meta.shape_yx[0];fullCanvas.height=meta.shape_yx[1];fullCanvas.getContext('2d').putImageData(imageData({x:0,y:0,width:meta.shape_yx[1],height:meta.shape_yx[0]}),0,0);renderCrop();renderThumbnails();drawOverview()}
function path(points,close=true){ctx.beginPath();points.forEach((p,i)=>{const [x,y]=screen(p);i?ctx.lineTo(x,y):ctx.moveTo(x,y)});if(close)ctx.closePath()}
function draw(){canvas.style.cursor=gesture?.kind==='pan'?'grabbing':(tool==='move'||space)?'grab':tool==='select'?'default':'crosshair';if(!doc)return;ctx.clearRect(0,0,W,H);const c=crop(),{scale,ox,oy}=transform();ctx.imageSmoothingEnabled=false;ctx.drawImage(cropCanvas,ox,oy,c.height*scale,c.width*scale);
 ctx.save();ctx.beginPath();ctx.rect(ox,oy,c.height*scale,c.width*scale);ctx.clip();
 if(showMasks)for(const a of visible()){
  const color=a.id===selected?'#ffffff':a.confidence==='uncertain'?'#f2c17d':colors[(a.number-1)%colors.length];path(a.points_xy);ctx.fillStyle=color+(a.id===selected?'22':'14');ctx.fill();ctx.strokeStyle=color;ctx.lineWidth=a.id===selected?1.7:1;ctx.stroke();
  const m=measure(a.points_xy),[x,y]=screen([m.centroid_x,m.centroid_y]);ctx.font='11px ui-monospace';ctx.fillStyle=color;ctx.fillText(a.number,x+3,y-3);
  if(a.id===selected&&tool==='select')for(const p of a.points_xy){const [vx,vy]=screen(p);ctx.beginPath();ctx.arc(vx,vy,3.5,0,2*Math.PI);ctx.fillStyle='#101916';ctx.fill();ctx.strokeStyle='#b8edc9';ctx.stroke()}
 }
 if(draft.length){path(draft,false);ctx.strokeStyle='#f2c17d';ctx.lineWidth=2;ctx.stroke();for(const p of [draft[0],draft[draft.length-1]]){const[x,y]=screen(p);ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fillStyle='#f2c17d';ctx.fill()}}
 ctx.restore();$('zoomLabel').textContent=Math.round(zoom*100)+'%';$('scale').style.width=(10*scale)+'px';$('scale').textContent='10 px';$('finish').disabled=draft.length<3;$('undo').disabled=!undoStack.length&&!draft.length;$('redo').disabled=!redoStack.length;
}
function drawOverview(){if(!doc)return;overview.width=fullCanvas.width;overview.height=fullCanvas.height;ov.drawImage(fullCanvas,0,0);const c=crop();ov.strokeStyle='#b8edc9';ov.lineWidth=3;ov.strokeRect(c.y,c.x,c.height,c.width);
 if(showMasks){ov.lineWidth=.8;ov.strokeStyle='#b8edc999';for(const a of doc.annotations){ov.beginPath();a.points_xy.forEach(([x,y],i)=>i?ov.lineTo(y,x):ov.moveTo(y,x));ov.closePath();ov.stroke()}}
}
function renderThumbnails(){const list=$('cropList');list.replaceChildren();for(const c of doc.crops){const b=document.createElement('button');b.className='crop-item'+(c.id===current?' active':'');b.dataset.id=c.id;b.title=`X ${c.x}–${c.x+c.width}, Y ${c.y}–${c.y+c.height}`;const th=document.createElement('canvas');th.width=80;th.height=80;const temp=document.createElement('canvas');temp.width=c.height;temp.height=c.width;temp.getContext('2d').putImageData(imageData(c),0,0);th.getContext('2d').drawImage(temp,0,0,80,80);const label=document.createElement('span');label.textContent=c.name;b.append(th,label);b.onclick=()=>switchCrop(c.id);list.append(b)}}
function renderUI(){const c=crop();$('cropTitle').textContent=c.name;$('cropCoords').textContent=`X ${c.x}–${c.x+c.width} · Y ${c.y}–${c.y+c.height}`;$('total').textContent=doc.annotations.length;
 const good=doc.annotations.filter(isReliable);$('usable').textContent=good.length;const ds=good.map(a=>measure(a.points_xy).equivalent_diameter_px).sort((a,b)=>a-b);$('median').textContent=ds.length?(ds[Math.floor((ds.length-1)/2)]/2+ds[Math.ceil((ds.length-1)/2)]/2).toFixed(1):'—';
 document.querySelectorAll('.crop-item').forEach(b=>b.classList.toggle('active',b.dataset.id===current));renderSelection();const list=$('cellList');list.replaceChildren();const cells=visible();$('cropCount').textContent=cells.length;
 if(!cells.length){const p=document.createElement('p');p.className='muted';p.textContent='Пока без контуров. Обведите первую сому.';list.append(p)}
 for(const a of cells){const b=document.createElement('button');b.className='cell-item'+(selected===a.id?' active':'')+(a.confidence==='uncertain'?' uncertain':'');const t=document.createElement('span');t.textContent=`#${String(a.number).padStart(3,'0')} ${a.truncated?'◩':''}${a.confidence==='uncertain'?' ?':''}`;const m=document.createElement('span');m.textContent='Ø '+measure(a.points_xy).equivalent_diameter_px.toFixed(1)+' px';b.append(t,m);b.onclick=()=>{if(draft.length)return status('Завершите текущий контур.');selected=a.id;setTool('select');renderUI()};list.append(b)}draw();drawOverview()
}
function renderSelection(){const a=cell(),box=$('selection');if(!a){box.innerHTML='<p class="muted">После обводки здесь появятся размеры сомы.</p>';return}
 const m=measure(a.points_xy);box.innerHTML=`<h3>Сома #${String(a.number).padStart(3,'0')}</h3><div class="measure-grid"><div><b>${m.area_px2.toFixed(1)}</b><span>площадь, px²</span></div><div><b>${m.equivalent_diameter_px.toFixed(1)}</b><span>экв. диаметр, px</span></div><div><b>${m.major_axis_px.toFixed(1)} × ${m.minor_axis_px.toFixed(1)}</b><span>оси эллипса, px</span></div><div><b>${m.aspect_ratio?.toFixed(2)||'—'}</b><span>отношение осей</span></div></div><label>Уверенность<select id="confidence"><option value="confident">Уверенная граница</option><option value="uncertain">Сомнительная граница</option></select></label><label><input type="checkbox" id="truncated"> Неполная сома / обрезана</label><label>Комментарий<textarea id="comment" placeholder="Например, контакт или слабая граница"></textarea></label><button id="delete" class="danger">Удалить сому</button>`;
 $('confidence').value=a.confidence;$('truncated').checked=a.truncated;$('comment').value=a.comment||'';
 $('confidence').onchange=()=>{checkpoint();a.confidence=$('confidence').value;changed()};$('truncated').onchange=()=>{checkpoint();a.truncated=$('truncated').checked;changed()};$('comment').onchange=()=>{checkpoint();a.comment=$('comment').value;changed()};$('delete').onclick=deleteSelected;
}
function deleteSelected(){if(!cell())return;checkpoint();doc.annotations=doc.annotations.filter(a=>a.id!==selected);selected=null;changed();status('Сома удалена. Ctrl+Z вернёт её.')}
function finish(){
 let p=draft.filter((v,i,a)=>i===0||Math.hypot(v[0]-a[i-1][0],v[1]-a[i-1][1])>.15);
 if(p.length>3&&Math.hypot(p[0][0]-p.at(-1)[0],p[0][1]-p.at(-1)[1])<.3)p.pop();
 if(!validPolygon(p)){status('Контур слишком мал или пересекает сам себя. Esc — отменить; в полигоне Backspace убирает точку.');return}
 checkpoint();const c=crop();const boundary=p.some(([x,y])=>x-c.x<1||c.x+c.width-x<1||y-c.y<1||c.y+c.height-y<1);
 const a={id:crypto.randomUUID(),number:doc.next_number++,points_xy:p.map(v=>v.map(n=>Math.round(n*1000)/1000)),crop_id:c.id,confidence:'confident',truncated:boundary,comment:'',created_at:new Date().toISOString()};doc.annotations.push(a);selected=a.id;draft=[];changed();status(`Сома #${a.number} сохранена${boundary?' · отмечена как неполная у края кропа':''}. Можно сразу обводить следующую.`)
}
function history(redo=false){if(draft.length){draft=[];draw();return}const from=redo?redoStack:undoStack,to=redo?undoStack:redoStack;if(!from.length)return;to.push(JSON.stringify(doc));doc=JSON.parse(from.pop());if(!doc.crops.some(c=>c.id===current))current=doc.crops[0].id;selected=null;renderAllImages();changed()}
canvas.addEventListener('pointerdown',e=>{
 if(!doc||e.button!==0)return;canvas.focus();const p=point(e),c=crop();
 if(space||tool==='move'){gesture={kind:'pan',sx:e.clientX,sy:e.clientY,pan:[...pan]};canvas.setPointerCapture(e.pointerId);draw();return}
 if(!onImage(e))return;
 if(tool==='select'){
  const a=cell();if(a&&showMasks){let best=-1,d=9;for(let i=0;i<a.points_xy.length;i++){const q=screen(a.points_xy[i]),s=screen(p),dist=Math.hypot(q[0]-s[0],q[1]-s[1]);if(dist<d){best=i;d=dist}}
   if(best>=0){if(e.altKey){if(a.points_xy.length<=3)return status('В контуре должно остаться хотя бы три вершины.');const candidate=a.points_xy.filter((_,i)=>i!==best);if(!validPolygon(candidate))return status('Удаление вершины создаёт некорректный контур.');checkpoint();a.points_xy=candidate;changed();return}checkpoint();gesture={kind:'vertex',index:best,old:clone(a.points_xy)};canvas.setPointerCapture(e.pointerId);return}
   if(e.shiftKey){let edge=-1,bestD=8;const ps=screen(p);for(let i=0;i<a.points_xy.length;i++){const u=screen(a.points_xy[i]),v=screen(a.points_xy[(i+1)%a.points_xy.length]);const dx=v[0]-u[0],dy=v[1]-u[1],t=Math.max(0,Math.min(1,((ps[0]-u[0])*dx+(ps[1]-u[1])*dy)/(dx*dx+dy*dy||1)));const dist=Math.hypot(ps[0]-u[0]-t*dx,ps[1]-u[1]-t*dy);if(dist<bestD){bestD=dist;edge=i}}if(edge>=0){checkpoint();a.points_xy.splice(edge+1,0,p);changed();return}}
  }
  selected=showMasks?[...visible()].reverse().find(a=>contains(a.points_xy,p))?.id||null:null;renderUI();return;
 }
 if(tool==='polygon'){draft.push(p);draw();return}
 draft=[p];gesture={kind:'draw'};canvas.setPointerCapture(e.pointerId);draw();
});
canvas.addEventListener('pointermove',e=>{if(!gesture)return;if(gesture.kind==='pan'){pan=[gesture.pan[0]+e.clientX-gesture.sx,gesture.pan[1]+e.clientY-gesture.sy];draw();return}const p=point(e);
 if(gesture.kind==='vertex'){cell().points_xy[gesture.index]=p;draw();return}
 if(gesture.kind==='draw'&&Math.hypot(p[0]-draft.at(-1)[0],p[1]-draft.at(-1)[1])>.65){draft.push(p);draw()}
});
function endPointer(e,cancel=false){if(!gesture)return;const g=gesture;gesture=null;if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
 if(g.kind==='pan')draw();
 if(g.kind==='vertex'){if(cancel||!validPolygon(cell().points_xy)){cell().points_xy=g.old;undoStack.pop();status('Некорректное пересечение контура: правка отменена.');renderUI()}else{changed();status('Контур обновлён.')}}
 if(g.kind==='draw'){if(cancel){draft=[];draw()}else finish()}
}
canvas.addEventListener('pointerup',e=>endPointer(e));canvas.addEventListener('pointercancel',e=>endPointer(e,true));
canvas.addEventListener('dblclick',e=>{if(tool==='polygon')finish()});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.6,Math.min(8,zoom*(e.deltaY<0?1.12:1/1.12)));draw()},{passive:false});
overview.onclick=e=>{if(draft.length)return status('Завершите контур перед выбором нового кропа.');const r=overview.getBoundingClientRect(),size=Number($('cropSize').value),y=(e.clientX-r.left)/r.width*meta.shape_yx[0],x=(e.clientY-r.top)/r.height*meta.shape_yx[1];const w=Math.min(size,meta.shape_yx[1]),h=Math.min(size,meta.shape_yx[0]);checkpoint();const c={id:crypto.randomUUID(),name:'Свой кроп '+(doc.crops.filter(c=>c.custom).length+1),x:Math.max(0,Math.min(meta.shape_yx[1]-w,Math.round(x-w/2))),y:Math.max(0,Math.min(meta.shape_yx[0]-h,Math.round(y-h/2))),width:w,height:h,custom:true};doc.crops.push(c);switchCrop(c.id);renderThumbnails();changed()};
['freehand','polygon','select','move'].forEach(t=>$(t).onclick=()=>setTool(t));$('finish').onclick=finish;$('undo').onclick=()=>history();$('redo').onclick=()=>history(true);$('hide').onclick=()=>{showMasks=!showMasks;draw();drawOverview()};$('zoomIn').onclick=()=>{zoom=Math.min(8,zoom*1.25);draw()};$('zoomOut').onclick=()=>{zoom=Math.max(.6,zoom/1.25);draw()};$('fit').onclick=()=>{zoom=1;pan=[0,0];draw()};
function setContrast(b,w){black=Math.max(0,b);white=Math.max(black+1,w);$('black').max=Math.max(500,Math.ceil(white));$('white').max=Math.max(1500,Math.ceil(white));$('black').value=black;$('white').value=white;$('blackVal').textContent=Math.round(black);$('whiteVal').textContent=Math.round(white);renderAllImages()}
$('black').oninput=()=>setContrast(Number($('black').value),white);$('white').oninput=()=>setContrast(black,Number($('white').value));$('autoContrast').onclick=()=>{const c=crop(),v=[];for(let y=c.y;y<c.y+c.height;y++)for(let x=c.x;x<c.x+c.width;x++)v.push(raw[y*meta.shape_yx[1]+x]);v.sort((a,b)=>a-b);setContrast(v[Math.floor(v.length*.01)],v[Math.floor(v.length*.995)])};
$('helpToggle').onclick=()=>{$('help').hidden=!$('help').hidden};
$('export').onclick=()=>{if(draft.length)return status('Завершите текущий контур перед экспортом.');download(`somas_frame${meta.frame_index}.json`,JSON.stringify(exportDoc(),null,2),'application/json');status('JSON экспортирован: контуры, координаты, признаки и размеры.')};
$('csv').onclick=()=>{const fields=['id','number','confidence','truncated','crop_id','area_px2','equivalent_diameter_px','major_axis_px','minor_axis_px','aspect_ratio','circularity','perimeter_px','orientation_deg','centroid_x','centroid_y','comment'];const quote=v=>'"'+String(v??'').replaceAll('"','""')+'"';const rows=doc.annotations.map(a=>{const all={...a,...measure(a.points_xy)};return fields.map(k=>quote(all[k])).join(',')});download(`soma_sizes_frame${meta.frame_index}.csv`,'\uFEFF'+[fields.join(','),...rows].join('\n'),'text/csv;charset=utf-8');status('CSV экспортирован. Для распределения используйте confident и truncated=false.')};
$('import').onclick=()=>$('fileInput').click();
function validateImport(d){if(d.format!=='soma-annotations-v1'||d.dataset?.frame_sha256!==meta.frame_sha256||d.dataset?.frame_index!==meta.frame_index||JSON.stringify(d.dataset?.shape_yx)!==JSON.stringify(meta.shape_yx))throw Error('JSON относится к другому кадру или формату.');if(!Array.isArray(d.annotations)||!Array.isArray(d.crops)||!d.crops.length)throw Error('Нет списка контуров или кропов.');const ids=new Set(),numbers=new Set();for(const a of d.annotations){if(!Number.isSafeInteger(a.number)||a.number<1||numbers.has(a.number)||ids.has(a.id)||typeof a.id!=='string'||!validPolygon(a.points_xy)||a.points_xy.some(([x,y])=>x<0||x>meta.shape_yx[1]||y<0||y>meta.shape_yx[0])||!['confident','uncertain'].includes(a.confidence)||typeof a.truncated!=='boolean')throw Error('Некорректная сома в JSON.');ids.add(a.id);numbers.add(a.number)}const cropIds=new Set();for(const c of d.crops){if(typeof c.id!=='string'||cropIds.has(c.id)||![c.x,c.y,c.width,c.height].every(Number.isInteger)||c.x<0||c.y<0||c.width<1||c.height<1||c.x+c.width>meta.shape_yx[1]||c.y+c.height>meta.shape_yx[0])throw Error('Некорректный кроп.');cropIds.add(c.id)}if(d.annotations.some(a=>!cropIds.has(a.crop_id)))throw Error('Контур относится к неизвестному кропу.');if(!Number.isSafeInteger(d.next_number)||d.next_number<=Math.max(0,...d.annotations.map(a=>a.number)))throw Error('Некорректный счётчик сом.');return d}
$('fileInput').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const d=validateImport(JSON.parse(await file.text()));if(doc.annotations.length&&!confirm('Заменить текущую разметку импортированным JSON? Текущую версию можно вернуть через Ctrl+Z.'))return;checkpoint();doc=d;current=doc.crops[0].id;selected=null;draft=[];zoom=1;pan=[0,0];renderAllImages();changed();status('Разметка импортирована.')}catch(e){error(e.message)}finally{$('fileInput').value=''}};
window.addEventListener('keydown',e=>{if(['INPUT','TEXTAREA','SELECT'].includes(document.activeElement.tagName))return;if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();save();return}if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'){e.preventDefault();history(e.shiftKey);return}if(e.code==='Space'){e.preventDefault();space=true;draw();return}if(e.key==='Escape'){draft=[];selected=null;renderUI();status('Контур отменён.');return}if(e.key==='Enter'&&draft.length){e.preventDefault();finish();return}if(e.key==='Backspace'&&draft.length){e.preventDefault();draft.pop();draw();return}if(e.key==='Delete'&&!draft.length){e.preventDefault();deleteSelected();return}const k=e.key.toLowerCase();if(k==='f')setTool('freehand');if(k==='p')setTool('polygon');if(k==='v')setTool('select');if(k==='m')setTool('move');if(k==='h')$('hide').click()});
window.addEventListener('keyup',e=>{if(e.code==='Space'){space=false;draw()}});window.addEventListener('blur',()=>{space=false;draw()});
window.addEventListener('beforeunload',e=>{if(pending||saving||draft.length){e.preventDefault();e.returnValue=''}});
new ResizeObserver(()=>{const r=$('stage').getBoundingClientRect();W=r.width;H=r.height;const dpr=window.devicePixelRatio||1;canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);draw()}).observe($('stage'));
async function init(){try{
 meta=await (await fetch('assets/dataset.json')).json();const buf=await(await fetch('assets/frame.bin')).arrayBuffer();if(buf.byteLength!==meta.shape_yx[0]*meta.shape_yx[1]*2)throw Error('Размер кадра не совпадает с метаданными');const dv=new DataView(buf);raw=new Uint16Array(buf.byteLength/2);for(let i=0;i<raw.length;i++)raw[i]=dv.getUint16(i*2,true);
 storageKey='soma-lab:'+meta.frame_sha256+':'+meta.frame_index;
 const saved=await fetch('/api/annotations');if(!saved.ok)throw Error('Не удалось прочитать локальную разметку');doc=await saved.json();let browserCopy;try{browserCopy=JSON.parse(localStorage.getItem(storageKey)||'null')}catch{}
 if(browserCopy){try{validateImport(browserCopy);if((browserCopy.updated_at||'')>(doc.updated_at||'')){doc=browserCopy;pending=true;status('Восстановлена более новая копия из браузера.')}}catch{}}
 validateImport(doc);current=(doc.crops[2]||doc.crops[0]).id;black=meta.display_range[0];white=meta.display_range[1];$('source').textContent=`${meta.source_name} · кадр ${meta.frame_index} (с нуля) · ${meta.shape_yx[0]}×${meta.shape_yx[1]} px`;$('saveState').textContent='✓ Локальная разметка';setContrast(black,white);renderUI();status('Готово. Обведите всю сому, включая тёмный центр. Изменения сохраняются автоматически.');if(pending)save();
 }catch(e){error('Не удалось открыть песочницу: '+e.message+'. Запустите python tools/annotate_somas.py из репозитория.');status('Ошибка загрузки')}
}
init();
