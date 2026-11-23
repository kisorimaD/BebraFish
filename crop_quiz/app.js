const $=id=>document.getElementById(id);
let meta,trial=null,answered=false,busy=false,stats,history=[];
const labels={real:'реальные данные',synthetic:'синтетика'};
function error(message){$('error').textContent=message;$('error').hidden=!message}
function status(message){$('status').textContent=message}
async function api(path,body){const response=await fetch(path,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});const data=await response.json();if(!response.ok)throw Error(data.error||'Ошибка сервера');return data}
function controls(){for(const id of ['real','synthetic'])$(id).disabled=busy||!trial||answered;$('next').disabled=busy||!answered;$('apply').disabled=busy;$('skip').disabled=busy;$('export').disabled=!history.length}
function renderStats(){if(!stats)return;$('accuracy').textContent=stats.total?Math.round(stats.accuracy*100)+'%':'—';$('count').textContent=stats.correct+' верных из '+stats.total+' ответов';for(const label of ['real','synthetic']){const s=stats.by_class[label];$(label+'Stat').textContent=s.correct+' / '+s.total}for(const mode of ['fixed','normalized']){const s=stats.by_contrast?.[mode]||{correct:0,total:0};$(mode+'Stat').textContent=s.correct+' / '+s.total}}
function settings(){const width=Number($('width').value),height=Number($('height').value);if(!Number.isInteger(width)||!Number.isInteger(height)||width<8||height<8||width>meta.max_width||height>meta.max_height)throw Error('Проверьте размеры кропа.');return{width,height,strict:$('strict').checked,contrast:$('contrast').value}}
function resize(){if(!trial)return;const z=$('zoom').value,b=$('stage').getBoundingClientRect();const scale=z==='fit'?Math.max(.1,Math.min((b.width-30)/trial.width,(b.height-30)/trial.height)):Number(z);$('crop').style.width=(trial.width*scale)+'px';$('crop').style.height=(trial.height*scale)+'px'}
async function next(){if(busy||!meta)return;busy=true;controls();error('');status('Выбираю случайный кроп…');try{
 const config=settings(),fresh=await api('/api/trial',config);const image=new Image();image.src=fresh.image_url;await image.decode();
 trial=fresh;answered=false;$('crop').src=image.src;$('crop').hidden=false;$('empty').hidden=true;
 $('feedback').className='feedback';$('feedbackText').textContent='Какой это источник?';$('reveal').textContent='';for(const id of ['real','synthetic'])$(id).classList.remove('chosen');
 $('sizeLabel').textContent=trial.width+' × '+trial.height+' px';$('trialLabel').textContent='Вопрос '+(stats.total+1);resize();
 status('Допустимых позиций: '+trial.eligible_positions.toLocaleString('ru-RU')+'. '+(config.contrast==='normalized'?'Контраст кропа нормирован. ':'Общая шкала ADU. ')+'Источник скрыт до ответа.');
 try{localStorage.setItem('crop-quiz-settings',JSON.stringify({...config,zoom:$('zoom').value}))}catch{}
 }catch(e){error(e.message);status('Не удалось получить новый кроп.')}finally{busy=false;controls()}}
async function answer(guess){if(busy||!trial||answered)return;busy=true;controls();error('');try{
 const data=await api('/api/answer',{id:trial.id,guess});answered=true;stats=data.stats;if(!history.some(r=>r.id===data.result.id))history.push(data.result);
 $(guess).classList.add('chosen');$('feedback').className='feedback '+(data.result.correct?'correct':'wrong');$('feedbackText').textContent=(data.result.correct?'Верно — это ':'Это ')+labels[data.result.truth]+(data.result.correct?'.':'. Ваш ответ: '+labels[guess]+'.');
 const[y0,y1,x0,x1]=data.result.crop_yx;$('reveal').textContent='Y '+y0+':'+y1+' · X '+x0+':'+x1+(data.result.frame===null?'':' · кадр '+data.result.frame);renderStats();status('Ответ записан. Enter — следующий кроп.');$('saved').textContent='✓ Ответы сохранены на диск';
 }catch(e){error(e.message)}finally{busy=false;controls()}}
$('settings').onsubmit=e=>{e.preventDefault();next()};$('next').onclick=next;$('skip').onclick=next;$('real').onclick=()=>answer('real');$('synthetic').onclick=()=>answer('synthetic');$('zoom').onchange=resize;
for(const button of document.querySelectorAll('[data-size]'))button.onclick=()=>{$('width').value=button.dataset.size;$('height').value=button.dataset.size};
$('export').onclick=()=>{const data={format:'crop-quiz-results-v1',session:meta.session,metadata:meta,stats,answers:history};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='crop_quiz_'+meta.session+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),5000)};
window.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA'].includes(document.activeElement?.tagName))return;if(e.key==='1'){e.preventDefault();answer('real')}if(e.key==='2'){e.preventDefault();answer('synthetic')}if(e.key==='Enter'&&answered){e.preventDefault();next()}});
new ResizeObserver(resize).observe($('stage'));
async function init(){try{
 meta=await api('/api/meta');if(meta.api_version!==2)throw Error('Нужна новая версия сервера. Откройте обновлённый адрес из чата или перезапустите tools/crop_quiz.py');const saved=await api('/api/history');stats=saved.stats;history=saved.answers;renderStats();
 $('width').max=meta.max_width;$('height').max=meta.max_height;
 try{const copy=JSON.parse(localStorage.getItem('crop-quiz-settings'));if(copy){$('width').value=copy.width;$('height').value=copy.height;$('strict').checked=copy.strict;$('contrast').value=copy.contrast==='normalized'?'normalized':'fixed';$('zoom').value=['1','2','4','fit'].includes(copy.zoom)?copy.zoom:'4'}}catch{}
 $('sources').textContent=meta.real_sampling+'; '+meta.synthetic_sampling.toLowerCase()+'. Шкала '+meta.display_range.map(v=>v.toFixed(0)).join('–')+' ADU. '+meta.orientation;
 $('sessionFile').textContent=meta.results_file;controls();await next();
 }catch(e){error('Не удалось открыть тест: '+e.message+'. Запустите python tools/crop_quiz.py --open');status('Сервер недоступен.')}}
init();
