// Polygon vertices are native integer pixel centers; boundary pixels included.
export function inPolygon(x,y,polygon){
 if(!polygon)return true;let inside=false;
 for(let i=0,j=polygon.length-1;i<polygon.length;j=i++){
  const[a,b]=polygon[j],[c,d]=polygon[i];
  if(Math.abs((x-a)*(d-b)-(y-b)*(c-a))<1e-8&&x>=Math.min(a,c)&&x<=Math.max(a,c)&&y>=Math.min(b,d)&&y<=Math.max(b,d))return true;
  if((b>y)!==(d>y)&&x<(c-a)*(y-b)/(d-b)+a)inside=!inside;
 }return inside;
}
// Native raster order: index=y*width+x. Each seed has an absolute upper ADU
// threshold. Connectivity never follows a drifting neighbor-relative tolerance.
export function flood(image,width,height,seed,connectivity=4,bounds=null){
 if(image.length!==width*height)throw Error('Размер изображения не совпадает');
 const result=new Uint8Array(image.length);
 if(!seed.enabled)return {mask:result,count:0,touchesEdge:false};
 const x=seed.x,y=seed.y,t=seed.threshold;
 if(![4,8].includes(connectivity))throw Error('Связность должна быть 4 или 8');
 if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||x>=width||y<0||y>=height||!Number.isFinite(t))throw Error('Некорректная точка');
 const b=bounds||{x:0,y:0,width,height};
 if(![b.x,b.y,b.width,b.height].every(Number.isInteger)||b.width<1||b.height<1||b.x<0||b.y<0||b.x+b.width>width||b.y+b.height>height||x<b.x||x>=b.x+b.width||y<b.y||y>=b.y+b.height)throw Error('Точка или границы кропа некорректны');
 const x0=b.x,y0=b.y,x1=b.x+b.width-1,y1=b.y+b.height-1;
 const start=y*width+x;if(!inPolygon(x,y,seed.polygon)||image[start]>t+1e-4)return {mask:result,count:0,touchesEdge:false};
 const queue=new Int32Array(b.width*b.height);let head=0,tail=1;queue[0]=start;result[start]=1;let touchesEdge=false;
 const visit=index=>{if(!result[index]&&image[index]<=t+1e-4&&inPolygon(index%width,Math.floor(index/width),seed.polygon)){result[index]=1;queue[tail++]=index}};
 while(head<tail){const i=queue[head++],cx=i%width,cy=Math.floor(i/width);
  if(cx===x0||cy===y0||cx===x1||cy===y1)touchesEdge=true;
  if(cx>x0)visit(i-1);if(cx<x1)visit(i+1);if(cy>y0)visit(i-width);if(cy<y1)visit(i+width);
  if(connectivity===8){if(cx>x0&&cy>y0)visit(i-width-1);if(cx<x1&&cy>y0)visit(i-width+1);if(cx>x0&&cy<y1)visit(i+width-1);if(cx<x1&&cy<y1)visit(i+width+1)}
 }
 return {mask:result,count:tail,touchesEdge};
}
export function unionMasks(masks,size){const out=new Uint8Array(size);for(const mask of masks)for(let i=0;i<size;i++)out[i]|=mask[i];return out}
