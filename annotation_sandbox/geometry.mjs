export function areaOf(points) {
  return Math.abs(points.reduce((s,p,i)=>{const q=points[(i+1)%points.length];return s+p[0]*q[1]-q[0]*p[1]},0))/2;
}
function cross(a,b,c){return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])}
export function validPolygon(p){
  if(p.length<3||p.some(v=>v.length!==2||!v.every(Number.isFinite))||areaOf(p)<2)return false;
  const on=(a,b,c)=>Math.abs(cross(a,b,c))<1e-8&&c[0]>=Math.min(a[0],b[0])-1e-8&&c[0]<=Math.max(a[0],b[0])+1e-8&&c[1]>=Math.min(a[1],b[1])-1e-8&&c[1]<=Math.max(a[1],b[1])+1e-8;
  const hits=(a,b,c,d)=>{let u=cross(a,b,c),v=cross(a,b,d),w=cross(c,d,a),z=cross(c,d,b);return (u*v<0&&w*z<0)||on(a,b,c)||on(a,b,d)||on(c,d,a)||on(c,d,b)};
  for(let i=0;i<p.length;i++)for(let j=i+1;j<p.length;j++){
    if(j===i+1||(i===0&&j===p.length-1))continue;
    if(hits(p[i],p[(i+1)%p.length],p[j],p[(j+1)%p.length]))return false;
  }
  return true;
}
export function contains(p,[x,y]){
  let inside=false;for(let i=0,j=p.length-1;i<p.length;j=i++){
    const a=p[i],b=p[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])inside=!inside;
  }return inside;
}
export function measure(p){
  // Exact area moments of the filled polygon (not covariance of its vertices).
  let s=0,cx=0,cy=0,xx=0,yy=0,xy=0,perimeter=0;
  for(let i=0;i<p.length;i++){
    const [x,y]=p[i],[u,v]=p[(i+1)%p.length],c=x*v-u*y;
    s+=c;cx+=(x+u)*c;cy+=(y+v)*c;
    xx+=(x*x+x*u+u*u)*c;yy+=(y*y+y*v+v*v)*c;
    xy+=(2*x*y+x*v+u*y+2*u*v)*c;
    perimeter+=Math.hypot(x-u,y-v);
  }
  const a=s/2;cx/=6*a;cy/=6*a;
  xx=xx/(12*a)-cx*cx;yy=yy/(12*a)-cy*cy;xy=xy/(24*a)-cx*cy;
  const d=Math.hypot(xx-yy,2*xy),l1=Math.max(0,(xx+yy+d)/2),l2=Math.max(0,(xx+yy-d)/2);
  const area=Math.abs(a),major=4*Math.sqrt(l1),minor=4*Math.sqrt(l2);
  return {area_px2:area,perimeter_px:perimeter,equivalent_diameter_px:2*Math.sqrt(area/Math.PI),
    major_axis_px:major,minor_axis_px:minor,aspect_ratio:minor>0?major/minor:null,
    circularity:4*Math.PI*area/(perimeter*perimeter),orientation_deg:Math.atan2(2*xy,xx-yy)*90/Math.PI,
    centroid_x:cx,centroid_y:cy};
}
