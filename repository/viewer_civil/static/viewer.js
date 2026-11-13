/* MyCON civil inspection UI - independent Plotly WebGL frontend, no backend calls. */
'use strict';
(() => {
const B = JSON.parse(document.getElementById('mycon-payload').textContent);
const A=B.assets, M=B.manifest;
const $=id=>document.getElementById(id);
const state={mode:'confidence',selectedCamera:null,layers:{mesh:true,point_cloud:true,planes:false,edges:true,corners:true,cameras:true,path:true,unknown:true,bim:false,deviation:false}};
const swatches={point_cloud:'#caa9df',mesh:'#4cb8a4',planes:'#577f99',edges:'#f1bc7e',corners:'#c6e6f0',cameras:'#b4e3f1',path:'#7cb9e6',unknown:'#ce8ab2',bim:'#8799c1',deviation:'#ef7777'};
const labels={point_cloud:'Point samples (optional)',mesh:'Reconstructed surfaces',planes:'Plane patches',edges:'Edges / corners lines',corners:'Corner candidates',cameras:'Camera frustums',path:'Camera trajectory',unknown:'Unknown / unsupported',bim:'Shape 4 / BIM overlay',deviation:'Metric deviation heatmap'};
const traces=[], groupIndices={};
let HAS_WEBGL=(()=>{try{const c=document.createElement('canvas');return !!(c.getContext('webgl')||c.getContext('experimental-webgl'));}catch(e){return false}})();
const fallback={yaw:.74,pitch:.42,zoom:1,canvas:null,coords:[]};
// World bounds are data-driven: real COLMAP scenes are not fixed-size synthetic rooms.
const scenePoints=[...(A.mesh?.vertices||[]),...(A.point_cloud?.points||[]),
  ...(A.planes?.planes||[]).flatMap(p=>p.vertices||[]),
  ...(A.corners?.corners||[]).map(p=>p.xyz),
  ...(A.camera_poses?.cameras||[]).map(c=>[c.camera_to_world[0][3],c.camera_to_world[1][3],c.camera_to_world[2][3]])]
  .filter(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite));
const bounds=[0,1,2].map(i=>[Math.min(...scenePoints.map(p=>p[i])),Math.max(...scenePoints.map(p=>p[i]))]);
const sceneCenter=bounds.map(b=>(Number.isFinite(b[0])&&Number.isFinite(b[1]))?(b[0]+b[1])/2:0);
const sceneRadius=Math.max(1,...bounds.map(b=>Number.isFinite(b[1]-b[0])?(b[1]-b[0])/2:1));
const extent=bounds.map((b,i)=>[sceneCenter[i]-sceneRadius*1.4,sceneCenter[i]+sceneRadius*1.4]);
function add(group, trace) { trace.visible=state.layers[group] ? true : false; trace.showlegend=false; trace.hoverinfo=trace.hoverinfo||'skip'; traces.push(trace); (groupIndices[group]??=[]).push(traces.length-1); }
function scalarColor(value,mode){
 if(value===null||value===undefined||!Number.isFinite(Number(value)))return '#b176a8';
 if(mode==='neutral')return '#668e9a';
 const v=Math.max(0,Math.min(1,Number(value)));
 return v>.8?'#58c5a6':v>.55?'#d4be74':v>.3?'#d58c65':'#ad6d75';
}
function world(m,v) {return [0,1,2].map(i=>m[i][0]*v[0]+m[i][1]*v[1]+m[i][2]*v[2]+m[i][3]);}
function line(points, color, width=3,extra={}) {
 return {type:'scatter3d',mode:'lines',x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(p=>p[2]),line:{color,width},...extra};
}
function buildTraces(){
 traces.length=0;Object.keys(groupIndices).forEach(k=>delete groupIndices[k]);
 const mesh=A.mesh,attrs=mesh.face_attributes||[];
 if(A.point_cloud?.points?.length){
  const pts=A.point_cloud.points;
  add('point_cloud',{type:'scatter3d',mode:'markers',x:pts.map(p=>p[0]),y:pts.map(p=>p[1]),z:pts.map(p=>p[2]),
   marker:{size:2,color:'#caa9df'},hoverinfo:'skip'});
 }
 const xs=mesh.vertices.map(v=>v[0]),ys=mesh.vertices.map(v=>v[1]),zs=mesh.vertices.map(v=>v[2]);
 const perFace=mesh.faces.map((f,i)=>scalarColor(attrs[i]?.[state.mode==='coverage'?'coverage':'confidence'],state.mode));
 if(mesh.faces.length) add('mesh',{type:'mesh3d',x:xs,y:ys,z:zs,i:mesh.faces.map(f=>f[0]),j:mesh.faces.map(f=>f[1]),k:mesh.faces.map(f=>f[2]),facecolor:perFace,opacity:.29,flatshading:true,lighting:{ambient:0.9,diffuse:0.4},hovertemplate:'Geometry candidate<extra></extra>'});
 (A.planes.planes||[]).forEach(plane=>{
  const v=plane.vertices;if(!v||v.length<3)return;
  add(plane.state==='unknown'?'unknown':'planes',line([...v,v[0]],scalarColor(plane[state.mode==='coverage'?'coverage':'confidence'],state.mode),4,{hoverinfo:'text',text:[plane.id]}));
 });
 (A.edges.edges||[]).forEach(edge=>{
  const unknown=edge.state==='unknown'||edge.state==='candidate_extrapolated';
  add(unknown?'unknown':'edges',line([edge.start,edge.end],scalarColor(edge.confidence,state.mode),6,{hoverinfo:'text',text:[edge.id,edge.id]}));
 });
 const known=A.corners.corners.filter(c=>c.state!=='unknown'&&c.confidence!==null);
 const unknown=A.corners.corners.filter(c=>c.state==='unknown'||c.confidence===null);
 for(const [group,arr] of [['corners',known],['unknown',unknown]]){
  if(arr.length)add(group,{type:'scatter3d',mode:'markers',x:arr.map(c=>c.xyz[0]),y:arr.map(c=>c.xyz[1]),z:arr.map(c=>c.xyz[2]),marker:{size:6,color:arr.map(c=>scalarColor(c.confidence,state.mode)),symbol:group==='unknown'?'diamond':'circle',line:{color:'#dbe6ec',width:.5}},text:arr.map(c=>c.id),hoverinfo:'text'});
 }
 const cameras=A.camera_poses.cameras;
 add('cameras',{type:'scatter3d',mode:'markers+text',x:cameras.map(c=>c.camera_to_world[0][3]),y:cameras.map(c=>c.camera_to_world[1][3]),z:cameras.map(c=>c.camera_to_world[2][3]),marker:{size:7,color:'#83e7d1',symbol:'diamond',line:{color:'#e9fff5',width:1}},text:cameras.map(c=>c.camera_id.replace('CAM-','C')),textposition:'top center',textfont:{size:10,color:'#bde8e3'},customdata:cameras.map(c=>c.camera_id),hovertemplate:'Camera %{customdata}<extra></extra>'});
 cameras.forEach(c=>{
  const m=c.camera_to_world, center=world(m,[0,0,0]),z=.45;
  const fl=[world(m,[-z*c.cx/c.fx,-z*c.cy/c.fy,z]),world(m,[z*(c.width-c.cx)/c.fx,-z*c.cy/c.fy,z]),world(m,[z*(c.width-c.cx)/c.fx,z*(c.height-c.cy)/c.fy,z]),world(m,[-z*c.cx/c.fx,z*(c.height-c.cy)/c.fy,z])];
  const pts=[];for(const f of fl){pts.push(center,f,[NaN,NaN,NaN]);}
  pts.push(fl[0],fl[1],fl[2],fl[3],fl[0]);
  add('cameras',line(pts,'#70cbb8',3));
 });
 add('path',line(cameras.map(c=>[c.camera_to_world[0][3],c.camera_to_world[1][3],c.camera_to_world[2][3]]),'#86aaff',4));
 const bim=mesh.bim_overlay;
 if(bim?.vertices?.length&&bim?.faces?.length){
  const edgeSet=new Set();bim.faces.forEach(t=>{for(let j=0;j<3;j++){const a=t[j],b=t[(j+1)%3];edgeSet.add([Math.min(a,b),Math.max(a,b)].join(','));}});
  for(const s of edgeSet){const [a,b]=s.split(',').map(Number);add('bim',line([bim.vertices[a],bim.vertices[b]],'#b1a1dc',2));}
 }
 if(B.deviation_enabled && A.mesh.deviations?.per_face_m){
  const vals=A.mesh.deviations.per_face_m;
  const maxAbs=Math.max(1e-9,...vals.filter(v=>typeof v==='number'&&Number.isFinite(v)).map(v=>Math.abs(v)));
  add('deviation',{type:'mesh3d',x:xs,y:ys,z:zs,i:mesh.faces.map(f=>f[0]),j:mesh.faces.map(f=>f[1]),k:mesh.faces.map(f=>f[2]),facecolor:vals.map(v=>v===null?'#7a6685':v<-.35*maxAbs?'#62aac8':v>.35*maxAbs?'#e47e67':'#73baab'),opacity:.60,hoverinfo:'skip'});
 }
}
function render(preserve=true){
 if(!HAS_WEBGL || typeof Plotly==='undefined'){HAS_WEBGL=false;renderCanvas();return Promise.resolve();}
 const camera=preserve && $('scene')._fullLayout?.scene?.camera ? $('scene')._fullLayout.scene.camera : {eye:{x:1.55,y:1.7,z:1.07},up:{x:0,y:0,z:1}};
 buildTraces();
 return Plotly.react('scene',traces,{paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:0,r:0,t:0,b:0},showlegend:false,
   scene:{bgcolor:'rgba(0,0,0,0)',camera,aspectmode:'data',xaxis:{range:extent[0],title:'X',gridcolor:'#2d475577',zerolinecolor:'#2d4755',showbackground:false,color:'#7893a7',tickfont:{size:9}},yaxis:{range:extent[1],title:'Y',gridcolor:'#2d475577',zerolinecolor:'#2d4755',showbackground:false,color:'#7893a7',tickfont:{size:9}},zaxis:{range:extent[2],title:'Z',gridcolor:'#2d475577',zerolinecolor:'#2d4755',showbackground:false,color:'#7893a7',tickfont:{size:9}}}},
 {responsive:true,displaylogo:false,modeBarButtonsToRemove:['sendDataToCloud','lasso2d','select2d']});
}
function refreshLayers(){
 const holder=$('layers');holder.innerHTML='';Object.entries(labels).forEach(([key,name])=>{
  const isDisabled=key==='deviation'&&!B.deviation_enabled;
  const wrap=document.createElement('label');wrap.className='layer'+(isDisabled?' disabled':'');wrap.title=isDisabled?'Disabled: '+B.deviation_reason:'';
  const badge=document.createElement('span');badge.className='name';
  const sw=document.createElement('i');sw.className='swatch';sw.style.background=swatches[key];badge.appendChild(sw);badge.appendChild(document.createTextNode(name));
  const cb=document.createElement('input');cb.type='checkbox';cb.checked=state.layers[key];cb.disabled=isDisabled;cb.dataset.layer=key;
  cb.addEventListener('change',()=>setLayer(key,cb.checked));wrap.append(badge,cb);holder.appendChild(wrap);
 });
}
function setLayer(key,on){
 if(!(key in state.layers))throw new Error('Unknown layer '+key);
 if(key==='deviation'&&!B.deviation_enabled){state.layers.deviation=false;refreshLayers();return false;}
 state.layers[key]=!!on;
 const idx=groupIndices[key]||[];
 if(!HAS_WEBGL)renderCanvas();else if(idx.length)Plotly.restyle('scene',{visible:state.layers[key]},idx);
 refreshLayers();return true;
}
function selectCamera(id){
 const c=A.camera_poses.cameras.find(c=>c.camera_id===id);if(!c)throw new Error('Camera not found: '+id);
 const f=A.frame_manifest.frames.find(f=>f.frame_id===c.frame_id);if(!f)throw new Error('Missing linked frame '+c.frame_id);
 state.selectedCamera=id;$('selected-camera').textContent=id;$('camera-position').textContent='Pose in '+M.coordinate_frame+' · units: '+M.units;
 $('selected-frame').src=f.preview_path;$('selected-frame').alt='Source frame for '+id;
 $('frame-meta').textContent=f.frame_id+' · '+c.width+' × '+c.height+' px · view '+(f.source_view_id||'unknown')+' · '+(f.split||'unknown split')+' · '+(c.calibration_model||'calibration unspecified')+(c.frustum_projection==='approximate_ignore_distortion'?' · APPROXIMATE FRUSTUM (distortion)':'');
 document.querySelectorAll('.camera-btn').forEach(el=>el.classList.toggle('active',el.dataset.camera===id));
 if(!HAS_WEBGL && fallback.canvas)renderCanvas();
 return c.frame_id;
}
function orbit(t){if(!HAS_WEBGL){fallback.yaw=t*.4;renderCanvas();return Promise.resolve()}const a=t*.40;return Plotly.relayout('scene',{'scene.camera.eye':{x:1.9*Math.cos(a),y:1.9*Math.sin(a),z:1.08}});}

// Zero-network, Canvas 2D perspective fallback for research review environments
// with disabled WebGL (for example restricted CI/remote Chromium). Geometry is
// projected from actual XYZ and is still rotatable/zoomable and layer selectable.
function canvasProject(p,W,H){
 const x=p[0]-sceneCenter[0],y=p[1]-sceneCenter[1],z=p[2]-sceneCenter[2],ca=Math.cos(fallback.yaw),sa=Math.sin(fallback.yaw),cp=Math.cos(fallback.pitch),sp=Math.sin(fallback.pitch);
 const a=ca*x-sa*y,b=sa*x+ca*y;
 const perspective=Math.max(.15,1 + .08*(b*cp+z*sp)/sceneRadius);
 const scale=Math.min(W,H)/(sceneRadius*3.1)*fallback.zoom;
 return [W/2+a*scale/perspective,H/2+(b*sp-z*cp)*scale/perspective,b*cp+z*sp];
}
function renderCanvas(){
 let canvas=fallback.canvas;
 const host=$('scene');
 if(!canvas){host.innerHTML='';canvas=document.createElement('canvas');canvas.id='scene-canvas';canvas.style.width='100%';canvas.style.height='100%';canvas.style.display='block';host.appendChild(canvas);fallback.canvas=canvas;
 let down=null;
 canvas.addEventListener('pointerdown',e=>{const r=canvas.getBoundingClientRect();down={x:e.clientX,y:e.clientY,rx:e.clientX-r.left,ry:e.clientY-r.top,moved:false};canvas.setPointerCapture(e.pointerId)});
 canvas.addEventListener('pointermove',e=>{if(!down)return;const dx=e.clientX-down.x,dy=e.clientY-down.y;if(Math.abs(dx)+Math.abs(dy)>2)down.moved=true;fallback.yaw+=dx*.006;fallback.pitch=Math.max(-1.0,Math.min(1.18,fallback.pitch+dy*.004));down.x=e.clientX;down.y=e.clientY;renderCanvas()});
 canvas.addEventListener('pointerup',e=>{if(!down)return;const old=down;down=null;if(old.moved)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;let found=null,dist=18;for(const q of fallback.coords){const d=Math.hypot(q.x-x,q.y-y);if(d<dist){dist=d;found=q.id}}if(found)selectCamera(found)});
 canvas.addEventListener('wheel',e=>{e.preventDefault();fallback.zoom=Math.min(3.3,Math.max(.45,fallback.zoom*(e.deltaY<0?1.09:.91)));renderCanvas()},{passive:false});
 window.addEventListener('resize',()=>{if(fallback.canvas)renderCanvas()});
 }
 const W=Math.max(500,Math.round(host.clientWidth||800)),H=Math.max(350,Math.round(host.clientHeight||700)),dpr=window.devicePixelRatio||1;
 if(canvas.width!==Math.round(W*dpr)||canvas.height!==Math.round(H*dpr)){canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr)}
 const ctx=canvas.getContext('2d');ctx.setTransform(dpr,0,0,dpr,0,0);
 const gr=ctx.createLinearGradient(0,0,W,H);gr.addColorStop(0,'#0f293c');gr.addColorStop(.6,'#0a1a2a');gr.addColorStop(1,'#07131e');ctx.fillStyle=gr;ctx.fillRect(0,0,W,H);
 const proj=p=>canvasProject(p,W,H);
 function stroke(points,color,width=2,dash=[]){const good=points.filter(p=>p&&p.every(Number.isFinite));if(good.length<2)return;ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();let penDown=false;for(const p of points){if(!p||p.some(x=>!Number.isFinite(x))){penDown=false;continue}const t=proj(p);if(penDown)ctx.lineTo(t[0],t[1]);else{ctx.moveTo(t[0],t[1]);penDown=true}}ctx.stroke();ctx.setLineDash([])}
 function point(p,color,size=4,caption=''){const q=proj(p);ctx.fillStyle=color;ctx.beginPath();ctx.arc(q[0],q[1],size,0,Math.PI*2);ctx.fill();if(caption){ctx.fillStyle='#b8d5e4';ctx.font='10px Segoe UI, sans-serif';ctx.fillText(caption,q[0]+9,q[1]-9)}return q}
 // Reference grid in nonmetric/local coordinates, no scale claim.
 if(M.synthetic===true){for(let x=-4;x<=4;x+=1){stroke([[x,-3,0],[x,3,0]],'#2b4557',.65)}
 for(let y=-3;y<=3;y+=1){stroke([[-4,y,0],[4,y,0]],'#2b4557',.65)}}
 const mesh=A.mesh;
 if(state.layers.point_cloud){for(const p of (A.point_cloud?.points||[]))point(p,'#caa9df',1.5)}
 if(state.layers.mesh||state.layers.unknown){let f=mesh.faces.map((face,i)=>({pts:face.map(idx=>mesh.vertices[idx]),attr:mesh.face_attributes?.[i]||{},id:i}));
 f.sort((a,b)=>(b.pts.reduce((s,p)=>s+proj(p)[2],0)/3)-(a.pts.reduce((s,p)=>s+proj(p)[2],0)/3));
 for(const face of f){const unknown=face.attr.state==='unknown';if((unknown&&!state.layers.unknown)||(!unknown&&!state.layers.mesh))continue;
  const pts=face.pts.map(proj);ctx.beginPath();ctx.moveTo(pts[0][0],pts[0][1]);pts.slice(1).forEach(q=>ctx.lineTo(q[0],q[1]));ctx.closePath();ctx.globalAlpha=.20;ctx.fillStyle=scalarColor(face.attr[state.mode==='coverage'?'coverage':'confidence'],state.mode);ctx.fill();ctx.globalAlpha=.50;ctx.strokeStyle=unknown?'#be7baf':'#567d87';ctx.lineWidth=.8;ctx.stroke();ctx.globalAlpha=1;
 }}
 if(state.layers.planes){for(const p of A.planes.planes){if(p.vertices?.length>2)stroke([...p.vertices,p.vertices[0]],scalarColor(p[state.mode==='coverage'?'coverage':'confidence'],state.mode),2,[5,3])}}
 if(state.layers.edges||state.layers.unknown){for(const e of A.edges.edges){const u=e.state==='candidate_extrapolated'||e.state==='unknown';if((u&&!state.layers.unknown)||(!u&&!state.layers.edges))continue;stroke([e.start,e.end],scalarColor(e.confidence,state.mode),u?2.2:3.1,u?[4,4]:[])}}
 if(state.layers.bim&&mesh.bim_overlay?.vertices){const bim=mesh.bim_overlay;const set=new Set();for(const t of bim.faces){for(let i=0;i<3;i++){const a=t[i],b=t[(i+1)%3];set.add([Math.min(a,b),Math.max(a,b)].join(':'))}}for(const k of set){const [a,b]=k.split(':').map(Number);stroke([bim.vertices[a],bim.vertices[b]],'#b6a4e9',1.4,[4,5])}}
 if(state.layers.corners||state.layers.unknown){for(const c of A.corners.corners){const u=c.confidence===null||c.state==='unknown';if((u&&!state.layers.unknown)||(!u&&!state.layers.corners))continue;point(c.xyz,scalarColor(c.confidence,state.mode),u?5:4)}}
 const cams=A.camera_poses.cameras;
 if(state.layers.path)stroke(cams.map(c=>c.camera_to_world.map(r=>r[3]).slice(0,3)),'#7fa6fa',2.8,[6,5]);
 fallback.coords=[];
 if(state.layers.cameras){for(const c of cams){const m=c.camera_to_world,cen=world(m,[0,0,0]),z=.55,dx=z*c.width/2/c.fx,dy=z*c.height/2/c.fy;
  const far=[world(m,[-z*c.cx/c.fx,-z*c.cy/c.fy,z]),world(m,[z*(c.width-c.cx)/c.fx,-z*c.cy/c.fy,z]),world(m,[z*(c.width-c.cx)/c.fx,z*(c.height-c.cy)/c.fy,z]),world(m,[-z*c.cx/c.fx,z*(c.height-c.cy)/c.fy,z])];
  for(const corner of far)stroke([cen,corner],'#81ddca',1.1);
  stroke([...far,far[0]],'#81ddca',1.3);
  const q=point(cen,c.camera_id===state.selectedCamera?'#fbd08d':'#7de3d0',5,c.camera_id.replace('CAM-','C'));
  fallback.coords.push({id:c.camera_id,x:q[0],y:q[1]});
 }}
 ctx.font='11px Segoe UI, sans-serif';ctx.fillStyle='#7fa2b4';ctx.fillText('CANVAS 3D · WEBGL UNAVAILABLE IN THIS ENVIRONMENT',19,H-20);
 ctx.fillStyle='#608fa4';ctx.fillText('X / Y / Z = '+M.coordinate_frame+' • '+M.units+' • NONMETRIC',19,26);
}

function init(){
 $('run-name').textContent=M.run_id;
 $('badge').textContent=M.synthetic?'SYNTHETIC • NO FIELD METRICS':B.deviation_enabled?'VERIFIED METRIC INPUT':'METRIC UNVERIFIED';
 $('frame-coordinates').textContent='FRAME  '+M.coordinate_frame+'  /  '+M.units;
 $('metric-state').textContent=B.deviation_enabled?'METRIC DEVIATION AVAILABLE':'METRIC SCALE NOT VERIFIED · NO DEVIATION VALUES';
 $('integrity-note').textContent=(M.synthetic?'SYNTHETIC: coverage/confidence are illustrative; no physical claim. ':'Source image SHA256 verified. Unknown is not zero execution. ')+(M.geometry_alignment_status?('Geometry / camera: '+M.geometry_alignment_status+'. '):'')+(A.camera_poses.quality?.frustum_distortion_approximation?'Distorted camera frusta are approximate; source images remain unchanged.':'');
 $('bim-status').textContent=B.deviation_enabled?'Verified alignment and independent controls supplied.':'Overlay is illustrative/unregistered until survey controls establish metric scale and alignment. Deviation is disabled ('+B.deviation_reason+').';
 $('camera-count').textContent=A.camera_poses.cameras.length;
 $('cameras').innerHTML='';A.camera_poses.cameras.forEach(c=>{
  const b=document.createElement('button');b.className='camera-btn';b.dataset.camera=c.camera_id;
  const ico=document.createElement('span');ico.className='camera-icon';ico.textContent='◇';
  const txt=document.createElement('span');const strong=document.createElement('b');strong.textContent=c.camera_id;const sub=document.createElement('small');sub.textContent=c.frame_id+' · '+c.width+'×'+c.height;txt.append(strong,sub);b.append(ico,txt);b.addEventListener('click',()=>selectCamera(c.camera_id));$('cameras').appendChild(b);
 });
 document.querySelectorAll('.mode').forEach(btn=>btn.addEventListener('click',()=>{state.mode=btn.dataset.mode;document.querySelectorAll('.mode').forEach(x=>x.classList.toggle('active',x===btn));render();}));
 const corners=A.corners.corners||[],faces=A.mesh.faces||[],edges=A.edges.edges||[];
 const rows=[['Surface facets',faces.length],['Plane patches',A.planes.planes.length],['Edge candidates',edges.length],['Corner candidates',corners.length],['Unknown features',corners.filter(c=>c.confidence===null).length+edges.filter(e=>e.state==='unknown').length]];
 $('evidence').innerHTML='';rows.forEach(([label,value])=>{const el=document.createElement('div');el.className='ev-row';const t=document.createElement('span');t.textContent=label;const n=document.createElement('strong');n.textContent=value;el.append(t,n);$('evidence').appendChild(el);});
 refreshLayers();render(false).then(()=>{selectCamera(A.camera_poses.cameras[0].camera_id);if(HAS_WEBGL)$('scene').on('plotly_click',event=>{const p=event?.points?.[0];const id=p?.customdata;if(typeof id==='string'&&A.camera_poses.cameras.some(c=>c.camera_id===id))selectCamera(id);});window.__VIEWER_READY__=true;}).catch(e=>{HAS_WEBGL=false;fallback.canvas=null;renderCanvas();selectCamera(A.camera_poses.cameras[0].camera_id);
 $('integrity-note').textContent='WebGL renderer unavailable; Canvas fallback active. '+String(e.message||e);
 window.__VIEWER_READY__=true;});
}
window.MyCONViewer={selectCamera,setLayer,orbit,state,manifest:M,cameraMarkers:()=>fallback.coords.map(x=>({...x}))};
window.__VIEWER_READY__=false;
init();
})();
