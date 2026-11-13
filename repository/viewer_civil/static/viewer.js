/* MyCON read-only Civil Viewer, Plotly/WebGL with Canvas fallback. No network calls. */
"use strict";
(() => {
  const payload = JSON.parse(document.getElementById("mycon-payload").textContent);
  const A = payload.assets, M = payload.manifest, $ = id => document.getElementById(id);
  const state = {mode: "confidence", selectedCamera: null,
    layers: {mesh: true, planes: false, edges: true, corners: true, unknown: true,
             cameras: true, path: true, point_cloud: true, bim: false, deviation: false}};
  const names = {mesh:"Reconstruction mesh", planes:"Finite surface patches", edges:"Structural edge candidates",
    corners:"Corner candidates", unknown:"Unknown / insufficient evidence", cameras:"Calibrated camera frustums",
    path:"Camera capture trajectory", point_cloud:"Point samples", bim:"Reference BIM overlay",
    deviation:"Verified metric deviation"};
  const palette = {mesh:"#55a88b", planes:"#5dbbad", edges:"#e8b76e", corners:"#83cbdb",
    unknown:"#c280ab", cameras:"#73d5c5", path:"#9da7ff", point_cloud:"#d0a3df",
    bim:"#b9a4e7", deviation:"#e98b72"};
  const view = {yaw:.65, pitch:.4, zoom:1, panX:0, panY:0, canvas:null, markers:[]};
  let hasWebGL = false, usingCanvas = false;
  try {
    let c = document.createElement("canvas");
    hasWebGL = !!(c.getContext("webgl2") || c.getContext("webgl") || c.getContext("experimental-webgl"));
  } catch (e) { hasWebGL = false; }
  const cameras = A.camera_poses.cameras, frames = new Map(A.frame_manifest.frames.map(f => [f.frame_id,f]));
  const world = (m,p) => [0,1,2].map(i=>m[i][0]*p[0]+m[i][1]*p[1]+m[i][2]*p[2]+m[i][3]);
  const positions = () => cameras.map(c => world(c.camera_to_world,[0,0,0]));
  const frustum = c => {
    const z = .45, dx = .45*c.width/(2*c.fx), dy = .45*c.height/(2*c.fy), m=c.camera_to_world;
    const o=world(m,[0,0,0]), q=[[-dx,-dy,z],[dx,-dy,z],[dx,dy,z],[-dx,dy,z]].map(p=>world(m,p));
    return {origin:o, corners:q};
  };
  const cssColor = (value, status) => {
    if (value === null || value === undefined || !Number.isFinite(Number(value)) || status==="unknown"
        || status==="candidate_extrapolated") return palette.unknown;
    if (state.mode === "neutral") return "#8ca5af";
    let v = Math.max(0,Math.min(1,Number(value)));
    return v>.8?"#5bc5a0":v>.55?"#d9bb69":v>.3?"#dd9870":"#bd7181";
  };
  function setBadge() {
    $("run-name").textContent = M.run_id;
    $("source-status").textContent = M.synthetic ? "SYNTHETIC • NOT FIELD EVIDENCE" :
      payload.deviation_enabled ? "METRIC GATE VERIFIED" : "GEOMETRY • METRIC NOT VERIFIED";
    $("metric-status").textContent = payload.deviation_enabled ?
      "Metric deviation available only for independently verified source faces." :
      "Numerical deviation DISABLED: " + payload.deviation_reason + ". Unknown does not imply zero progress.";
    $("frame-coordinates").textContent = "FRAME " + M.coordinate_frame + " / UNITS " + M.units;
    $("evidence").textContent = "Surfaces " + (A.planes.planes||[]).length + " • edges " +
      (A.edges.edges||[]).length + " • corners " + (A.corners.corners||[]).length +
      " • cameras " + cameras.length;
    $("uncertainty").textContent = M.synthetic ?
      "DEMO: confidence and coverage are illustrative, not calibrated." :
      "Visual coverage/quality are not physical progress or surveyed geometry.";
  }
  function selectCamera(id) {
    const c=cameras.find(x=>x.camera_id===id);
    if(!c) throw new Error("Camera does not exist: "+id);
    const f=frames.get(c.frame_id);
    if(!f) throw new Error("Camera frame missing: "+id);
    state.selectedCamera=id;
    $("selected-camera").textContent=id;
    $("selected-frame").src=f.preview_path;
    $("selected-frame").alt="Recorded frame for "+id;
    $("frame-meta").textContent=f.frame_id+" • "+c.width+"×"+c.height+" • "+(f.quality?.state||"source verified by SHA256");
    $("camera-position").textContent=c.camera_to_world.slice(0,3).map(r=>r[3].toFixed(3)).join(", ")+" ("+M.units+")";
    document.querySelectorAll("button[data-camera]").forEach(b=>b.classList.toggle("active",b.dataset.camera===id));
    if(usingCanvas) drawCanvas();
    return f.frame_id;
  }
  function refreshToggles() {
    const node=$("layers");node.replaceChildren();
    Object.keys(names).forEach(k=>{
      const label=document.createElement("label");label.className="layer";
      const span=document.createElement("span");span.textContent=names[k];
      const dot=document.createElement("i");dot.style.background=palette[k];span.prepend(dot);
      const input=document.createElement("input");input.type="checkbox";input.dataset.layer=k;
      input.checked=state.layers[k];input.disabled=(k==="deviation"&&!payload.deviation_enabled);
      input.title=input.disabled?payload.deviation_reason:"";
      input.addEventListener("change",()=>setLayer(k,input.checked));
      label.append(span,input);node.append(label);
    });
  }
  function setLayer(key,value) {
    if(!Object.hasOwn(state.layers,key)) throw new Error("Unknown layer "+key);
    if(key==="deviation"&&!payload.deviation_enabled) return false;
    state.layers[key]=Boolean(value);
    refreshToggles(); render();
    return true;
  }
  function traces() {
    const t=[], add=(key,trace)=>{ if(state.layers[key]) t.push({...trace,showlegend:false}); };
    const line=(p,color,width=3,extra={})=>({type:"scatter3d",mode:"lines",
      x:p.map(a=>a?.[0]??NaN), y:p.map(a=>a?.[1]??NaN), z:p.map(a=>a?.[2]??NaN),
      line:{color,width},hoverinfo:"skip",...extra});
    const mesh=A.mesh, verts=mesh.vertices||[], faces=mesh.faces||[], attr=mesh.face_attributes||[];
    const x=verts.map(v=>v[0]),y=verts.map(v=>v[1]),z=verts.map(v=>v[2]);
    const faceColor=i=>cssColor(attr[i]?.[state.mode==="coverage"?"coverage":"confidence"],attr[i]?.state);
    if(faces.length) add("mesh",{type:"mesh3d",x,y,z,i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),
      k:faces.map(f=>f[2]),facecolor:faces.map((_,i)=>faceColor(i)),opacity:.3,flatshading:true,hoverinfo:"skip"});
    for (const p of A.planes.planes||[]) if(p.vertices?.length>=3) {
      const v=p.vertices;
      add(p.state==="unknown"?"unknown":"planes",line([...v,v[0]],cssColor(p[state.mode==="coverage"?"coverage":"confidence"],p.state),4));
    }
    for(const e of A.edges.edges||[]) add(
      e.state==="unknown"||e.state==="candidate_extrapolated"?"unknown":"edges",
      line([e.start,e.end],cssColor(e.confidence,e.state),5,{text:[e.id,e.id],hoverinfo:"text"}));
    for(const c of A.corners.corners||[]) add(
      c.state==="unknown"||c.confidence==null?"unknown":"corners",
      {type:"scatter3d",mode:"markers",x:[c.xyz[0]],y:[c.xyz[1]],z:[c.xyz[2]],
       marker:{color:cssColor(c.confidence,c.state),size:6},text:[c.id],hoverinfo:"text"});
    const pos=positions();
    add("path",line(pos,palette.path,4));
    for(const c of cameras) {
      const f=frustum(c), pts=[];
      for(const corner of f.corners) pts.push(f.origin,corner,[NaN,NaN,NaN]);
      add("cameras",line([...pts,...f.corners,f.corners[0]],palette.cameras,3));
    }
    add("cameras",{type:"scatter3d",mode:"markers+text",x:pos.map(v=>v[0]),
      y:pos.map(v=>v[1]),z:pos.map(v=>v[2]),
      marker:{size:7,color:"#79eed1"},text:cameras.map(c=>c.camera_id),
      textposition:"top center",textfont:{size:10,color:"#c2ded9"},
      customdata:cameras.map(c=>c.camera_id),hovertemplate:"%{customdata}<extra></extra>"});
    if(A.point_cloud?.points?.length) add("point_cloud",{type:"scatter3d",mode:"markers",
      x:A.point_cloud.points.map(p=>p[0]),y:A.point_cloud.points.map(p=>p[1]),
      z:A.point_cloud.points.map(p=>p[2]),marker:{size:2,color:palette.point_cloud},hoverinfo:"skip"});
    const bim=mesh.bim_overlay;
    if(bim?.vertices?.length&&bim?.faces?.length) {
      const edges=new Set();
      bim.faces.forEach(f=>{for(let k=0;k<3;k++){let a=f[k],b=f[(k+1)%3];edges.add([Math.min(a,b),Math.max(a,b)].join(","));}});
      for (const e of edges) {const [a,b]=e.split(",").map(Number);
        add("bim",line([bim.vertices[a],bim.vertices[b]],palette.bim,2));}
    }
    if(payload.deviation_enabled&&mesh.deviations?.per_face_m){
      const v=mesh.deviations.per_face_m, finite=v.filter(Number.isFinite);
      const max=Math.max(.000001,...finite.map(Math.abs));
      add("deviation",{type:"mesh3d",x,y,z,i:faces.map(f=>f[0]),j:faces.map(f=>f[1]),
        k:faces.map(f=>f[2]),facecolor:v.map(q=>q==null?palette.unknown:q < -max*.35?"#69b5db":q>max*.35?"#e6877b":"#74c8aa"),
        opacity:.7,hoverinfo:"skip"});
    }
    return t;
  }
  function project(p,W,H) {
    const x=p[0],y=p[1],z=p[2]-1.5,ca=Math.cos(view.yaw),sa=Math.sin(view.yaw);
    const a=ca*x-sa*y,b=sa*x+ca*y,cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);
    const k=1+.018*(b*cp+z*sp),scale=Math.min(W/12,H/8)*view.zoom;
    return [W/2+view.panX+a*scale/k,H/2+view.panY+(b*sp-z*cp)*scale/k,b*cp+z*sp];
  }
  function makeCanvas() {
    usingCanvas=true;
    if(view.canvas)return;
    const host=$("scene");host.replaceChildren();
    const canvas=document.createElement("canvas");canvas.id="scene-canvas";
    canvas.style.width="100%";canvas.style.height="100%";canvas.style.touchAction="none";
    host.append(canvas);view.canvas=canvas;
    let down=null;
    canvas.addEventListener("pointerdown",e=>{down={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,shift:e.shiftKey};canvas.setPointerCapture(e.pointerId);});
    canvas.addEventListener("pointermove",e=>{if(!down)return;const dx=e.clientX-down.x,dy=e.clientY-down.y;
      if(down.shift){view.panX+=dx;view.panY+=dy;}else{view.yaw+=dx*.008;view.pitch=Math.max(-1,Math.min(1.2,view.pitch+dy*.005));}
      down.x=e.clientX;down.y=e.clientY;drawCanvas();});
    canvas.addEventListener("pointerup",e=>{if(!down)return;const moved=Math.hypot(e.clientX-down.startX,e.clientY-down.startY);down=null;
      if(moved>5)return;const rect=canvas.getBoundingClientRect(),x=e.clientX-rect.left,y=e.clientY-rect.top;
      const hit=view.markers.find(m=>Math.hypot(m.x-x,m.y-y)<16);if(hit)selectCamera(hit.id);});
    canvas.addEventListener("wheel",e=>{e.preventDefault();view.zoom=Math.max(.25,Math.min(5,view.zoom*(e.deltaY<0?1.12:.89)));drawCanvas();},{passive:false});
    window.addEventListener("resize",()=>{if(view.canvas)drawCanvas();});
  }
  function drawCanvas() {
    makeCanvas();
    const canvas=view.canvas, host=$("scene"),W=Math.max(300,host.clientWidth),H=Math.max(300,host.clientHeight),dpr=window.devicePixelRatio||1;
    canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);
    const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);
    const bg=ctx.createLinearGradient(0,0,W,H);bg.addColorStop(0,"#102b3b");bg.addColorStop(1,"#07111d");
    ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
    const pr=p=>project(p,W,H);
    function line(pts,color,width=2,dash=[]) {
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);ctx.beginPath();
      pts.forEach((p,i)=>{const q=pr(p);i?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1]);});
      ctx.stroke();ctx.setLineDash([]);
    }
    function dot(pos,color,size=4,text) {
      const p=pr(pos);ctx.fillStyle=color;ctx.beginPath();ctx.arc(p[0],p[1],size,0,2*Math.PI);ctx.fill();
      if(text){ctx.fillStyle="#c5dde3";ctx.font="11px sans-serif";ctx.fillText(text,p[0]+8,p[1]-6);}
      return p;
    }
    if(state.layers.mesh) {
      const v=A.mesh.vertices;A.mesh.faces.forEach((face,i)=>{
        const attr=A.mesh.face_attributes?.[i]||{},q=face.map(k=>pr(v[k]));
        ctx.beginPath();ctx.moveTo(q[0][0],q[0][1]);for(let k=1;k<q.length;k++)ctx.lineTo(q[k][0],q[k][1]);ctx.closePath();
        ctx.globalAlpha=.23;ctx.fillStyle=cssColor(attr[state.mode==="coverage"?"coverage":"confidence"],attr.state);ctx.fill();ctx.globalAlpha=1;
        ctx.strokeStyle="#375569";ctx.stroke();
      });
    }
    if(state.layers.planes) for(const p of A.planes.planes||[])if(p.vertices?.length>2)
      line([...p.vertices,p.vertices[0]],cssColor(p[state.mode==="coverage"?"coverage":"confidence"],p.state),2,[4,3]);
    for(const e of A.edges.edges||[]) {
      const unknown=e.state==="unknown"||e.state==="candidate_extrapolated";
      if(state.layers[unknown?"unknown":"edges"])line([e.start,e.end],cssColor(e.confidence,e.state),unknown?2:3,unknown?[4,4]:[]);
    }
    for(const c of A.corners.corners||[]) {
      const unknown=c.state==="unknown"||c.confidence==null;
      if(state.layers[unknown?"unknown":"corners"])dot(c.xyz,cssColor(c.confidence,c.state),unknown?5:4);
    }
    if(state.layers.path)line(positions(),palette.path,2,[4,5]);
    if(state.layers.bim&&A.mesh.bim_overlay?.vertices){
      let b=A.mesh.bim_overlay,seen=new Set();
      for(const f of b.faces)for(let i=0;i<3;i++){const a=f[i],z=f[(i+1)%3],k=[Math.min(a,z),Math.max(a,z)].join("-");
        if(!seen.has(k)){seen.add(k);line([b.vertices[a],b.vertices[z]],palette.bim,1,[3,3]);}}
    }
    if(state.layers.point_cloud)for(const p of A.point_cloud?.points||[])dot(p,palette.point_cloud,1);
    view.markers=[];
    if(state.layers.cameras)for(const c of cameras){
      const f=frustum(c);for(const q of f.corners)line([f.origin,q],palette.cameras,1);
      line([...f.corners,f.corners[0]],palette.cameras,1.5);
      const q=dot(f.origin,c.camera_id===state.selectedCamera?"#ffc97c":palette.cameras,5,c.camera_id);
      view.markers.push({id:c.camera_id,x:q[0],y:q[1]});
    }
    ctx.font="11px sans-serif";ctx.fillStyle="#94b7c7";
    ctx.fillText("CANVAS FALLBACK • RECONSTRUCTION UNITS • NO METRIC CLAIM",14,H-18);
  }
  let plotlyStarted=false;
  function render() {
    if(!hasWebGL||typeof Plotly==="undefined"){drawCanvas();return Promise.resolve();}
    const opts={paper_bgcolor:"rgba(0,0,0,0)",margin:{l:0,r:0,t:0,b:0},showlegend:false,
      scene:{aspectmode:"data",bgcolor:"rgba(0,0,0,0)",
      xaxis:{title:"X",color:"#86a6b5"},yaxis:{title:"Y",color:"#86a6b5"},
      zaxis:{title:"Z",color:"#86a6b5"}}};
    let job=plotlyStarted?Plotly.react("scene",traces(),opts,{responsive:true,displaylogo:false}):
      Plotly.newPlot("scene",traces(),opts,{responsive:true,displaylogo:false});
    plotlyStarted=true;
    return job.then(()=>{const node=$("scene");
      if(!node.__cameraClickBound){node.__cameraClickBound=true;node.on("plotly_click",ev=>{
        const id=ev?.points?.[0]?.customdata;if(typeof id==="string"&&cameras.some(c=>c.camera_id===id))selectCamera(id);
      });}
    }).catch(e=>{console.warn("WebGL rejected; Canvas fallback",e);hasWebGL=false;drawCanvas();});
  }
  function orbit(t) {
    if(usingCanvas){view.yaw=t*.5;drawCanvas();return Promise.resolve();}
    return Plotly.relayout("scene",{"scene.camera.eye":{x:2*Math.cos(t*.45),y:2*Math.sin(t*.45),z:1.2}});
  }
  function init() {
    setBadge();refreshToggles();
    const list=$("camera-list");
    for(const c of cameras){const btn=document.createElement("button");btn.dataset.camera=c.camera_id;
      btn.textContent=c.camera_id+" / "+c.frame_id;
      btn.addEventListener("click",()=>selectCamera(c.camera_id));list.append(btn);}
    document.querySelectorAll("button[data-mode]").forEach(btn=>btn.addEventListener("click",()=>{
      state.mode=btn.dataset.mode;
      document.querySelectorAll("button[data-mode]").forEach(b=>b.classList.toggle("active",b===btn));
      render();
    }));
    render().then(()=>{if(cameras.length)selectCamera(cameras[0].camera_id);window.__VIEWER_READY__=true;})
      .catch(e=>{window.__VIEWER_ERROR__=String(e);});
  }
  window.MyCONViewer={state,selectCamera,setLayer,orbit,cameraMarkers:()=>view.markers.map(m=>({...m}))};
  window.__VIEWER_READY__=false;window.__VIEWER_ERROR__=null;init();
})();
