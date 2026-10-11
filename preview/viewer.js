import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";

const $ = id => document.getElementById(id);
const defaultColors = ["#adb8c6","#ae7bff","#3bcbec","#298df2","#ffb75e","#f28dc2","#67db8c"];
const validPalette = values => Array.isArray(values) &&
    values.length === defaultColors.length &&
    values.every(color => typeof color === "string" && /^#[0-9a-f]{6}$/i.test(color));
function loadPalette(serverColors) {
    if (validPalette(serverColors)) return [...serverColors];
    try {
        const cached = JSON.parse(localStorage.getItem("silksteel.feature-colors.v1"));
        if (validPalette(cached)) return [...cached];
    } catch {}
    return [...defaultColors];
}
async function loadPreviewMetadata() {
    const indicator = $("progress");
    for (;;) {
        const response = await fetch("/meta.json", {cache:"no-store"});
        if (!response.ok) {
            indicator.textContent = "Preview-Server nicht erreichbar (HTTP "+response.status+")";
            throw new Error(indicator.textContent);
        }
        const metadata = await response.json();
        if (metadata.status === "error") {
            indicator.hidden = false;
            indicator.textContent = "G-Code-Analyse fehlgeschlagen: "+metadata.message+
                " · Du kannst eine andere Datei öffnen oder ins Fenster ziehen.";
            return metadata;
        }
        if (metadata.status === "ready" || !metadata.status) return metadata;
        indicator.hidden = false;
        indicator.textContent = (metadata.message || "G-Code wird analysiert…") +
            " " + (metadata.progress || 0) + "% · " +
            (metadata.lines || 0).toLocaleString("de-DE") + " Zeilen";
        await new Promise(resolve => setTimeout(resolve, 500));
    }
}
let layout = await loadPreviewMetadata();
const canvasHost = $("viewport");
const scene = new THREE.Scene();
scene.background = new THREE.Color("#0c1119");
const camera = new THREE.PerspectiveCamera(45, 1, .02, 100000);
const renderer = new THREE.WebGLRenderer({antialias:true, powerPreference:"high-performance"});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
canvasHost.insertBefore(renderer.domElement, canvasHost.firstChild);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .075;
const grid = new THREE.GridHelper(300, 60, "#4d6076", "#263144");
scene.add(grid);
const modelSelect = $("model"), slider = $("layer");
let model = null, data = null, hopData = null, cache = new Map(), hopCache = new Map(), lastPick = null;
let activeLayers = [], currentTime = 0, visibleCount = 0, playing = false, previousFrame = 0;
let playbackMode = "print", selectedLayer = 0, visualLayer = 0, markerKey = "", renderLayers = [];
const TIMELINE_STEPS = 1000000;
const SEGMENT_SIZE = 48, HOP_SIZE = 40;
const visibleRoles = new Set(layout.roles.map((_, i) => i));
const colors = loadPalette(layout.colors);
const rgb = colors.map(c => new THREE.Color(c));
const pickers = [];
async function savePalette() {
    const palette = [...colors];
    try { localStorage.setItem("silksteel.feature-colors.v1", JSON.stringify(palette)); } catch {}
    try {
        const response = await fetch("/settings/colors", {
            method:"POST",headers:{"Content-Type":"application/json"},
            body:JSON.stringify({colors:palette})
        });
        if (!response.ok) throw Error("HTTP "+response.status);
    } catch (error) {
        console.warn("Feature-Farben konnten nicht dauerhaft gespeichert werden:", error);
    }
}
function applyColor(index, hex) {
    if (!/^#[0-9a-f]{6}$/i.test(hex)) return;
    colors[index] = hex;
    rgb[index].set(hex);
    for (const group of cache.values()) {
        for (const mesh of group.children) {
            if (mesh.userData.role === index)
                mesh.material.uniforms.uColor.value.copy(rgb[index]);
        }
    }
    if (headMesh.visible) {
        const current = Number(headMesh.userData.role);
        if (current === index)
            headMesh.material.uniforms.uColor.value.copy(rgb[index]);
    }
    if (pickers[index]) pickers[index].value = hex;
}

function refreshModelOptions() {
    modelSelect.replaceChildren();
    layout.models.forEach((item,i) => {
        const opt=document.createElement("option");
        opt.value=String(i);
        opt.textContent=item.name;
        modelSelect.append(opt);
    });
}
refreshModelOptions();
layout.roles.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "feature-row";
    const check = document.createElement("input");
    check.type = "checkbox"; check.checked = true;
    check.setAttribute("aria-label", name+" anzeigen");
    check.addEventListener("change", () => {
        if (check.checked) visibleRoles.add(i); else visibleRoles.delete(i);
        updateVisibility();
    });
    const picker = document.createElement("input");
    picker.type = "color";
    picker.value = colors[i];
    picker.title = name+"-Farbe ändern";
    picker.setAttribute("aria-label", name+"-Farbe wählen");
    picker.addEventListener("input", () => applyColor(i,picker.value));
    picker.addEventListener("change", savePalette);
    pickers.push(picker);
    const title = document.createElement("span");
    title.textContent = name;
    row.append(check,picker,title);
    $("legend").append(row);
});
$("resetColors").addEventListener("click", () => {
    defaultColors.forEach((hex,index) => applyColor(index,hex));
    savePalette();
});

// Unit cross section consists of TWO semicircles and horizontal top/bottom.
// Width is NOT applied by scaling a circle: shader keeps the end radii = h/2.
function sectionMesh() {
    const vertices = [], indexes = [];
    const append = (side, cos, sin, end, cap) => {
        vertices.push(side, cos, sin, end, cap);
        return vertices.length / 5 - 1;
    };
    const sides = [];
    const steps = 8;
    for (let i = 0; i <= steps; i++) {
        const a = -Math.PI / 2 + Math.PI * i / steps;
        sides.push([1, Math.cos(a), Math.sin(a)]);
    }
    for (let i = 0; i <= steps; i++) {
        const a = Math.PI / 2 + Math.PI * i / steps;
        sides.push([-1, Math.cos(a), Math.sin(a)]);
    }
    const n = sides.length;
    const rings = [[],[]], caps = [[],[]];
    for (let end = 0; end < 2; end++) {
        for (const [side,cos,sin] of sides) {
            rings[end].push(append(side,cos,sin,end,0));
            caps[end].push(append(side,cos,sin,end,1));
        }
    }
    for (let i = 0; i < n; i++) {
        const j = (i+1)%n;
        indexes.push(rings[0][i],rings[1][i],rings[0][j],
                     rings[1][i],rings[1][j],rings[0][j]);
    }
    for(let end=0;end<2;end++){
        const center = append(0,0,0,end,2);
        for(let i=0;i<n;i++){
            const j = (i+1)%n;
            if(end===0) indexes.push(center,caps[end][i],caps[end][j]);
            else indexes.push(center,caps[end][j],caps[end][i]);
        }
    }
    const geometry = new THREE.InstancedBufferGeometry();
    const coords = new Float32Array(vertices);
    for (let i=0; i<5; i++) {
        const attribute = new THREE.Float32BufferAttribute(coords.filter((_, j) => j%5===i), 1);
        geometry.setAttribute(["aSide","aCos","aSin","aEnd","aCap"][i],attribute);
    }
    geometry.setIndex(indexes);
    return geometry;
}
const base = sectionMesh();
const vertexShader = `
precision highp float;
attribute float aSide, aCos, aSin, aEnd, aCap;
attribute vec3 iStart, iStop;
attribute vec2 iSize;
uniform float uAlpha;
varying vec3 vNormal;
varying vec3 vPosition;
void main(){
  vec3 d=iStop-iStart;
  float len=length(d.xy);
  vec3 perpendicular=len>0.0000001 ? vec3(-d.y/len,0.0,d.x/len) : vec3(1.0,0.0,0.0);
  float h=iSize.y;
  float w=max(iSize.x,h);
  float lateral=aSide*(w-h)*0.5 + aCos*h*0.5;
  vec3 origin=mix(iStart,iStop,aEnd);
  vec3 p=vec3(origin.x,origin.z-h*0.5,origin.y)
      + perpendicular*lateral + vec3(0.0,h*0.5*aSin,0.0);
  vec3 tangent=normalize(vec3(d.x,d.z,d.y));
  vNormal=aCap>0.5 ? (aEnd>0.5 ? tangent : -tangent)
                      : normalize(perpendicular*aCos+vec3(0.0,aSin,0.0));
  vPosition=p;
  gl_Position=projectionMatrix*modelViewMatrix*vec4(p,1.0);
}
`;
const fragmentShader = `
precision highp float;
uniform vec3 uColor;
uniform float uAlpha;
varying vec3 vNormal;
varying vec3 vPosition;
void main(){
 vec3 light=normalize(vec3(-0.45,0.85,0.5));
 float diffuse=max(dot(normalize(vNormal),light),0.0);
 float shade=0.47+0.53*diffuse;
 gl_FragColor=vec4(uColor*shade,uAlpha);
 #include <tonemapping_fragment>
 #include <colorspace_fragment>
}
`;
const newMaterial = (role, faded) => new THREE.ShaderMaterial({
    vertexShader, fragmentShader, side:THREE.DoubleSide,
    uniforms:{uColor:{value:rgb[role].clone()},uAlpha:{value:faded?.19:1}},
    transparent:!!faded,depthWrite:!faded
});
function disposeGroup(group) {
    group.traverse(o => {
        if(o.isMesh){o.geometry.dispose();o.material.dispose();}
    });
}
function forgetLayers() {
    for (const group of cache.values()) {scene.remove(group);disposeGroup(group);}
    for (const group of hopCache.values()) {scene.remove(group);disposeHopGroup(group);}
    cache.clear();hopCache.clear();activeLayers=[];
    headMesh.visible=false; hopTip.visible=false;
    if(lastPick){scene.remove(lastPick);lastPick.geometry.dispose();lastPick.material.dispose();lastPick=null;}
    $("pickInfo").style.display="none";
}
function readRole(index) {
    return data.getUint32(index*SEGMENT_SIZE+32,true);
}
function readSegment(index) {
    const off=index*SEGMENT_SIZE;
    return {
      a:[data.getFloat32(off,true),data.getFloat32(off+4,true),data.getFloat32(off+8,true)],
      b:[data.getFloat32(off+12,true),data.getFloat32(off+16,true),data.getFloat32(off+20,true)],
      w:data.getFloat32(off+24,true),
      h:data.getFloat32(off+28,true),
      role:readRole(index),
      line:data.getUint32(off+36,true),
      start:data.getFloat32(off+40,true),
      end:data.getFloat32(off+44,true)
    };
}
function readHop(index) {
    const off = index*HOP_SIZE;
    return {
      a:[hopData.getFloat32(off,true),hopData.getFloat32(off+4,true),hopData.getFloat32(off+8,true)],
      b:[hopData.getFloat32(off+12,true),hopData.getFloat32(off+16,true),hopData.getFloat32(off+20,true)],
      kind:hopData.getUint32(off+24,true),line:hopData.getUint32(off+28,true),
      start:hopData.getFloat32(off+32,true),end:hopData.getFloat32(off+36,true)
    };
}
function completedBefore(n, t, endAt) {
    let lo=0,hi=n;
    while(lo<hi){const mid=(lo+hi)>>>1;if(endAt(mid)<=t)lo=mid+1;else hi=mid;}
    return lo;
}
function withinPrefix(indices, cutoff) {
    let lo=0,hi=indices.length;
    while(lo<hi){const mid=(lo+hi)>>>1;if(indices[mid]<cutoff)lo=mid+1;else hi=mid;}
    return lo;
}
const endAt = i => data.getFloat32(i*SEGMENT_SIZE+44,true);
const hopEndAt = i => hopData.getFloat32(i*HOP_SIZE+36,true);
function layerAtTime(seconds) {
    let low=0, high=model.layers.length;
    while (low<high) {
        const mid=(low+high)>>>1;
        if (model.layers[mid].timeStart<=seconds) low=mid+1;
        else high=mid;
    }
    return Math.max(0, Math.min(model.layers.length-1,low-1));
}
function layerRange(index=selectedLayer) {
    const layer=model.layers[index];
    return [Math.max(0,layer.timeStart ?? 0),Math.max(layer.timeStart ?? 0,layer.timeEnd ?? model.duration)];
}
function activeRange() {
    return playbackMode==="layer" ? layerRange() : [0,model.duration];
}
function formatTime(seconds) {
    const s=Math.max(0,Math.floor(seconds));
    return String(Math.floor(s/3600))+":"+String(Math.floor(s%3600/60)).padStart(2,"0")+":"+String(s%60).padStart(2,"0");
}
function setPlaying(next) {
    playing=next;
    previousFrame=0;
    $("playToggle").textContent=playing?"Ⅱ Pause":"▶ Abspielen";
}
function createLayer(index) {
    const info = model.layers[index];
    const buckets = Array.from({length:layout.roles.length},()=>[]);
    for(let i=info.start;i<info.start+info.count;i++){
        const role=readRole(i);
        if(role>=buckets.length)throw new Error("Ungültiger Extrusionstyp "+role+" in Segment "+i);
        buckets[role].push(i);
    }
    const group = new THREE.Group();
    const maxInstances = 28000;
    for(let role=0;role<buckets.length;role++){
        const indices=buckets[role];
        for(let offset=0;offset<indices.length;offset+=maxInstances){
            const items=indices.slice(offset,offset+maxInstances), n=items.length;
            const starts=new Float32Array(n*3),ends=new Float32Array(n*3),sizes=new Float32Array(n*2);
            for(let j=0;j<n;j++){
                const p=readSegment(items[j]);
                starts.set(p.a,j*3);ends.set(p.b,j*3);
                sizes.set([p.w,p.h],j*2);
            }
            const geometry=base.clone();
            geometry.setAttribute("iStart",new THREE.InstancedBufferAttribute(starts,3));
            geometry.setAttribute("iStop",new THREE.InstancedBufferAttribute(ends,3));
            geometry.setAttribute("iSize",new THREE.InstancedBufferAttribute(sizes,2));
            geometry.instanceCount=n;
            const mesh=new THREE.Mesh(geometry,newMaterial(role,false));
            mesh.frustumCulled=false;
            mesh.userData.role=role;
            mesh.userData.indices=Uint32Array.from(items);
            group.add(mesh);
        }
    }
    group.userData.layer=index;
    return group;
}
function getLayer(index) {
    let g=cache.get(index);
    if(!g){g=createLayer(index);cache.set(index,g);}
    return g;
}
function createHopLayer(index) {
    const info=model.layers[index], positions=new Float32Array(info.hopCount*6);
    for(let n=0;n<info.hopCount;n++){
        const hop=readHop(info.hopStart+n);
        positions.set([hop.a[0],hop.a[2],hop.a[1],hop.b[0],hop.b[2],hop.b[1]],n*6);
    }
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute("position",new THREE.BufferAttribute(positions,3));
    const lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:"#ff4655",depthTest:false,depthWrite:false}));
    lines.frustumCulled=false;lines.renderOrder=20;
    return lines;
}
function disposeHopGroup(group) {
    group.geometry.dispose();group.material.dispose();
}
function renderTip(mesh, p, fraction) {
    const stop=p.a.map((v,i)=>v+(p.b[i]-v)*fraction);
    const attrs=mesh.geometry.attributes;
    attrs.iStart.array.set(p.a);
    attrs.iStop.array.set(stop);
    attrs.iStart.needsUpdate=true;
    attrs.iStop.needsUpdate=true;
    if(attrs.iSize){attrs.iSize.array.set([p.w,p.h]);attrs.iSize.needsUpdate=true;}
    mesh.visible=true;
}
const headGeometry=base.clone();
headGeometry.setAttribute("iStart",new THREE.InstancedBufferAttribute(new Float32Array(3),3));
headGeometry.setAttribute("iStop",new THREE.InstancedBufferAttribute(new Float32Array(3),3));
headGeometry.setAttribute("iSize",new THREE.InstancedBufferAttribute(new Float32Array(2),2));
headGeometry.instanceCount=1;
const headMesh=new THREE.Mesh(headGeometry,newMaterial(0,false));
headMesh.frustumCulled=false;headMesh.visible=false;scene.add(headMesh);
const hopTipGeom=new THREE.BufferGeometry();
hopTipGeom.setAttribute("position",new THREE.BufferAttribute(new Float32Array(6),3));
const hopTip=new THREE.Line(hopTipGeom,new THREE.LineBasicMaterial({color:"#ff4655",depthTest:false,depthWrite:false}));
hopTip.renderOrder=21;hopTip.frustumCulled=false;hopTip.visible=false;scene.add(hopTip);
function formatTimelineTime(seconds) {
    if (seconds < 120) {
        const sec=Math.max(0,seconds);
        return Math.floor(sec/60)+":"+String(Math.floor(sec%60)).padStart(2,"0")+
            "."+String(Math.floor((sec%1)*100)).padStart(2,"0");
    }
    return formatTime(seconds);
}
function changeTimelineMode(mode) {
    if (!model || (mode!=="print"&&mode!=="layer")) return;
    setPlaying(false);
    selectedLayer=layerAtTime(currentTime);
    playbackMode=mode;
    if(mode==="layer") $("playSpeed").value="1";
    else $("playSpeed").value="100";
    $("modePrint").setAttribute("aria-pressed",String(mode==="print"));
    $("modeLayer").setAttribute("aria-pressed",String(mode==="layer"));
    $("timelineMarkers").classList.toggle("layer-mode",mode==="layer");
    markerKey="";
    seek(currentTime);
}
function chooseLayer(index) {
    if(!model)return;
    setPlaying(false);
    selectedLayer=Math.max(0,Math.min(model.layers.length-1,index));
    const [from,to]=layerRange(selectedLayer);
    const last=model.layers[selectedLayer].start+model.layers[selectedLayer].count-1;
    // The final deposition time is inside this layer even if a layer change
    // occurs immediately at the boundary. Empty layers show their endpoint.
    let time=last>=model.layers[selectedLayer].start?endAt(last):to;
    time=Math.max(from,Math.min(to,time));
    if(playbackMode==="print"&&selectedLayer<model.layers.length-1)
        time=Math.max(from,Math.min(time,to-0.00002));
    seek(time);
    if(playbackMode==="layer"){markerKey="";updateTimelineMarkers();}
}
function markerLabel(text,x,width,onClick) {
    const mark=document.createElement("button");
    mark.type="button"; mark.className="timeline-marker";
    mark.style.left=(100*x/Math.max(1,width)).toFixed(5)+"%";
    mark.title=text;
    const label=document.createElement("span");
    label.className="marker-label";
    label.textContent=text;
    if(x<32){label.style.left="0";label.style.transform="none";}
    else if(width-x<32){label.style.left="auto";label.style.right="0";label.style.transform="none";}
    mark.append(label);
    mark.addEventListener("click",onClick);
    $("timelineMarkers").append(mark);
}
function updateTimelineMarkers() {
    if(!model)return;
    const wrap=$("timelineMarkers");
    const width=Math.max(1,wrap.clientWidth);
    const active=playbackMode==="layer"?selectedLayer:visualLayer;
    const key=playbackMode+"|"+active+"|"+width;
    if(key===markerKey)return;
    markerKey=key;
    wrap.querySelectorAll(".timeline-marker").forEach(node=>node.remove());
    const canvas=$("timelineTicks");
    const ratio=Math.min(window.devicePixelRatio||1,2);
    canvas.width=Math.max(1,Math.round(width*ratio));
    canvas.height=Math.round(30*ratio);
    const ctx=canvas.getContext("2d");
    if(!ctx)return;
    ctx.setTransform(ratio,0,0,ratio,0,0);
    ctx.clearRect(0,0,width,30);
    if(playbackMode==="layer"){
        ctx.fillStyle="#6ea6cb";
        ctx.fillRect(0,18,2,12);ctx.fillRect(width-2,18,2,12);
        const number=model.layers[selectedLayer].number;
        markerLabel("Start L"+number,0,width,()=>{setPlaying(false);seek(layerRange()[0]);});
        markerLabel("Ende L"+number,width,width,()=>{setPlaying(false);seek(layerRange()[1]);});
        return;
    }
    const duration=Math.max(0.001,model.duration);
    const boundaries=model.layers.map((layer,index)=>({
        index, x:Math.max(0,Math.min(width,width*layer.timeStart/duration)),
        name:"L"+layer.number
    }));
    ctx.fillStyle="#55697f";
    for(const point of boundaries){
        const current=point.index===visualLayer;
        ctx.fillStyle=current?"#a8d7ff":"#526478";
        ctx.fillRect(Math.round(point.x),current?15:21,current?2:1,current?15:9);
    }
    const minDistance=64;
    let lastShown=-1000;
    const labeled=[];
    for(const point of boundaries){
        if(point.x-lastShown<minDistance)continue;
        labeled.push(point);
        lastShown=point.x;
    }
    const selected=boundaries[visualLayer];
    if(selected && !labeled.some(p=>p.index===selected.index)){
        const clash=labeled.findIndex(p=>Math.abs(p.x-selected.x)<minDistance);
        if(clash>=0)labeled.splice(clash,1);
        labeled.push(selected);
    }
    for(const point of labeled){
        markerLabel(point.name,point.x,width,()=>{
            setPlaying(false);
            selectedLayer=point.index;
            seek(model.layers[point.index].timeStart);
        });
        const el=wrap.lastElementChild;
        if(point.index===visualLayer)el.classList.add("current");
    }
}
function updateTimelineUI() {
    const [from,to]=activeRange();
    const span=Math.max(0,to-from);
    const fraction=span>1e-8?Math.max(0,Math.min(1,(currentTime-from)/span)):1;
    $("timeline").value=String(Math.round(fraction*TIMELINE_STEPS));
    $("elapsed").textContent=formatTimelineTime(Math.max(0,currentTime-from));
    $("duration").textContent=formatTimelineTime(span);
    $("percent").textContent=(fraction*100).toFixed(2)+" %";
    if(playbackMode==="layer"){
        const layer=model.layers[selectedLayer];
        $("timelineScope").textContent="Layer "+layer.number+
            " · Z "+layer.z.toFixed(3)+" mm · "+formatTime(from)+" – "+formatTime(to)+" (Druckzeit)";
    }else{
        $("timelineScope").textContent=model.layers.length+" Layer · "+formatTime(model.duration)+" Gesamtdauer";
    }
    updateTimelineMarkers();
}
function seek(seconds) {
    if(!model||!data)return;
    currentTime=Math.max(0,Math.min(model.duration,seconds));
    visualLayer=playbackMode==="layer"?selectedLayer:layerAtTime(currentTime);
    if(playbackMode==="print")selectedLayer=visualLayer;
    updateTimelineUI();
    updateVisibility();
}
function updateVisibility() {
    if(!model || !data)return;
    const finished=completedBefore(model.count,currentTime,endAt);
    const current=visualLayer;
    visibleCount=finished;
    slider.value=String(current);
    const below=Number($("below").value);
    const start=below<0?0:Math.max(0,current-below);
    const keep=new Set();
    activeLayers=[];
    for(let i=start;i<=current;i++){
        const group=getLayer(i);keep.add(i);activeLayers.push(group);
        if(!group.parent)scene.add(group);
        for(const mesh of group.children){
            mesh.geometry.instanceCount=withinPrefix(mesh.userData.indices,finished);
            mesh.visible=mesh.geometry.instanceCount>0&&visibleRoles.has(mesh.userData.role);
            const faded=i!==current&&$("ghost").checked;
            if(mesh.userData.faded!==faded){
                mesh.material.dispose();
                mesh.material=newMaterial(mesh.userData.role,faded);
                mesh.userData.faded=faded;
            }
        }
    }
    for(const [i,group] of cache){
        if(!keep.has(i)){
            scene.remove(group);
            if(cache.size>24||below===0){disposeGroup(group);cache.delete(i);}
        }
    }
    headMesh.visible=false;
    if(finished<model.count){
        const p=readSegment(finished);
        if(p.start<=currentTime&&currentTime<p.end&&visibleRoles.has(p.role)){
            const alpha=Math.max(0,Math.min(1,(currentTime-p.start)/(p.end-p.start)));
            if(alpha>0){
                headMesh.material.uniforms.uColor.value.copy(rgb[p.role]);
                headMesh.userData.role=p.role;
                renderTip(headMesh,p,alpha);
            }
        }
    }
    const hopsFinished=completedBefore(model.hopCount,currentTime,hopEndAt);
    const showHops=$("showHops").checked;
    for(let i=start;i<=current;i++){
        const info=model.layers[i];
        if(!showHops||!info.hopCount)continue;
        let g=hopCache.get(i);
        if(!g){g=createHopLayer(i);hopCache.set(i,g);}
        keep.add("h"+i);
        if(!g.parent)scene.add(g);
        g.geometry.setDrawRange(0,2*Math.max(0,Math.min(info.hopCount,hopsFinished-info.hopStart)));
        g.visible=true;
    }
    for(const [i,g] of hopCache){
        if(!showHops||!keep.has("h"+i)){
            scene.remove(g);
            if(hopCache.size>24||!showHops){disposeHopGroup(g);hopCache.delete(i);}
        }
    }
    hopTip.visible=false;
    if(showHops&&hopsFinished<model.hopCount){
        const hop=readHop(hopsFinished);
        if(hop.start<=currentTime&&currentTime<hop.end){
            const startIndex=model.layers.findIndex(l=>hopsFinished>=l.hopStart&&hopsFinished<l.hopStart+l.hopCount);
            if(startIndex>=start&&startIndex<=current){
                const ratio=Math.max(0,Math.min(1,(currentTime-hop.start)/(hop.end-hop.start)));
                const p=hopTip.geometry.attributes.position;
                p.array.set([hop.a[0],hop.a[2],hop.a[1],
                    hop.a[0]+(hop.b[0]-hop.a[0])*ratio,
                    hop.a[2]+(hop.b[2]-hop.a[2])*ratio,
                    hop.a[1]+(hop.b[1]-hop.a[1])*ratio]);
                p.needsUpdate=true;hopTip.visible=ratio>0;
            }
        }
    }
    const layer=model.layers[current];
    $("layerText").textContent="Layer "+layer.number+" / "+model.layers.at(-1).number;
    $("layerZ").textContent="Z "+layer.z.toFixed(3)+" mm";
    $("layerPrev").disabled=current===0;
    $("layerNext").disabled=current===model.layers.length-1;
    slider.setAttribute("aria-valuetext","Layer "+layer.number+" · Z "+layer.z.toFixed(3)+" mm");
}
function fit() {
    const [lo,hi] = model.bounds;
    const cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2,cz=(lo[2]+hi[2])/2;
    const size=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],10);
    controls.target.set(cx,cz,cy);
    camera.position.set(cx+size*.6,cz+size*.85,cy+size*.85);
    camera.near=Math.max(.01,size/15000);camera.far=size*150;camera.updateProjectionMatrix();
    controls.update();
    grid.position.set(cx,lo[2]-.15,cy);
    grid.scale.set(Math.max(1,size/300),1,Math.max(1,size/300));
}
async function loadModel(index) {
    forgetLayers();
    model=layout.models[index];
    $("progress").hidden=false;
    $("progress").textContent="Lade "+model.name+" · "+model.count.toLocaleString("de-DE")+" Stränge…";
    const [response,hopResponse]=await Promise.all([fetch(model.url),fetch(model.hopUrl)]);
    if(!response.ok||!hopResponse.ok)throw Error("HTTP "+response.status+" / "+hopResponse.status);
    data=new DataView(await response.arrayBuffer());
    hopData=new DataView(await hopResponse.arrayBuffer());
    if(data.byteLength!==model.count*SEGMENT_SIZE || hopData.byteLength!==model.hopCount*HOP_SIZE){
        throw new Error("G-Code-Vorschau: Ungültiges Binärformat oder Segmentgröße");
    }
    $("fileName").textContent=model.filename;
    $("stats").textContent=model.count.toLocaleString("de-DE")+" Extrusionssegmente · "+model.layers.length+" Layer · Filament Ø "+model.filament.toFixed(2)+" mm"+(model.ignored_arcs?" · Achtung: "+model.ignored_arcs+" Bögen nicht dargestellt":"");
    slider.max=String(model.layers.length-1);
    selectedLayer=Math.max(0,model.layers.length-1);
    markerKey="";
    fit();
    seek(model.duration);
    $("progress").hidden=true;
}
modelSelect.addEventListener("change",()=>{
    setPlaying(false);
    loadModel(Number(modelSelect.value)).catch(showError);
});
slider.addEventListener("input",()=>chooseLayer(Number(slider.value)));
$("layerNext").addEventListener("click",()=>chooseLayer(selectedLayer+1));
$("layerPrev").addEventListener("click",()=>chooseLayer(selectedLayer-1));
$("modePrint").addEventListener("click",()=>changeTimelineMode("print"));
$("modeLayer").addEventListener("click",()=>changeTimelineMode("layer"));
$("timeline").addEventListener("input",()=>{
    if(!model)return;
    setPlaying(false);
    const [from,to]=activeRange();
    seek(from+Number($("timeline").value)/TIMELINE_STEPS*(to-from));
});
$("timelineTicks").addEventListener("click",event=>{
    if(!model)return;
    const bounds=$("timelineTicks").getBoundingClientRect();
    if(!bounds.width)return;
    setPlaying(false);
    const [from,to]=activeRange();
    const fraction=Math.max(0,Math.min(1,(event.clientX-bounds.left)/bounds.width));
    seek(from+(to-from)*fraction);
});
$("timelineTicks").addEventListener("mousemove",event=>{
    if(!model)return;
    const canvas=$("timelineTicks");
    if(playbackMode==="layer"){
        canvas.title="Layer "+model.layers[selectedLayer].number+": Position setzen";
        return;
    }
    const bounds=canvas.getBoundingClientRect();
    const fraction=Math.max(0,Math.min(1,(event.clientX-bounds.left)/Math.max(1,bounds.width)));
    const time=fraction*model.duration;
    const index=layerAtTime(time);
    canvas.title="Layer "+model.layers[index].number+" · "+formatTime(time);
});
$("playToggle").addEventListener("click",()=>{
    if(!model)return;
    const [from,to]=activeRange();
    if(!playing&&currentTime>=to-1e-6)seek(from);
    setPlaying(!playing);
});
$("restart").addEventListener("click",()=>{
    setPlaying(false);
    seek(activeRange()[0]);
});
$("below").addEventListener("change",updateVisibility);
$("ghost").addEventListener("change",updateVisibility);
$("showHops").addEventListener("change",updateVisibility);
$("gridToggle").addEventListener("change",()=>{grid.visible=$("gridToggle").checked;});
$("resetView").addEventListener("click",fit);
function showError(error) {
    $("progress").hidden=false;
    $("progress").textContent="Vorschaufehler: "+error.message;
    console.error(error);
}
const mouse=new THREE.Vector2();
function selectSegment(event) {
    if(!model||!data)return;
    const bounds=renderer.domElement.getBoundingClientRect();
    mouse.set(((event.clientX-bounds.left)/bounds.width)*2-1,
              -((event.clientY-bounds.top)/bounds.height)*2+1);
    const current=Number(slider.value), layer=model.layers[current];
    let best=12*12, pick=null;
    const a=new THREE.Vector3(),b=new THREE.Vector3();
    for(let i=layer.start;i<Math.min(layer.start+layer.count,visibleCount);i++){
        const p=readSegment(i);
        if(!visibleRoles.has(p.role))continue;
        a.set(p.a[0],p.a[2],p.a[1]).project(camera);
        b.set(p.b[0],p.b[2],p.b[1]).project(camera);
        const dx=b.x-a.x,dy=b.y-a.y;
        const t=Math.max(0,Math.min(1,((mouse.x-a.x)*dx+(mouse.y-a.y)*dy)/(dx*dx+dy*dy||1)));
        const x=a.x+t*dx,y=a.y+t*dy;
        const d=((x-mouse.x)*bounds.width/2)**2+((y-mouse.y)*bounds.height/2)**2;
        if(d<best){best=d;pick=p;}
    }
    if(lastPick){scene.remove(lastPick);lastPick.geometry.dispose();lastPick.material.dispose();lastPick=null;}
    if(!pick){$("pickInfo").style.display="none";return;}
    const a3=new THREE.Vector3(pick.a[0],pick.a[2],pick.a[1]),b3=new THREE.Vector3(pick.b[0],pick.b[2],pick.b[1]);
    const geom=new THREE.BufferGeometry().setFromPoints([a3,b3]);
    lastPick=new THREE.Line(geom,new THREE.LineBasicMaterial({color:"#ffffff",depthTest:false}));
    lastPick.renderOrder=999;scene.add(lastPick);
    const length=Math.hypot(...pick.a.map((v,i)=>pick.b[i]-v));
    const area=pick.h*(pick.w-pick.h)+Math.PI*pick.h*pick.h/4;
    const extruded=area*length/(Math.PI*(model.filament/2)**2);
    $("pickInfo").style.display="block";
    $("pickInfo").replaceChildren();
    const title=document.createElement("strong");
    title.textContent=layout.roles[pick.role]+" · G-Code-Zeile "+pick.line;
    const body=document.createElement("div");
    body.textContent="Breite "+pick.w.toFixed(3)+" mm · Höhe "+pick.h.toFixed(3)+" mm · Länge "+length.toFixed(2)+" mm · ΔE "+extruded.toFixed(5)+" mm";
    $("pickInfo").append(title,body);
}
let down;
renderer.domElement.addEventListener("pointerdown",e=>{down=[e.clientX,e.clientY]});
renderer.domElement.addEventListener("pointerup",e=>{if(down&&Math.hypot(e.clientX-down[0],e.clientY-down[1])<4)selectSegment(e);down=null;});
const resize = new ResizeObserver(() => {
    const w=canvasHost.clientWidth,h=canvasHost.clientHeight;
    if(w&&h){renderer.setSize(w,h,false);camera.aspect=w/h;camera.updateProjectionMatrix();}
});
resize.observe(canvasHost);
const timelineResize=new ResizeObserver(()=>{
    markerKey="";
    if(model)updateTimelineMarkers();
});
timelineResize.observe($("timelineMarkers"));
renderer.setAnimationLoop(()=>{
    const now=performance.now();
    if(playing&&model&&data){
        if(previousFrame){
            const [from,to]=activeRange();
            const next=Math.min(to,Math.max(from,currentTime+(now-previousFrame)/1000*Number($("playSpeed").value)));
            seek(next);
            if(next>=to)setPlaying(false);
        }
        previousFrame=now;
    }
    controls.update();renderer.render(scene,camera);
});
// The detached Python server releases its temporary binary files after inactivity.
setInterval(() => fetch("/ping", {cache:"no-store"}).catch(() => {}), 20000);
async function uploadGcode(file) {
    if(!file || uploading)return;
    if(!/\.(?:gcode(?:\.pp)?|gc|g)$/i.test(file.name)){
        showError(new Error("Bitte eine .gcode-, .gcode.pp-, .gc- oder .g-Datei wählen."));
        return;
    }
    uploading=true;
    setPlaying(false);
    $("openFile").disabled=true;
    const indicator=$("progress");
    indicator.hidden=false;
    indicator.textContent="Lade "+file.name+" hoch…";
    try{
        await new Promise((resolve,reject)=>{
            const xhr=new XMLHttpRequest();
            xhr.open("POST","/upload");
            xhr.setRequestHeader("X-Gcode-Filename",encodeURIComponent(file.name));
            xhr.upload.onprogress=event=>{
                if(event.lengthComputable){
                    indicator.textContent="Lade "+file.name+" hoch: "+
                        Math.floor(event.loaded/event.total*100)+" %";
                }
            };
            xhr.onerror=()=>reject(new Error("Upload fehlgeschlagen: Server nicht erreichbar"));
            xhr.onload=()=>xhr.status===202?resolve():
                reject(new Error("Upload fehlgeschlagen (HTTP "+xhr.status+"): "+xhr.responseText.slice(0,180)));
            xhr.send(file);
        });
        const next=await loadPreviewMetadata();
        if(next.status==="error")throw new Error(next.message);
        layout=next;
        refreshModelOptions();
        modelSelect.value=String(layout.models.length-1);
        await loadModel(layout.models.length-1);
    }catch(error){
        showError(error);
    }finally{
        uploading=false;
        $("openFile").disabled=false;
        $("gcodeFile").value="";
    }
}
let uploading=false;
$("openFile").addEventListener("click",()=>$("gcodeFile").click());
$("gcodeFile").addEventListener("change",event=>uploadGcode(event.target.files?.[0]));
let dragDepth=0;
window.addEventListener("dragenter",event=>{
    if(!Array.from(event.dataTransfer?.types||[]).includes("Files"))return;
    event.preventDefault();
    dragDepth++;
    document.body.classList.add("dragging");
});
window.addEventListener("dragover",event=>{
    if(Array.from(event.dataTransfer?.types||[]).includes("Files"))event.preventDefault();
});
window.addEventListener("dragleave",event=>{
    dragDepth=Math.max(0,dragDepth-1);
    if(!dragDepth)document.body.classList.remove("dragging");
});
window.addEventListener("drop",event=>{
    event.preventDefault();
    dragDepth=0;
    document.body.classList.remove("dragging");
    uploadGcode(event.dataTransfer?.files?.[0]);
});
if(layout.models.length){
    loadModel(0).catch(showError);
}else{
    $("fileName").textContent="Keine G-Code-Datei geladen";
    $("stats").textContent="G-Code per Drag & Drop oder über „Datei öffnen“ laden.";
    if(layout.status!=="error")$("progress").hidden=true;
}
