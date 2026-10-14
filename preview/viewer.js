import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";
import {loftBoundary,planLoft,spatialChunks} from "./loft_geometry.js";

import {profileArea} from "./bead_profile.js";
import {nearestSourceRecord,rawClass} from "./gcode_inspector.js";
import {MOVE_SIZE,sourceLineAtTime,sourceLineAtLayerEnd,sourceMotionForLine} from "./source_timeline.js";
import {analysePrint} from "./diagnostics.js";
import {closestLayer, compareLayerVolumes} from "./comparison.js";
import {buildCutContours} from "./section.js";

const $ = id => document.getElementById(id);
const defaultColors = ["#adb8c6","#ae7bff","#3bcbec","#298df2","#ffb75e","#f28dc2","#67db8c","#ff4655"];
const validPalette = values => Array.isArray(values) &&
    values.length === defaultColors.length &&
    values.every(color => typeof color === "string" && /^#[0-9a-f]{6}$/i.test(color));
function loadPalette(serverColors) {
    if (validPalette(serverColors)) return [...serverColors];
    if (Array.isArray(serverColors) && serverColors.length===defaultColors.length-1 &&
        serverColors.every(color=>typeof color==="string"&&/^#[0-9a-f]{6}$/i.test(color)))
        return [...serverColors,defaultColors.at(-1)];
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
renderer.localClippingEnabled = true;
canvasHost.insertBefore(renderer.domElement, canvasHost.firstChild);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = .075;
const clipPlane = new THREE.Plane(new THREE.Vector3(1,0,0),1e9);
const clipPlanes=[clipPlane];
const grid = new THREE.GridHelper(300, 60, "#4d6076", "#263144");
scene.add(grid);
const modelSelect = $("model"), slider = $("layer");
let model = null, data = null, hopData = null, cache = new Map(), hopCache = new Map(), lastPick = null;
let activeLayers = [], currentTime = 0, visibleCount = 0, playing = false, previousFrame = 0;
let rawData = null, rawLayerCache = new Map(), rawRequest = 0, selectedSourceLine = 0;
let moveData = null, moveLoading = null, browsingLayer = false;
let currentSourceLine=0, sourceWindowStart=0, sourceWindowEnd=0;
let lastSourceAutoFetch=0, sourceSelectionTime=0;
const RAW_SIZE = 40;
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
    if(frontierCap.visible && frontierCap.userData.role===index)
        frontierCap.material.uniforms.uColor.value.copy(rgb[index]);
    if(index===7 && typeof hopTip!=="undefined"){
        for(const group of hopCache.values())group.material.color.set(hex);
        hopTip.material.color.set(hex);
        $("hopColor").value=hex;
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
    if(layout.originalAvailable&&!layout.models.some(m=>m.name==="Original")){
        const opt=document.createElement("option");
        opt.value="original-pending";
        opt.textContent="Original (bei Bedarf laden)";
        modelSelect.append(opt);
    }
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
$("hopColor").value=colors[7];
$("hopColor").addEventListener("input",event=>applyColor(7,event.target.value));
$("hopColor").addEventListener("change",savePalette);

// Parametric strongly rounded rectangle (r=0.4*min(width,height)).
// Shared-profile GPU loft mantles contain no caps; caps are drawn only at
// genuine strand boundaries. The temporary animation head is closed.
function sectionMesh(steps=4,mode="mantle"){
    const vertices=[],indices=[];
    const append=(side,top,cos,sin,end,cap)=>{
        vertices.push(side,top,cos,sin,end,cap);
        return vertices.length/6-1;
    };
    const corners=[];
    for(const [sx,sy,from] of [[1,1,0],[-1,1,Math.PI/2],
                                 [-1,-1,Math.PI],[1,-1,3*Math.PI/2]]){
        for(let step=0;step<=steps;step++){
            const angle=from+step*Math.PI/(2*steps);
            corners.push([sx,sy,Math.cos(angle),Math.sin(angle)]);
        }
    }
    const n=corners.length,rings=[];
    const makeRing=(end,cap)=>{
        const ring=[];
        for(const [side,top,cos,sin] of corners)
            ring.push(append(side,top,cos,sin,end,cap));
        return ring;
    };
    if(mode!=="cap"){
        rings.push(makeRing(0,0),makeRing(1,0));
        for(let i=0;i<n;i++){
            const j=(i+1)%n;
            indices.push(rings[0][i],rings[1][i],rings[0][j],
                         rings[1][i],rings[1][j],rings[0][j]);
        }
    }
    if(mode!=="mantle"){
        for(const end of mode==="cap"?[0]:[0,1]){
            const ring=makeRing(end,1),center=append(0,0,0,0,end,2);
            for(let i=0;i<n;i++){
                const j=(i+1)%n;
                if(end===0)indices.push(center,ring[i],ring[j]);
                else indices.push(center,ring[j],ring[i]);
            }
        }
    }
    const geometry=new THREE.InstancedBufferGeometry();
    const coords=new Float32Array(vertices);
    for(let attr=0;attr<6;attr++){
        if(mode==="cap"&&attr===4)continue;
        const values=new Float32Array(coords.length/6);
        for(let i=0;i<values.length;i++)values[i]=coords[i*6+attr];
        geometry.setAttribute(
            ["aSide","aTop","aCos","aSin","aEnd","aCap"][attr],
            new THREE.Float32BufferAttribute(values,1)
        );
    }
    geometry.setIndex(indices);
    return geometry;
}
const base=sectionMesh(4,"mantle");
const capBase=sectionMesh(4,"cap");
const headBase=sectionMesh(4,"closed");
// Low/high profile templates only contain shared vertices/indices. Per-instance
// attributes are reused across LODs: no duplicate path data on the GPU.
const mantleLods=[sectionMesh(2,"mantle"),base,sectionMesh(7,"mantle")];
const capLods=[sectionMesh(2,"cap"),capBase,sectionMesh(7,"cap")];
const vertexShader = `
precision highp float;
attribute float aSide, aTop, aCos, aSin, aEnd, aCap;
attribute vec3 iStart, iStop;
attribute vec2 iSize;
attribute vec3 iJoinStart, iJoinEnd;
attribute vec2 iSizeStart, iSizeEnd;
uniform float uAlpha;
varying vec3 vNormal;
varying vec3 vPosition;
#include <clipping_planes_pars_vertex>
void main(){
  vec3 d=iStop-iStart;
  float len=length(d.xy);
  vec3 join=mix(iJoinStart,iJoinEnd,aEnd);
  vec3 perpendicular=len>0.0000001 ? vec3(-d.y/len,0.0,d.x/len) : vec3(1.0,0.0,0.0);
  vec3 joinVector=vec3(join.x,0.0,join.y);
  float h=mix(iSizeStart.y,iSizeEnd.y,aEnd);
  float w=mix(iSizeStart.x,iSizeEnd.x,aEnd);
  float radius=0.4*min(w,h);
  float lateral=aSide*(w*0.5-radius)+aCos*radius;
  float vertical=aTop*(h*0.5-radius)+aSin*radius;
  vec3 origin=mix(iStart,iStop,aEnd);
  vec3 basePoint=vec3(origin.x,origin.z-h*0.5,origin.y);
  // The top stays at G-code nozzle Z; only the bottom stretches down.
  vec3 p=basePoint + joinVector*lateral + vec3(0.0,vertical,0.0);
  vec3 tangent=normalize(vec3(d.x,d.z,d.y));
  vNormal=aCap>0.5 ? (aEnd>0.5 ? tangent : -tangent)
                      : normalize(joinVector*aCos+vec3(0.0,aSin,0.0));
  vPosition=p;
  vec4 mvPosition=modelViewMatrix*vec4(p,1.0);
  gl_Position=projectionMatrix*mvPosition;
  #include <clipping_planes_vertex>
}
`;
// The cap uses the same transformed ring as its neighboring loft mantle.
// iCapEnd selects the start/end ring per cap INSTANCE, not per base vertex.
const capVertexShader=vertexShader.replaceAll("aEnd","iCapEnd");
const fragmentShader = `
precision highp float;
uniform vec3 uColor;
uniform float uAlpha;
varying vec3 vNormal;
varying vec3 vPosition;
#include <clipping_planes_pars_fragment>
void main(){
 #include <clipping_planes_fragment>
 vec3 light=normalize(vec3(-0.45,0.85,0.5));
 float diffuse=max(dot(normalize(vNormal),light),0.0);
 float shade=0.47+0.53*diffuse;
 gl_FragColor=vec4(uColor*shade,uAlpha);
 #include <tonemapping_fragment>
 #include <colorspace_fragment>
}
`;
const newMaterial = (role, faded, geometryShader=vertexShader) => new THREE.ShaderMaterial({
    vertexShader:geometryShader, fragmentShader, side:THREE.DoubleSide,
    clipping:true, clippingPlanes:clipPlanes,
    uniforms:{uColor:{value:rgb[role].clone()},uAlpha:{value:faded?.19:1}},
    transparent:!!faded,depthWrite:!faded,
    polygonOffset:false
});
function disposeGroup(group) {
    group.traverse(o => {
        if(o.isMesh){
            for(const geometry of new Set(o.userData.lodVariants?.filter(Boolean)||
                [o.geometry]))geometry.dispose();
            o.material.dispose();
        }
    });
}
function forgetLayers() {
    for (const group of cache.values()) {scene.remove(group);disposeGroup(group);}
    for (const group of hopCache.values()) {scene.remove(group);disposeHopGroup(group);}
    cache.clear();hopCache.clear();activeLayers=[];
    for(const overlay of rawLayerCache.values()) {
        scene.remove(overlay);
        overlay.geometry.dispose();
        overlay.material.dispose();
    }
    rawLayerCache.clear();
    rawData=null;rawRequest++;selectedSourceLine=0;
    moveData=null;moveLoading=null;browsingLayer=false;
    currentSourceLine=0;sourceWindowStart=sourceWindowEnd=0;
    sourceRequest++;lastSourceAutoFetch=sourceSelectionTime=0;
    $("selectedSource").classList.remove("visible");
    $("playingSource").classList.remove("visible");
    $("rawInfo").textContent="Originalbewegungen im G-Code prüfen";
    $("sourceDrawer").classList.remove("visible");
    disposeOverlay(diagnosticObjects);
    clearComparison();
    removeSection();
    sectionKey="";
    diagnosticReport=null;
    comparisonSerial++;
    headMesh.visible=false; hopTip.visible=false;frontierCap.visible=false;
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
function readRaw(index) {
    const off=index*RAW_SIZE;
    return {
        a:[rawData.getFloat32(off,true),rawData.getFloat32(off+4,true),rawData.getFloat32(off+8,true)],
        b:[rawData.getFloat32(off+12,true),rawData.getFloat32(off+16,true),rawData.getFloat32(off+20,true)],
        kind:rawData.getUint32(off+24,true),
        line:rawData.getUint32(off+28,true),
        start:rawData.getFloat32(off+32,true),
        end:rawData.getFloat32(off+36,true)
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
    if(!playing && typeof scheduleSection==="function")scheduleSection();
    else if(playing)removeSection();
    if(!playing && referenceData)refreshComparison();
}
const joinRead = index => index>=0 && model && index<model.count ? readSegment(index) : null;
function boundLoft(geometry,bounds){
    if(!bounds)return;
    const {lo,hi,padding}=bounds;
    const center=new THREE.Vector3((lo[0]+hi[0])*.5,(lo[2]+hi[2])*.5,(lo[1]+hi[1])*.5);
    const dx=(hi[0]-lo[0])*.5+padding;
    const dy=(hi[2]-lo[2])*.5+padding;
    const dz=(hi[1]-lo[1])*.5+padding;
    geometry.boundingSphere=new THREE.Sphere(center,Math.hypot(dx,dy,dz));
}
function bindLoftAttributes(geometry,plan,capItems=null){
    const vectors=[
        ["iStart",plan.starts,3],["iStop",plan.ends,3],
        ["iSize",plan.sizes,2],["iJoinStart",plan.joinsStart,3],
        ["iJoinEnd",plan.joinsEnd,3],["iSizeStart",plan.sizesStart,2],
        ["iSizeEnd",plan.sizesEnd,2]
    ];
    if(capItems){
        for(const [key,values,stride] of vectors){
            const selected=new Float32Array(capItems.length*stride);
            for(let j=0;j<capItems.length;j++){
                const source=capItems[j].local*stride, target=j*stride;
                for(let k=0;k<stride;k++)selected[target+k]=values[source+k];
            }
            geometry.setAttribute(key,new THREE.InstancedBufferAttribute(selected,stride));
        }
        const capEnds=new Float32Array(capItems.length);
        for(let j=0;j<capItems.length;j++)capEnds[j]=capItems[j].end;
        geometry.setAttribute("iCapEnd",new THREE.InstancedBufferAttribute(capEnds,1));
        geometry.instanceCount=capItems.length;
    }else{
        for(const [key,values,stride] of vectors)
            geometry.setAttribute(key,new THREE.InstancedBufferAttribute(values,stride));
        geometry.instanceCount=plan.starts.length/3;
    }
    boundLoft(geometry,plan.bounds);
}
function makeLoftLod(mesh,level){
    const templates=mesh.userData.cap?capLods:mantleLods;
    const geometry=templates[level].clone();
    for(const [name,attribute] of Object.entries(mesh.geometry.attributes)){
        if(name.startsWith("i"))geometry.setAttribute(name,attribute);
    }
    geometry.instanceCount=mesh.geometry.instanceCount;
    geometry.boundingSphere=mesh.geometry.boundingSphere?.clone()||null;
    return geometry;
}
function updateLoftLods(){
    if(!model||!data)return;
    const height=renderer.domElement.clientHeight;
    if(!height)return;
    const f=height/(2*Math.tan(camera.fov*Math.PI/360));
    for(const group of activeLayers){
        for(const mesh of group.children){
            if(!mesh.userData.lodVariants||!mesh.visible)continue;
            const bounds=mesh.geometry.boundingSphere;
            if(!bounds)continue;
            const distance=Math.max(camera.near,
                camera.position.distanceTo(bounds.center));
            const pixelWidth=f*mesh.userData.lodWidth/distance;
            // Hysteresis prevents constant buffer/VAO churn at thresholds.
            const previous=mesh.userData.lodLevel;
            const next=previous===0
                ? (pixelWidth>1.7?1:0)
                : previous===2?(pixelWidth<6.2?1:2)
                : pixelWidth<1.1?0:pixelWidth>8.0?2:1;
            if(next===previous)continue;
            let variant=mesh.userData.lodVariants[next];
            if(!variant){
                variant=makeLoftLod(mesh,next);
                mesh.userData.lodVariants[next]=variant;
            }
            variant.instanceCount=mesh.geometry.instanceCount;
            mesh.geometry=variant;
            mesh.userData.lodLevel=next;
        }
    }
}
function createLayer(index) {
    const layerStarted=performance.now();
    const info=model.layers[index],group=new THREE.Group();
    const first=info.start,last=first+info.count;
    const recordCache=new Array(info.count);
    const cachedRead=id=>{
        if(id<first||id>=last)return joinRead(id);
        return recordCache[id-first] ||= readSegment(id);
    };
    const chunks=spatialChunks(first,last,cachedRead,40,6000);
    const seamless=$("seamlessJoins").checked;
    for(const chunk of chunks){
        const plan=planLoft(chunk.indices,cachedRead,first,last,seamless);
        const geometry=base.clone();
        bindLoftAttributes(geometry,plan);
        const mesh=new THREE.Mesh(geometry,newMaterial(chunk.role,false));
        mesh.frustumCulled=true;
        mesh.userData.role=chunk.role;
        mesh.userData.indices=Uint32Array.from(chunk.indices);
        mesh.userData.lodVariants=[null,geometry,null];
        mesh.userData.lodLevel=1;
        mesh.userData.lodWidth=plan.sizes.length?
            plan.sizes[Math.floor((plan.sizes.length/2)*.5)*2]:.45;
        group.add(mesh);
        if(plan.caps.length){
            const capGeometry=capBase.clone();
            bindLoftAttributes(capGeometry,plan,plan.caps);
            const caps=new THREE.Mesh(capGeometry,newMaterial(chunk.role,false,capVertexShader));
            caps.frustumCulled=true;
            caps.userData.role=chunk.role;
            caps.userData.cap=true;
            caps.userData.indices=Uint32Array.from(plan.caps.map(item=>item.index));
            caps.userData.lodVariants=[null,capGeometry,null];
            caps.userData.lodLevel=1;
            caps.userData.lodWidth=mesh.userData.lodWidth;
            group.add(caps);
        }
    }
    group.userData.layer=index;
    const ms=performance.now()-layerStarted;
    if(ms>40)console.debug(`[Viewer] Layer ${index}: ${info.count} segments, ${group.children.length} meshes, ${ms.toFixed(1)}ms`);
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
    const lines=new THREE.LineSegments(geometry,new THREE.LineBasicMaterial({color:colors[7],depthTest:false,depthWrite:false,clippingPlanes:clipPlanes}));
    lines.frustumCulled=false;lines.renderOrder=20;
    return lines;
}
function disposeHopGroup(group) {
    group.geometry.dispose();group.material.dispose();
}
function renderTip(mesh,p,fraction,index) {
    const stop=p.a.map((v,i)=>v+(p.b[i]-v)*fraction);
    const attrs=mesh.geometry.attributes;
    const first=model.layers[visualLayer],boundary=loftBoundary(
        joinRead,index,false,first.start,first.start+first.count,$("seamlessJoins").checked);
    attrs.iStart.array.set(boundary.point);
    attrs.iStop.array.set(stop);
    attrs.iStart.needsUpdate=true;attrs.iStop.needsUpdate=true;
    attrs.iSize.array.set([p.w,p.h]);attrs.iSize.needsUpdate=true;
    const left=boundary;
    const dx=p.b[0]-p.a[0],dy=p.b[1]-p.a[1],l=Math.hypot(dx,dy)||1;
    const direction=[-dy/l,dx/l,0];
    attrs.iJoinStart.array.set(left.join);
    attrs.iJoinEnd.array.set(direction);
    attrs.iJoinStart.needsUpdate=attrs.iJoinEnd.needsUpdate=true;
    attrs.iSizeStart.array.set(left.size);
    attrs.iSizeEnd.array.set([p.w,p.h]);
    attrs.iSizeStart.needsUpdate=attrs.iSizeEnd.needsUpdate=true;
    mesh.visible=true;
}
const headGeometry=headBase.clone();
headGeometry.setAttribute("iJoinStart",new THREE.InstancedBufferAttribute(new Float32Array(3),3));
headGeometry.setAttribute("iJoinEnd",new THREE.InstancedBufferAttribute(new Float32Array(3),3));
headGeometry.setAttribute("iSizeStart",new THREE.InstancedBufferAttribute(new Float32Array(2),2));
headGeometry.setAttribute("iSizeEnd",new THREE.InstancedBufferAttribute(new Float32Array(2),2));
headGeometry.setAttribute("iStart",new THREE.InstancedBufferAttribute(new Float32Array(3),3));
headGeometry.setAttribute("iStop",new THREE.InstancedBufferAttribute(new Float32Array(3),3));
headGeometry.setAttribute("iSize",new THREE.InstancedBufferAttribute(new Float32Array(2),2));
headGeometry.instanceCount=1;
const headMesh=new THREE.Mesh(headGeometry,newMaterial(0,false));
headMesh.frustumCulled=false;headMesh.visible=false;scene.add(headMesh);
// A fully completed strand can still have an open endpoint during playback,
// because the following loft segment has not been deposited yet. This one
// transient cap closes only that advancing frontier, not every segment.
const frontierGeometry=capBase.clone();
for(const [name,size] of [
    ["iStart",3],["iStop",3],["iSize",2],
    ["iJoinStart",3],["iJoinEnd",3],
    ["iSizeStart",2],["iSizeEnd",2],["iCapEnd",1]
]){
    frontierGeometry.setAttribute(name,new THREE.InstancedBufferAttribute(
        new Float32Array(size),size));
}
frontierGeometry.attributes.iCapEnd.array[0]=1;
frontierGeometry.instanceCount=1;
const frontierCap=new THREE.Mesh(frontierGeometry,newMaterial(0,false,capVertexShader));
frontierCap.visible=false;frontierCap.frustumCulled=false;
scene.add(frontierCap);
function renderFrontierCap(index){
    const p=readSegment(index);
    const layer=model.layers.find(l=>index>=l.start&&index<l.start+l.count);
    if(!layer)return;
    const ring=loftBoundary(joinRead,index,true,layer.start,layer.start+layer.count,
        $("seamlessJoins").checked);
    if(!ring.connected)return;
    const attrs=frontierGeometry.attributes;
    for(const [key,values] of [
        ["iStart",p.a],["iStop",ring.point],["iSize",[p.w,p.h]],
        ["iJoinStart",ring.join],["iJoinEnd",ring.join],
        ["iSizeStart",ring.size],["iSizeEnd",ring.size]
    ]){
        attrs[key].array.set(values);
        attrs[key].needsUpdate=true;
    }
    frontierCap.userData.role=p.role;
    frontierCap.material.uniforms.uColor.value.copy(rgb[p.role]);
    frontierCap.material.uniforms.uAlpha.value=
        (layer.number!==model.layers[visualLayer].number && $("ghost").checked)?0.19:1;
    frontierCap.material.transparent=frontierCap.material.uniforms.uAlpha.value<1;
    frontierCap.material.depthWrite=!frontierCap.material.transparent;
    frontierCap.visible=true;
}
const hopTipGeom=new THREE.BufferGeometry();
hopTipGeom.setAttribute("position",new THREE.BufferAttribute(new Float32Array(6),3));
const hopTip=new THREE.Line(hopTipGeom,new THREE.LineBasicMaterial({color:colors[7],depthTest:false,depthWrite:false,clippingPlanes:clipPlanes}));
hopTip.renderOrder=21;hopTip.frustumCulled=false;hopTip.visible=false;scene.add(hopTip);
// Inspection overlays are allocated only when explicitly enabled.
let diagnosticReport=null, diagnosticLayer=null, diagnosticObjects=[];
let comparisonLayer=-1, comparisonObjects=[], referenceData=null, referenceModel=null;
let comparisonSerial=0, cutHelper=null, sectionLines=null, sectionKey="", sectionPending=false;
function disposeOverlay(objects){
    for(const obj of objects){scene.remove(obj);obj.geometry?.dispose();obj.material?.dispose();}
    objects.length=0;
}
function refreshCut(){
    const enabled=$("cutEnabled").checked && !!model;
    const axis=$("cutAxis").value, n=new THREE.Vector3(0,0,0);
    const idx=axis==="x"?0:axis==="y"?1:2;
    const coord=idx===2?1:idx===1?2:0;
    n.setComponent(coord,$("cutInvert").checked?-1:1);
    const bounds=model?.bounds||[[0,0,0],[1,1,1]];
    const min=bounds[0][idx],max=bounds[1][idx];
    const position=min+(max-min)*Number($("cutPosition").value)/1000;
    clipPlane.normal.copy(n);
    clipPlane.constant=enabled?-position*n.getComponent(coord):1e9;
    $("cutValue").textContent=enabled?axis.toUpperCase()+" = "+position.toFixed(3)+" mm":"Schnitt aus";
    if(cutHelper){scene.remove(cutHelper);cutHelper.geometry.dispose();cutHelper.material.dispose();cutHelper=null;}
    if(enabled){
        const size=Math.max(10,...bounds[1].map((v,i)=>v-bounds[0][i]))*1.08;
        cutHelper=new THREE.PlaneHelper(clipPlane,size,0x80c0ff);
        scene.add(cutHelper);
    }
    sectionKey="";
    scheduleSection();
}
function removeSection(){
    if(sectionLines){
        scene.remove(sectionLines);
        sectionLines.geometry.dispose();
        sectionLines.material.dispose();
        sectionLines=null;
    }
}
function scheduleSection(){
    if(sectionPending)return;
    sectionPending=true;
    requestAnimationFrame(()=>{
        sectionPending=false;
        refreshSection();
    });
}
function refreshSection(){
    if(!model || !data || !$("cutEnabled").checked){removeSection();return;}
    if(playing){removeSection();return;}
    const axis=$("cutAxis").value, idx=axis==="x"?0:axis==="y"?1:2;
    const min=model.bounds[0][idx],max=model.bounds[1][idx];
    const position=min+(max-min)*Number($("cutPosition").value)/1000;
    const count=browsingLayer
        ? model.layers[visualLayer].start+model.layers[visualLayer].count
        : completedBefore(model.count,currentTime,endAt);
    const below=Number($("below").value);
    const first=below<0?0:Math.max(0,visualLayer-below);
    const key=[axis,position.toFixed(4),first,visualLayer,count].join(":");
    if(sectionKey===key)return;
    sectionKey=key;
    removeSection();
    const slice=buildCutContours(data,model.layers.slice(first,visualLayer+1),axis,position,count);
    if(slice.vertices.length){
        const geo=new THREE.BufferGeometry();
        geo.setAttribute("position",new THREE.BufferAttribute(slice.vertices,3));
        sectionLines=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({
            color:0xfff5c0,transparent:true,opacity:.88,depthTest:false,depthWrite:false
        }));
        sectionLines.renderOrder=30;
        scene.add(sectionLines);
    }
    $("cutValue").textContent=axis.toUpperCase()+" = "+position.toFixed(3)+
        " mm · "+slice.contours+" Schnittkonturen"+(slice.truncated?" (begrenzt)":"");
}
function showDiagnostics(){
    disposeOverlay(diagnosticObjects);
    const list=$("diagnosticsList");
    list.replaceChildren();
    if(!model || !data){$("diagnosticsSummary").textContent="Kein G-Code geladen";return;}
    if(!$("diagnosticsToggle").checked){
        $("diagnosticsSummary").textContent="Diagnose deaktiviert";
        return;
    }
    if(!diagnosticReport)diagnosticReport=analysePrint(data,hopData,model.layers);
    const {findings,counts,total}=diagnosticReport;
    $("diagnosticsSummary").textContent=total+" Auffälligkeiten ("+
        findings.length+" exemplarisch gelistet) · heuristische Hinweise, keine Druckfehler-Garantie";
    for(const issue of findings){
        const btn=document.createElement("button");
        const dot=document.createElement("span");dot.className="warn-dot";
        const label=document.createElement("span");
        label.textContent=issue.type+": "+issue.text;
        btn.append(dot,label);
        btn.title="Zu "+issue.type+" springen";
        btn.addEventListener("click",()=>{
            setPlaying(false);
            if(playbackMode==="layer")selectedLayer=issue.layer;
            seek(issue.time);
            controls.target.set(issue.position[0],issue.position[2],issue.position[1]);
            const dist=Math.max(3,(model.bounds[1][0]-model.bounds[0][0])*.08);
            camera.position.set(issue.position[0]+dist,issue.position[2]+dist,issue.position[1]+dist);
            controls.update();
        });
        list.append(btn);
    }
    const maxPoints=Math.min(findings.length,400);
    const points=new Float32Array(maxPoints*3);
    for(let i=0;i<maxPoints;i++){
        const p=findings[i].position;
        points.set([p[0],p[2],p[1]],i*3);
    }
    const geo=new THREE.BufferGeometry();
    geo.setAttribute("position",new THREE.BufferAttribute(points,3));
    const obj=new THREE.Points(geo,new THREE.PointsMaterial({
        color:0xffa54d,size:7,sizeAttenuation:false,depthTest:false,clippingPlanes:clipPlanes
    }));
    obj.renderOrder=25;
    scene.add(obj);diagnosticObjects.push(obj);
}
function referenceIndex(){
    const selected=Number(modelSelect.value);
    if(layout.models.length<2)return -1;
    const original=layout.models.findIndex((m,i)=>i!==selected&&m.name==="Original");
    return original>=0?original:layout.models.findIndex((_,i)=>i!==selected);
}
let originalRequest=null;
async function ensureOriginal(){
    const available=layout.models.findIndex(m=>m.name==="Original");
    if(available>=0)return available;
    if(!layout.originalAvailable)throw new Error("Keine Originaldatei verfügbar");
    if(!originalRequest){
        originalRequest=(async()=>{
            const response=await fetch("/prepare-original",{method:"POST"});
            if(!response.ok)throw new Error("Originalanalyse konnte nicht gestartet werden (HTTP "+response.status+")");
            for(;;){
                const result=await fetch("/meta.json",{cache:"no-store"});
                if(!result.ok)throw new Error("Metadaten nicht verfügbar (HTTP "+result.status+")");
                const next=await result.json();
                const previous=modelSelect.value;
                layout.models=next.models;
                layout.originalState=next.originalState;
                const index=layout.models.findIndex(m=>m.name==="Original");
                if(index>=0){
                    refreshModelOptions();
                    modelSelect.value=previous==="original-pending"?"0":previous;
                    return index;
                }
                if(next.originalState==="error")
                    throw new Error(next.originalError||"Originalanalyse fehlgeschlagen");
                await new Promise(resolve=>setTimeout(resolve,500));
            }
        })().catch(error=>{originalRequest=null;throw error;});
    }
    return originalRequest;
}
async function enableComparison(){
    comparisonSerial++;
    const serial=comparisonSerial;
    clearComparison();
    if(!$("compareToggle").checked || !model || !data)return;
    let idx=referenceIndex();
    if(idx<0&&layout.originalAvailable){
        $("comparisonSummary").textContent="Original wird bei Bedarf analysiert…";
        try{idx=await ensureOriginal();}
        catch(error){
            if(serial===comparisonSerial)
                $("comparisonSummary").textContent="Vergleich nicht möglich: "+error.message;
            return;
        }
    }
    if(serial!==comparisonSerial||!$("compareToggle").checked)return;
    if(idx<0){
        $("comparisonSummary").textContent="Bitte zuerst Original- und bearbeiteten G-Code laden.";
        return;
    }
    referenceModel=layout.models[idx];
    $("comparisonSummary").textContent="Lade "+referenceModel.filename+"…";
    try{
        const response=await fetch(referenceModel.url);
        if(!response.ok)throw Error("HTTP "+response.status);
        const buffer=await response.arrayBuffer();
        if(serial!==comparisonSerial)return;
        if(buffer.byteLength!==referenceModel.count*SEGMENT_SIZE)throw Error("Referenz-Binärformat ungültig");
        referenceData=new DataView(buffer);
        comparisonLayer=-1;
        refreshComparison();
    }catch(error){
        if(serial===comparisonSerial)$("comparisonSummary").textContent="Vergleich nicht möglich: "+error.message;
    }
}
function clearComparison(){
    disposeOverlay(comparisonObjects);
    comparisonLayer=-1;
    referenceData=null;
    referenceModel=null;
}
function refreshComparison(){
    if(!$("compareToggle").checked || !model || !data || !referenceData)return;
    if(playing){
        if(comparisonLayer!==-1){disposeOverlay(comparisonObjects);comparisonLayer=-1;}
        return;
    }
    if(comparisonLayer===visualLayer)return;
    disposeOverlay(comparisonObjects);
    comparisonLayer=visualLayer;
    const active=model.layers[visualLayer];
    if(!active)return;
    const refIndex=closestLayer(referenceModel.layers,active.z);
    const refLayer=referenceModel.layers[refIndex];
    const result=compareLayerVolumes(data,active,referenceData,refLayer);
    const positions=[];
    const stride=Math.max(1,Math.ceil(refLayer.count/120000));
    for(let i=refLayer.start;i<refLayer.start+refLayer.count;i+=stride){
        const o=i*SEGMENT_SIZE;
        positions.push(referenceData.getFloat32(o,true),referenceData.getFloat32(o+8,true),
            referenceData.getFloat32(o+4,true),referenceData.getFloat32(o+12,true),
            referenceData.getFloat32(o+20,true),referenceData.getFloat32(o+16,true));
    }
    if(positions.length){
        const geo=new THREE.BufferGeometry();
        geo.setAttribute("position",new THREE.Float32BufferAttribute(positions,3));
        const lines=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({
            color:0x7ae0e8,transparent:true,opacity:.55,depthWrite:false,clippingPlanes:clipPlanes
        }));
        lines.renderOrder=5;
        scene.add(lines);comparisonObjects.push(lines);
    }
    const colors=[],points=[];
    for(const entry of result.diffs){
        points.push(entry.pos[0],entry.pos[2],entry.pos[1]);
        if(entry.delta>0)colors.push(0.3,1,.45);
        else colors.push(1,.28,.38);
    }
    if(points.length){
        const geo=new THREE.BufferGeometry();
        geo.setAttribute("position",new THREE.Float32BufferAttribute(points,3));
        geo.setAttribute("color",new THREE.Float32BufferAttribute(colors,3));
        const cloud=new THREE.Points(geo,new THREE.PointsMaterial({
            vertexColors:true,size:4,sizeAttenuation:false,depthTest:false,clippingPlanes:clipPlanes
        }));
        cloud.renderOrder=22;scene.add(cloud);comparisonObjects.push(cloud);
    }
    $("comparisonSummary").textContent=
        "Layer "+active.number+" ↔ "+refLayer.number+" (Z "+active.z.toFixed(2)+
        " / "+refLayer.z.toFixed(2)+" mm)\n"+
        "Volumen: "+result.processedVolume.toFixed(1)+" / "+result.originalVolume.toFixed(1)+
        " mm³ · Δ "+(result.deltaVolume>=0?"+":"")+result.deltaVolume.toFixed(1)+
        " mm³\nGrün: mehr Material · Rot: weniger · Cyan: Originalpfade\n"+
        "Räumlicher Vergleich in 0,65-mm-Zellen (Näherung).";
}

// Source diagnostics remain on-demand to keep multi-million-line files compact.
function createRawOverlay(index){
    const layer=model.layers[index];
    const indices=[],positions=[],colors=[];
    for(let j=layer.rawStart;j<layer.rawStart+layer.rawCount;j++){
        const record=readRaw(j);
        if((record.kind===0 || record.kind===4) && !$("rawAll").checked)continue;
        indices.push(j);
        positions.push(record.a[0],record.a[2],record.a[1],
                       record.b[0],record.b[2],record.b[1]);
        const c=record.kind===1?[1,.30,.18]:record.kind===2?[1,.38,.85]:
            record.kind===3?[.72,.55,1]:record.kind===4?[.50,.88,1]:[.20,.85,.95];
        colors.push(...c,...c);
    }
    const geo=new THREE.BufferGeometry();
    geo.setAttribute("position",new THREE.Float32BufferAttribute(positions,3));
    geo.setAttribute("color",new THREE.Float32BufferAttribute(colors,3));
    const overlay=new THREE.LineSegments(geo,new THREE.LineBasicMaterial({
        vertexColors:true,depthTest:false,depthWrite:false,transparent:true,opacity:.95,
        clippingPlanes:clipPlanes
    }));
    overlay.frustumCulled=false;overlay.renderOrder=27;
    overlay.userData.indices=Uint32Array.from(indices);
    return overlay;
}
async function loadRawMoves(){
    if(!model||rawData)return;
    if(!model.rawUrl){
        $("rawInfo").textContent="Rohbewegungen erfordern eine neu erzeugte Vorschau.";
        return;
    }
    const serial=++rawRequest,expected=model;
    $("rawInfo").textContent="Analysiere Original-Extrusionsbewegungen…";
    const response=await fetch(model.rawUrl);
    if(!response.ok)throw new Error("HTTP "+response.status+" beim Laden der Rohbahnen");
    const bytes=await response.arrayBuffer();
    if(serial!==rawRequest||model!==expected)return;
    if(bytes.byteLength!==model.rawCount*RAW_SIZE)
        throw new Error("Rohbahnformat ist unvollständig.");
    rawData=new DataView(bytes);
    const count=[0,0,0,0,0];
    for(let i=0;i<model.rawCount;i++)
        count[rawData.getUint32(i*RAW_SIZE+24,true)]++;
    $("rawInfo").textContent=(count[0]+count[4])+" gerendert · "+count[1]+
        " ausgelassen/teilweise · "+count[4]+" echte Bögen · "+count[2]+
        " approximierte Bögen · "+count[3]+" vertikale Extrusionen. "+
        "Rohbahnen anklicken → G-Code-Zeile.";
    updateRawOverlay();
}
function updateRawOverlay(){
    const enabled=$("rawOmitted").checked&&!!rawData&&!!model;
    const keep=new Set();
    if(enabled){
        const below=Number($("below").value);
        const minLayer=below<0?0:Math.max(0,visualLayer-below);
        const finished=browsingLayer
            ? model.layers[visualLayer].rawStart+model.layers[visualLayer].rawCount
            : currentTime>=model.duration-0.0001?model.rawCount:
                completedBefore(model.rawCount,currentTime,
                    i=>rawData.getFloat32(i*RAW_SIZE+36,true));
        for(let i=minLayer;i<=visualLayer;i++){
            let overlay=rawLayerCache.get(i);
            if(!overlay){overlay=createRawOverlay(i);rawLayerCache.set(i,overlay);}
            keep.add(i);
            if(!overlay.parent)scene.add(overlay);
            overlay.geometry.setDrawRange(0,2*withinPrefix(overlay.userData.indices,finished));
        }
    }
    for(const [index,overlay] of rawLayerCache){
        if(!keep.has(index)){
            scene.remove(overlay);
            if(!enabled||rawLayerCache.size>24){
                overlay.geometry.dispose();overlay.material.dispose();rawLayerCache.delete(index);
            }
        }
    }
}
let sourceRequest=0;
async function ensureMotionData(){
    if(moveData || !model?.moveUrl || !model.moveCount)return;
    if(moveLoading)return moveLoading;
    const expected=model;
    const request=(async()=>{
        const response=await fetch(expected.moveUrl,{cache:"no-store"});
        if(!response.ok)throw Error("G-Code-Zeitleiste HTTP "+response.status);
        const bytes=await response.arrayBuffer();
        if(bytes.byteLength!==expected.moveCount*MOVE_SIZE)
            throw Error("G-Code-Zeitleiste unvollständig");
        if(model===expected){
            moveData=new DataView(bytes);
            syncSourceHighlights();
        }
    })();
    moveLoading=request;
    try{await request;}
    finally{if(moveLoading===request)moveLoading=null;}
}
function playedSourceLine(){
    if(!model)return 0;
    if(moveData){
        if(browsingLayer){
            const done=sourceLineAtLayerEnd(moveData,model.moveCount,model.layers[visualLayer]);
            if(done)return done;
        }
        return sourceLineAtTime(moveData,model.moveCount,currentTime);
    }
    // Old cached previews can lack a motion file: fall back to deposition.
    if(!data||!model.count)return 0;
    const last=browsingLayer
        ? model.layers[visualLayer].start+model.layers[visualLayer].count-1
        : Math.min(model.count-1,Math.max(0,visibleCount-1));
    return last>=0?readSegment(last).line:0;
}
function syncSourceHighlights(){
    if(!$("sourceDrawer").classList.contains("visible") || !model)return;
    const next=playedSourceLine();
    const changed=next!==currentSourceLine;
    currentSourceLine=next;
    const playingBadge=$("playingSource"),selectedBadge=$("selectedSource");
    playingBadge.classList.toggle("visible",next>0);
    selectedBadge.classList.toggle("visible",selectedSourceLine>0);
    if(next>0)playingBadge.textContent="▶ Wiedergabe · Zeile "+next;
    if(selectedSourceLine>0)
        selectedBadge.textContent="◆ Ausgewählt · Zeile "+selectedSourceLine;
    let playingRow=null;
    for(const row of $("sourceLines").children){
        const line=Number(row.dataset.line);
        row.classList.toggle("active",line===selectedSourceLine);
        row.classList.toggle("playing",line===next);
        if(line===next)playingRow=row;
    }
    if(changed && playingRow && playing){
        playingRow.scrollIntoView({block:"nearest",behavior:"instant"});
    }
    const now=performance.now();
    // Follow the actual G-code cursor during playback/layer browsing, but
    // don't immediately steal the viewport when a user picks a distant line.
    if(next && (next<sourceWindowStart || next>sourceWindowEnd) &&
       now-lastSourceAutoFetch>300 &&
       (playing||browsingLayer||now-sourceSelectionTime>1300)){
        lastSourceAutoFetch=now;
        openSourceLine(next,false).catch(showError);
    }
}
async function openSourceLine(line,select=true){
    if(!model || !Number.isSafeInteger(line) || line<1)return;
    const request=++sourceRequest,expected=model;
    if(select){
        selectedSourceLine=line;
        sourceSelectionTime=performance.now();
        $("sourceLine").value=String(line);
    }
    $("sourceDrawer").classList.add("visible");
    syncSourceHighlights();
    ensureMotionData().catch(showError);
    $("sourceStatus").textContent=model.filename+" · Lade G-Code bei Zeile "+line+"…";
    const response=await fetch("/source?model="+encodeURIComponent(model.url)+
        "&line="+line+"&context=35",{cache:"no-store"});
    if(!response.ok)throw new Error("G-Code konnte nicht geladen werden (HTTP "+response.status+")");
    const source=await response.json();
    if(request!==sourceRequest||model!==expected)return;
    const content=$("sourceLines");
    content.replaceChildren();
    let chosen=null;
    sourceWindowStart=source.rows[0]?.line||0;
    sourceWindowEnd=source.rows.at(-1)?.line||0;
    for(const row of source.rows){
        const el=document.createElement("div");
        el.className="source-line";
        el.dataset.line=String(row.line);
        const num=document.createElement("span");
        num.className="line-num";num.textContent=row.line;
        const txt=document.createElement("span");txt.textContent=row.text;
        el.append(num,txt);
        el.addEventListener("click",()=>navigateSourceLine(row.line));
        content.append(el);
        if(row.line===line)chosen=el;
    }
    $("sourceStatus").textContent=model.filename+" · "+source.rows.length+
        " Zeilen · Quelle: "+model.name;
    syncSourceHighlights();
    if(chosen)chosen.scrollIntoView({block:"center",behavior:"instant"});
}
$("selectedSource").addEventListener("click",()=>{
    if(selectedSourceLine)openSourceLine(selectedSourceLine,false).catch(showError);
});
$("playingSource").addEventListener("click",()=>{
    const line=playedSourceLine();
    if(line)openSourceLine(line,false).catch(showError);
});
function navigateSourceLine(line){
    openSourceLine(line).catch(showError);
    if(!model||!data)return;
    if(moveData){
        const motion=sourceMotionForLine(moveData,model.moveCount,line);
        if(motion){
            setPlaying(false);
            const at=motion.start+(motion.end-motion.start)*.999;
            if(playbackMode==="layer")selectedLayer=layerAtTime(at);
            seek(at);
            return;
        }
    }
    const segment=nearestSourceRecord(data,SEGMENT_SIZE,36,model.count,line);
    const source=segment>=0?readSegment(segment):null;
    const raw=rawData?nearestSourceRecord(rawData,RAW_SIZE,28,model.rawCount,line):-1;
    const omitted=raw>=0?readRaw(raw):null;
    const point=(omitted&&omitted.line===line && omitted.kind!==0)?omitted:source;
    if(!point)return;
    setPlaying(false);
    if(playbackMode==="layer")selectedLayer=layerAtTime(point.end);
    seek(Math.min(model.duration,point.end+0.00001));
}
$("sourceClose").addEventListener("click",()=>{
    $("sourceDrawer").classList.remove("visible");
    sourceRequest++;
});
$("sourceGo").addEventListener("click",()=>navigateSourceLine(Number($("sourceLine").value)));
$("sourceLine").addEventListener("keydown",event=>{
    if(event.key==="Enter")navigateSourceLine(Number($("sourceLine").value));
});
$("sourceCopy").addEventListener("click",async()=>{
    const lines=[...$("sourceLines").querySelectorAll(".source-line")].map(
        el=>el.querySelector(".line-num").textContent+"  "+el.lastElementChild.textContent).join("\n");
    try{await navigator.clipboard.writeText(lines);}catch(error){showError(error);}
});
$("rawOmitted").addEventListener("change",()=>{
    if($("rawOmitted").checked)loadRawMoves().catch(showError);
    else updateRawOverlay();
});
$("rawAll").addEventListener("change",()=>{
    for(const overlay of rawLayerCache.values()){
        scene.remove(overlay);overlay.geometry.dispose();overlay.material.dispose();
    }
    rawLayerCache.clear();
    if($("rawAll").checked && !$("rawOmitted").checked)$("rawOmitted").checked=true;
    if($("rawOmitted").checked)loadRawMoves().then(updateRawOverlay).catch(showError);
    else updateRawOverlay();
});
const sideGroups=[...document.querySelectorAll("details[data-side-section]")];
for(const group of sideGroups){
    const key="silksteel.sidebar."+group.dataset.sideSection;
    try{
        const saved=layout.sidebar?.[group.dataset.sideSection];
        const fallback=localStorage.getItem(key);
        if(typeof saved==="boolean")group.open=saved;
        else if(fallback!==null)group.open=fallback==="1";
    }catch{}
}
let sidebarSaving=false, sidebarDirty=false;
function saveSidebarState(){
    sidebarDirty=true;
    if(sidebarSaving)return;
    sidebarSaving=true;
    (async()=>{
        try{
            do{
                sidebarDirty=false;
                const state=Object.fromEntries(sideGroups.map(group=>
                    [group.dataset.sideSection,group.open]));
                const response=await fetch("/settings/sidebar",{
                    method:"POST",headers:{"Content-Type":"application/json"},
                    body:JSON.stringify(state)
                });
                if(!response.ok)console.warn("Sidebar-Einstellungen konnten nicht gespeichert werden.");
            }while(sidebarDirty);
        }catch(error){console.warn("Sidebar-Einstellungen:",error);}
        finally{sidebarSaving=false;}
    })();
}
for(const group of sideGroups){
    group.addEventListener("toggle",()=>{
        try{localStorage.setItem("silksteel.sidebar."+group.dataset.sideSection,
            group.open?"1":"0");}catch{}
        saveSidebarState();
    });
}

function formatTimelineTime(seconds) {
    if (seconds < 120) {
        const sec=Math.max(0,seconds);
        return Math.floor(sec/60)+":"+String(Math.floor(sec%60)).padStart(2,"0")+
            "."+String(Math.floor((sec%1)*100)).padStart(2,"0");
    }
    return formatTime(seconds);
}
function changeTimelineMode(mode) {
    if (!model || mode===playbackMode || (mode!=="print"&&mode!=="layer")) return;
    setPlaying(false);
    selectedLayer=browsingLayer?selectedLayer:layerAtTime(currentTime);
    playbackMode=mode;
    if(mode==="layer") $("playSpeed").value="1";
    else $("playSpeed").value="100";
    $("modePrint").setAttribute("aria-pressed",String(mode==="print"));
    $("modeLayer").setAttribute("aria-pressed",String(mode==="layer"));
    $("timelineMarkers").classList.toggle("layer-mode",mode==="layer");
    markerKey="";
    if(browsingLayer)chooseLayer(selectedLayer);
    else seek(mode==="layer" ? layerRange(selectedLayer)[0] : currentTime);
}
function chooseLayer(index) {
    if(!model)return;
    setPlaying(false);
    selectedLayer=Math.max(0,Math.min(model.layers.length-1,index));
    visualLayer=selectedLayer;
    browsingLayer=true;
    // Layer navigation is a completed-layer inspection operation, not a
    // playback seek. The following layer must remain completely hidden even
    // when its start timestamp equals this layer's end timestamp.
    currentTime=Math.min(model.duration,model.layers[selectedLayer].timeEnd);
    markerKey="";
    updateTimelineUI();
    updateVisibility();
}
function markerLabel(text,x,width,onClick) {
    const mark=document.createElement("button");
    mark.type="button"; mark.className="timeline-marker";
    mark.style.left=(100*Math.max(9,Math.min(width-9,x))/Math.max(1,width)).toFixed(5)+"%";
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
        for(let i=labeled.length-1;i>=0;i--){
            if(Math.abs(labeled[i].x-selected.x)<minDistance)labeled.splice(i,1);
        }
        labeled.push(selected);
    }
    for(const point of labeled){
        markerLabel(point.name,point.x,width,()=>{
            chooseLayer(point.index);
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
    browsingLayer=false;
    currentTime=Math.max(0,Math.min(model.duration,seconds));
    visualLayer=playbackMode==="layer"?selectedLayer:layerAtTime(currentTime);
    if(playbackMode==="print")selectedLayer=visualLayer;
    updateTimelineUI();
    updateVisibility();
}
function updateVisibility() {
    if(!model || !data)return;
    // Binary timestamps are float32, while metadata.duration is float64.
    // Always include the COMPLETE final toolpath at the end of playback.
    const finished=browsingLayer
        ? model.layers[visualLayer].start+model.layers[visualLayer].count
        : currentTime>=model.duration-0.0001?model.count:
            completedBefore(model.count,currentTime,endAt);
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
                mesh.material=newMaterial(mesh.userData.role,faded,
                    mesh.userData.cap?capVertexShader:vertexShader);
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
    frontierCap.visible=false;
    if(!browsingLayer && finished<model.count){
        const p=readSegment(finished);
        if(p.start<=currentTime&&currentTime<p.end&&visibleRoles.has(p.role)){
            const alpha=Math.max(0,Math.min(1,(currentTime-p.start)/(p.end-p.start)));
            if(alpha>0){
                headMesh.material.uniforms.uColor.value.copy(rgb[p.role]);
                headMesh.userData.role=p.role;
                renderTip(headMesh,p,alpha,finished);
            }
        }
    }
    if(!browsingLayer&&!headMesh.visible&&finished>0){
        const last=finished-1;
        const layer=model.layers.find(l=>last>=l.start&&last<l.start+l.count);
        if(layer && layer.number>=model.layers[start].number &&
           layer.number<=model.layers[current].number &&
           visibleRoles.has(readRole(last))){
            renderFrontierCap(last);
        }
    }
    const hopsFinished=browsingLayer
        ? model.layers[visualLayer].hopStart+model.layers[visualLayer].hopCount
        : currentTime>=model.duration-0.0001?model.hopCount:
            completedBefore(model.hopCount,currentTime,hopEndAt);
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
    if(!browsingLayer&&showHops&&hopsFinished<model.hopCount){
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
    if($("compareToggle").checked && referenceData)refreshComparison();
    if($("cutEnabled").checked && !playing)scheduleSection();
    updateRawOverlay();
    syncSourceHighlights();
}
function fit() {
    const [lo,hi] = model.bounds;
    const cx=(lo[0]+hi[0])/2,cy=(lo[1]+hi[1])/2,cz=(lo[2]+hi[2])/2;
    const size=Math.max(hi[0]-lo[0],hi[1]-lo[1],hi[2]-lo[2],10);
    controls.target.set(cx,cz,cy);
    camera.position.set(cx+size*.6,cz+size*.85,cy+size*.85);
    camera.near=Math.max(.01,size/15000);camera.far=size*150;camera.updateProjectionMatrix();
    controls.update();
    // The grid represents the physical print-bed plane, not the lowest
    // extrusion Z minus an arbitrary display offset.
    grid.position.set(cx,0,cy);
    grid.scale.set(Math.max(1,size/300),1,Math.max(1,size/300));
}
async function loadModel(index) {
    const loadStarted=performance.now();
    forgetLayers();
    model=layout.models[index];
    $("progress").hidden=false;
    $("progress").textContent="Lade "+model.name+" · "+model.count.toLocaleString("de-DE")+" Stränge…";
    const [response,hopResponse]=await Promise.all([fetch(model.url),fetch(model.hopUrl)]);
    if(!response.ok||!hopResponse.ok)throw Error("HTTP "+response.status+" / "+hopResponse.status);
    const buffers=await Promise.all([response.arrayBuffer(),hopResponse.arrayBuffer()]);
    data=new DataView(buffers[0]);
    hopData=new DataView(buffers[1]);
    const transferMs=performance.now()-loadStarted;
    if(data.byteLength!==model.count*SEGMENT_SIZE || hopData.byteLength!==model.hopCount*HOP_SIZE){
        throw new Error("G-Code-Vorschau: Ungültiges Binärformat oder Segmentgröße");
    }
    $("fileName").textContent=model.filename;
    $("stats").textContent=model.count.toLocaleString("de-DE")+" Extrusionssegmente · "+model.layers.length+" Layer · Filament Ø "+model.filament.toFixed(2)+" mm"+
        (model.omittedCount?" · "+model.omittedCount+" nicht gerenderte Bewegungen":"")+
        (model.partialCount?" · "+model.partialCount+" teilweise unvollständige Bahnen":"")+
        (model.arcCount?" · "+model.arcCount+" Bögen gerendert":"")+
        (model.ignored_arcs?" · Achtung: "+model.ignored_arcs+" Bögen als gerade Näherung":"");
    if($("rawOmitted").checked)loadRawMoves().catch(showError);
    slider.max=String(model.layers.length-1);
    selectedLayer=Math.max(0,model.layers.length-1);
    markerKey="";
    fit();
    refreshCut();
    seek(model.duration);
    console.info(`[Viewer] ${model.name}: parse ${model.parseSeconds??"n/a"}s; transfer ${transferMs.toFixed(1)}ms; initial geometry ${(performance.now()-loadStarted-transferMs).toFixed(1)}ms; total ${(performance.now()-loadStarted).toFixed(1)}ms`);
    $("diagnosticsSummary").textContent="Diagnose bei Bedarf aktivieren";
    if($("diagnosticsToggle").checked)showDiagnostics();
    if($("compareToggle").checked)enableComparison();
    $("progress").hidden=true;
}
modelSelect.addEventListener("change",()=>{
    setPlaying(false);
    if(modelSelect.value==="original-pending"){
        const index=layout.models.indexOf(model);
        modelSelect.value=String(index<0?0:index);
        ensureOriginal().then(index=>{
            modelSelect.value=String(index);
            return loadModel(index);
        }).catch(showError);
        return;
    }
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
$("seamlessJoins").addEventListener("change",()=>{
    if(!model)return;
    for(const group of cache.values()){scene.remove(group);disposeGroup(group);}
    cache.clear();
    updateVisibility();
});
for(const id of ["cutEnabled","cutInvert","cutAxis","cutPosition"]){
    $(id).addEventListener("input",refreshCut);
}
$("diagnosticsToggle").addEventListener("change",showDiagnostics);
$("compareToggle").addEventListener("change",()=>{
    if($("compareToggle").checked)enableComparison();
    else{
        comparisonSerial++;clearComparison();
        $("comparisonSummary").textContent="Vergleich deaktiviert";
    }
});
function showError(error) {
    $("progress").hidden=false;
    $("progress").textContent="Vorschaufehler: "+error.message;
    console.error(error);
}
const mouse=new THREE.Vector2();
function auditGPUInstance(index, expected) {
    let found=false,drawn=false,correct=false;
    let reason="Instanz nicht in aktiven Layer-Meshes";
    for(const group of activeLayers){
        for(const mesh of group.children){
            if(mesh.userData.cap || !mesh.userData.indices)continue;
            const ids=mesh.userData.indices;
            const local=withinPrefix(ids,index);
            if(local>=ids.length || ids[local]!==index)continue;
            found=true;
            const starts=mesh.geometry.attributes.iStart.array;
            const ends=mesh.geometry.attributes.iStop.array;
            const sizes=mesh.geometry.attributes.iSize.array;
            const j=local*3,k=local*2;
            const close=(a,b)=>Math.abs(a-b)<0.00006;
            correct=expected.a.every((v,i)=>close(v,starts[j+i])) &&
                expected.b.every((v,i)=>close(v,ends[j+i])) &&
                close(expected.w,sizes[k])&&close(expected.h,sizes[k+1]);
            drawn=mesh.visible && local<mesh.geometry.instanceCount && !!group.parent;
            reason=!correct?"GPU-Attribute weichen vom Binärsegment ab":
                !drawn?"Instanz durch Layer-/Playback-Filter ausgeblendet":
                "GPU-Instanz aktiv (Verdeckung durch Geometrie möglich)";
            break;
        }
        if(found)break;
    }
    return {found,drawn,correct,reason};
}
function selectSegment(event) {
    if(!model||!data)return;
    const bounds=renderer.domElement.getBoundingClientRect();
    mouse.set(((event.clientX-bounds.left)/bounds.width)*2-1,
              -((event.clientY-bounds.top)/bounds.height)*2+1);
    const current=Number(slider.value), layer=model.layers[current];
    let best=14*14, pick=null, isRaw=false;
    const a=new THREE.Vector3(),b=new THREE.Vector3();
    const test=p=>{
        a.set(p.a[0],p.a[2],p.a[1]).project(camera);
        b.set(p.b[0],p.b[2],p.b[1]).project(camera);
        const dx=b.x-a.x,dy=b.y-a.y;
        const t=Math.max(0,Math.min(1,((mouse.x-a.x)*dx+(mouse.y-a.y)*dy)/(dx*dx+dy*dy||1)));
        const x=a.x+t*dx,y=a.y+t*dy;
        return ((x-mouse.x)*bounds.width/2)**2+((y-mouse.y)*bounds.height/2)**2;
    };
    if($("rawOmitted").checked && rawData){
        const rawEnd=currentTime>=model.duration-0.0001?model.rawCount:
            completedBefore(model.rawCount,currentTime,
                i=>rawData.getFloat32(i*RAW_SIZE+36,true));
        for(let i=layer.rawStart;i<Math.min(layer.rawStart+layer.rawCount,rawEnd);i++){
            const p=readRaw(i);
            if((p.kind===0 || p.kind===4) && !$("rawAll").checked)continue;
            const distance=test(p);
            if(distance<best){best=distance;pick=p;pick.rawIndex=i;isRaw=true;}
        }
    }
    for(let i=layer.start;i<Math.min(layer.start+layer.count,visibleCount);i++){
        const p=readSegment(i);
        if(!visibleRoles.has(p.role))continue;
        const distance=test(p);
        if(distance<best){best=distance;pick=p;pick.segmentIndex=i;isRaw=false;}
    }
    if(lastPick){scene.remove(lastPick);lastPick.geometry.dispose();lastPick.material.dispose();lastPick=null;}
    if(!pick){$("pickInfo").style.display="none";return;}
    const a3=new THREE.Vector3(pick.a[0],pick.a[2],pick.a[1]),b3=new THREE.Vector3(pick.b[0],pick.b[2],pick.b[1]);
    const geom=new THREE.BufferGeometry().setFromPoints([a3,b3]);
    lastPick=new THREE.Line(geom,new THREE.LineBasicMaterial({color:"#ffffff",depthTest:false}));
    lastPick.renderOrder=999;scene.add(lastPick);
    const length=Math.hypot(...pick.a.map((v,i)=>pick.b[i]-v));
    $("pickInfo").style.display="block";
    $("pickInfo").replaceChildren();
    const title=document.createElement("strong");
    title.textContent=(isRaw?"Rohbewegung":layout.roles[pick.role])+" · G-Code-Zeile "+pick.line;
    const body=document.createElement("div");
    if(isRaw){
        body.textContent=rawClass(pick.kind)+" · Länge "+length.toFixed(2)+" mm";
    }else{
        const area=profileArea(pick.w,pick.h);
        const extruded=area*length/(Math.PI*(model.filament/2)**2);
        const form=pick.h>pick.w*1.06?"vertikal gedehnt":
            pick.w>pick.h*1.06?"abgeflacht / aufliegend":"annähernd quadratisch";
        body.textContent="Profil "+form+" · Breite "+pick.w.toFixed(3)+" mm · Höhe "+
            pick.h.toFixed(3)+" mm · Länge "+length.toFixed(2)+" mm · ΔE "+
            extruded.toFixed(5)+" mm";
    }
    $("pickInfo").append(title,body);
    if(!isRaw){
        const audit=auditGPUInstance(pick.segmentIndex,pick);
        const status=document.createElement("div");
        status.style.marginTop="6px";
        status.style.color=audit.found&&audit.drawn&&audit.correct?"#9ed8ad":"#ffac8c";
        status.textContent="BIN #"+pick.segmentIndex+" → "+
            (audit.found?"GPU-Instanz gefunden":"KEINE GPU-Instanz")+
            " · "+audit.reason;
        $("pickInfo").append(status);
    }
    openSourceLine(pick.line).catch(showError);
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
let lastLodUpdate=0;
renderer.setAnimationLoop(()=>{
    const now=performance.now();
    if(now-lastLodUpdate>200){
        updateLoftLods();
        lastLodUpdate=now;
    }
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
