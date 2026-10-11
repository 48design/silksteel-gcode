import * as THREE from "three";
import {OrbitControls} from "three/addons/controls/OrbitControls.js";

const $ = id => document.getElementById(id);
const colors = ["#adb8c6","#ae7bff","#3bcbec","#298df2","#ffb75e","#f28dc2","#67db8c"];
const layout = await (await fetch("/meta.json")).json();
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
let model = null, data = null, cache = new Map(), groups = [], lastPick = null;
let activeLayers = [];
const visibleRoles = new Set(layout.roles.map((_, i) => i));
const rgb = colors.map(c => new THREE.Color(c));

layout.models.forEach((item, i) => {
    const opt = document.createElement("option");
    opt.value = String(i); opt.textContent = item.name; modelSelect.append(opt);
});
layout.roles.forEach((name, i) => {
    const label = document.createElement("label");
    const check = document.createElement("input");
    check.type = "checkbox"; check.checked = true;
    check.addEventListener("change", () => {
        if (check.checked) visibleRoles.add(i); else visibleRoles.delete(i);
        updateVisibility();
    });
    const dot = document.createElement("span");
    dot.className = "dot"; dot.style.background = colors[i];
    label.append(check, dot, document.createTextNode(name));
    $("legend").append(label);
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
    uniforms:{uColor:{value:rgb[role]},uAlpha:{value:faded?.19:1}},
    transparent:!!faded,depthWrite:!faded
});
function disposeGroup(group) {
    group.traverse(o => {
        if(o.isMesh){o.geometry.dispose();o.material.dispose();}
    });
}
function forgetLayers() {
    for (const group of cache.values()) {scene.remove(group);disposeGroup(group);}
    cache.clear(); groups=[]; activeLayers=[];
    if(lastPick){scene.remove(lastPick);lastPick.geometry.dispose();lastPick.material.dispose();lastPick=null;}
}
function readSegment(index) {
    const off=index*40;
    return {
      a:[data.getFloat32(off,true),data.getFloat32(off+4,true),data.getFloat32(off+8,true)],
      b:[data.getFloat32(off+12,true),data.getFloat32(off+16,true),data.getFloat32(off+20,true)],
      w:data.getFloat32(off+24,true),
      h:data.getFloat32(off+28,true),
      role:data.getUint32(off+32,true),
      line:data.getUint32(off+36,true)
    };
}
function createLayer(index) {
    const info = model.layers[index];
    const buckets = Array.from({length:layout.roles.length},()=>[]);
    for(let i=info.start;i<info.start+info.count;i++){
        const role = data.getUint32(i*40+32,true);
        if(buckets[role]) buckets[role].push(i);
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
function updateVisibility() {
    const current=Number(slider.value);
    const below=Number($("below").value);
    const start=below<0?0:Math.max(0,current-below);
    const keep = new Set();
    activeLayers=[];
    for(let i=start;i<=current;i++){
        const group=getLayer(i);keep.add(i);activeLayers.push(group);
        if(!group.parent) scene.add(group);
        for(const mesh of group.children){
            mesh.visible=visibleRoles.has(mesh.userData.role);
            const faded=i!==current && $("ghost").checked;
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
            if(cache.size>24 || below===0){disposeGroup(group);cache.delete(i);}
        }
    }
    $("layerText").textContent = "Layer "+ model.layers[current].number+" / "+model.layers.at(-1).number;
    $("zText").textContent = "Z "+model.layers[current].z.toFixed(3)+" mm · Höhe "+model.layers[current].height.toFixed(3)+" mm · "+model.layers[current].count.toLocaleString("de-DE")+" Segmente";
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
    const response=await fetch(model.url);
    if(!response.ok)throw Error("HTTP "+response.status);
    data=new DataView(await response.arrayBuffer());
    $("fileName").textContent=model.filename;
    $("stats").textContent=model.count.toLocaleString("de-DE")+" Extrusionssegmente · "+model.layers.length+" Layer · Filament Ø "+model.filament.toFixed(2)+" mm"+(model.ignored_arcs?" · Achtung: "+model.ignored_arcs+" Bögen nicht dargestellt":"");
    slider.max=String(model.layers.length-1);
    slider.value=slider.max;
    fit();
    updateVisibility();
    $("progress").hidden=true;
}
modelSelect.addEventListener("change",()=>loadModel(Number(modelSelect.value)).catch(showError));
slider.addEventListener("input",()=>updateVisibility());
$("below").addEventListener("change",updateVisibility);
$("ghost").addEventListener("change",updateVisibility);
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
    for(let i=layer.start;i<layer.start+layer.count;i++){
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
renderer.setAnimationLoop(()=>{controls.update();renderer.render(scene,camera);});
// The detached Python server releases its temporary binary files after inactivity.
setInterval(() => fetch("/ping", {cache:"no-store"}).catch(() => {}), 20000);
loadModel(0).catch(showError);
