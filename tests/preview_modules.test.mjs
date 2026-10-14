import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {dirname,join} from "node:path";

const root=dirname(dirname(fileURLToPath(import.meta.url)));
async function load(file){
    let code=readFileSync(join(root,"preview",file),"utf-8");
    // Data-URL ES modules cannot resolve relative imports. Inline the shared
    // profile module URI for isolated Node tests without package.json changes.
    const profile=readFileSync(join(root,"preview","bead_profile.js"),"utf-8");
    const profileUri="data:text/javascript;base64,"+
        Buffer.from(profile).toString("base64");
    code=code.replaceAll('"./bead_profile.js"',JSON.stringify(profileUri));
    const joins=readFileSync(join(root,"preview","extrusion_joins.js"),"utf-8")
        .replaceAll('"./bead_profile.js"',JSON.stringify(profileUri));
    const joinsUri="data:text/javascript;base64,"+Buffer.from(joins).toString("base64");
    code=code.replaceAll('"./extrusion_joins.js"',JSON.stringify(joinsUri));
    return import("data:text/javascript;base64,"+Buffer.from(code).toString("base64"));
}
const {computeJoin,joinContinuity,buildJoinAttributes,sharedSplitSection}=await load("extrusion_joins.js");
const {loftBoundary,planLoft,spatialChunks}=await load("loft_geometry.js");
const {makeJunctionGeometry,junctionVertexShader}=await load("bead_junctions.js");
const {analysePrint}=await load("diagnostics.js");
const {compareLayerVolumes}=await load("comparison.js");
const {buildCutContours}=await load("section.js");
const {nearestSourceRecord,rawClass}=await load("gcode_inspector.js");
const {profileArea,profileWidth,profileOutline,CORNER_LOSS}=await load("bead_profile.js");
const {MOVE_SIZE,sourceLineAtTime,sourceLineAtLayerEnd,sourceMotionForLine}=await load("source_timeline.js");

// Source cursor follows every timed G-code movement (including G0 and G4),
// not just the subset of extrusion segments that have instanced geometry.
const motionBuffer=new ArrayBuffer(MOVE_SIZE*5),motions=new DataView(motionBuffer);
const motionLines=[101,102,105,107,111],motionTimes=[0.2,1.2,1.8,2.3,3.6];
for(let i=0;i<motionLines.length;i++){
    motions.setFloat32(i*MOVE_SIZE,i?motionTimes[i-1]:0,true);
    motions.setFloat32(i*MOVE_SIZE+4,motionTimes[i],true);
    motions.setUint32(i*MOVE_SIZE+8,motionLines[i],true);
}
assert.equal(sourceLineAtTime(motions,5,.05),101);
assert.equal(sourceLineAtTime(motions,5,.7),102);
assert.equal(sourceLineAtTime(motions,5,1.5),105); // travel
assert.equal(sourceLineAtTime(motions,5,2.1),107); // G4 dwell
assert.equal(sourceLineAtTime(motions,5,3.6),111);
assert.equal(sourceLineAtLayerEnd(motions,5,{moveStart:0,moveCount:3}),105);
assert.equal(sourceLineAtLayerEnd(motions,5,{moveStart:3,moveCount:2}),111);
assert.equal(sourceLineAtLayerEnd(motions,5,{moveStart:5,moveCount:0}),0);
assert.equal(sourceLineAtTime(null,0,0),0);
assert.deepEqual(sourceMotionForLine(motions,5,105),{line:105,start:Math.fround(1.2),end:Math.fround(1.8)});
assert.equal(sourceMotionForLine(motions,5,106).line,107);
assert.equal(sourceMotionForLine(motions,5,999).line,111);
assert.equal(sourceMotionForLine(motions,0,102),null);

const bead=(a,b,role=0,start=0,end=1)=>({a,b,w:.46,h:.20,role,start,end});
// One material-conserving rounded rectangle for all aspect ratios.
for(const [width,height] of [[.55,.2],[.32,.32],[.16,.6],[.09,.9]]){
    const a=profileArea(width,height);
    assert(a>0);
    assert(Math.abs(profileWidth(a,height)-width)<1e-10);
    const corners=profileOutline(width,height);
    assert.equal(corners.length,24);
    const maxY=Math.max(...corners.map(p=>p[1]));
    const minY=Math.min(...corners.map(p=>p[1]));
    assert(Math.abs(maxY)<1e-10);          // nozzle-anchored TOP
    assert(Math.abs(minY+height)<1e-10);   // gravity only toward -Z
}
assert(Math.abs(CORNER_LOSS-(4-Math.PI)*.16)<1e-12);
const tracks=[
    bead([0,0,.2],[10,0,.2]),
    bead([10,0,.2],[20,0,.2],0,1,2),
    bead([22,0,.2],[30,0,.2],0,2.5,3)
];
const reader=i=>tracks[i]||null;
assert(computeJoin(reader,0,0,true));
assert(computeJoin(reader,1,0,false));
assert.equal(computeJoin(reader,1,0,true),null);
assert.equal(computeJoin(reader,2,0,false),null);
const joins=buildJoinAttributes([0,1,2],reader,0);
assert.equal(joins.joinsEnd[2],1);
assert.equal(joins.joinsStart[5],1);
assert.equal(joins.joinsEnd[5],0);
assert.equal(computeJoin(i=>[
    bead([0,0,.2],[10,0,.2]),bead([10,0,.2],[10,10,.2],0,1,2)
][i]||null,0,0,true).miter.length,2);

// GPU loft: shared rings, *only* true path-end caps and spatial chunks.
const straight=[
    {...bead([0,0,.2],[20,0,.2],0,0,1),line:1},
    {...bead([20,0,.2],[42,0,.2],0,1,2),line:2},
    {...bead([42,0,.2],[62,0,.2],0,2,3),line:3}
];
const straightRead=i=>straight[i]||null;
const loft=planLoft([0,1,2],straightRead,0,3,true);
assert.equal(loft.starts.length,9);
assert.equal(loft.caps.length,2);                 // not 6 internal caps
assert.deepEqual(loft.caps.map(x=>[x.index,x.end]),[[0,0],[2,1]]);
assert(Math.abs(loft.joinsEnd[0]-loft.joinsStart[3])<1e-6);
assert(Math.abs(loft.sizesEnd[0]-loft.sizesStart[2])<1e-6);
assert.equal(loftBoundary(straightRead,0,true,0,3).connected,true);
assert.equal(loftBoundary(straightRead,2,true,0,3).connected,false);
const legacyLoft=planLoft([0,1,2],straightRead,0,3,false);
assert.equal(legacyLoft.caps.length,6);           // debug fallback
const corner=[
    {...bead([0,0,.2],[10,0,.2],0,0,1),line:5},
    {...bead([10,0,.2],[10,10,.2],0,1,2),line:6}
];
const joined=planLoft([0,1],i=>corner[i]||null,0,2,true);
assert.equal(joined.caps.length,2);
assert(Math.abs(joined.joinsEnd[0]-joined.joinsStart[3])<1e-6);
assert(Math.abs(joined.joinsEnd[1]-joined.joinsStart[4])<1e-6);
// Real consecutive Smoothificator moves 727881–727883 have nearly
// identical E-per-mm. A noisy supporting raster may give one neighboring
// G1 a taller profile: this MUST NOT break continuity or create two caps.
const smooth=[
    {...bead([82.872,104.544,25.773],[82.236,104.324,25.773],1,0,.017),line:727881,w:.20,h:.24},
    {...bead([82.236,104.324,25.773],[81.58,104.178,25.773],1,.017,.034),line:727882,w:.39,h:.0933},
    {...bead([81.58,104.178,25.773],[80.912,104.108,25.773],1,.034,.051),line:727883,w:.20,h:.24}
];
const smoothRead=i=>smooth[i]||null;
const smoothLoft=planLoft([0,1,2],smoothRead,0,3,true);
assert.deepEqual(smoothLoft.caps.map(c=>[c.index,c.end]),[[0,0],[2,1]]);
for(let i=0;i<2;i++){
    for(let d=0;d<3;d++){
        assert.equal(smoothLoft.joinsEnd[i*3+d],smoothLoft.joinsStart[(i+1)*3+d]);
        assert.equal(smoothLoft.ends[i*3+d],smoothLoft.starts[(i+1)*3+d]);
    }
    for(let d=0;d<2;d++)
        assert.equal(smoothLoft.sizesEnd[i*2+d],smoothLoft.sizesStart[(i+1)*2+d]);
}
// Each new shared profile has an area matching the two G-code profiles'
// average, not a spike derived from only the wider side.
const expectedSmoothArea=(profileArea(smooth[0].w,smooth[0].h)+
    profileArea(smooth[1].w,smooth[1].h))/2;
assert(Math.abs(profileArea(smoothLoft.sizesEnd[0],smoothLoft.sizesEnd[1])-
    expectedSmoothArea)<1e-6);
const curvature=[
    {...bead([0,0,.3],[.20,0,.3],4,0,.01),w:.55,h:.12,line:1},
    {...bead([.20,0,.3],[.20,.22,.3],4,.01,.02),w:.55,h:.12,line:2}
];
const turn=planLoft([0,1],i=>curvature[i]||null,0,2,true);
assert.equal(turn.caps.length,2);
assert(Math.hypot(turn.joinsEnd[0],turn.joinsEnd[1])<=1.42+1e-5);
const reverse=[
    {...bead([0,0,.2],[10,0,.2],0,0,1),line:5},
    {...bead([10,0,.2],[0,0,.2],0,1,2),line:6}
];
assert.equal(planLoft([0,1],i=>reverse[i]||null,0,2,true).caps.length,4);
const spatial=spatialChunks(0,3,straightRead,40,6000);
assert.equal(spatial.length,2);
assert.deepEqual(spatial.flatMap(c=>c.indices).sort((a,b)=>a-b),[0,1,2]);
// Even when grouped in separate spatial chunks, adjoining rings agree.
for(const batch of spatial){
    const part=planLoft(batch.indices,straightRead,0,3,true);
    if(batch.indices.includes(1))assert(!part.caps.some(c=>c.index===1));
}
// Rounded coordinates still form an exactly watertight shared GPU ring.
const rounded=[
    {...bead([0,0,.2],[10,0,.2],0,0,1),line:22},
    {...bead([10.006,0,.2],[20,0,.2],0,1,2),line:23}
];
const snapped=planLoft([0,1],i=>rounded[i]||null,0,2,true);
assert.equal(snapped.caps.length,2);
assert.equal(snapped.ends[0],snapped.starts[3]);
assert(Math.abs(snapped.ends[0]-10.003)<1e-5);
for(let k=0;k<3;k++)assert.equal(snapped.joinsEnd[k],snapped.joinsStart[3+k]);
for(let k=0;k<2;k++)assert.equal(snapped.sizesEnd[k],snapped.sizesStart[2+k]);
const many=Array.from({length:9},(_,i)=>({
    ...bead([i,0,.2],[i+1,0,.2],0,i,i+1),line:i+1
}));
const different=spatialChunks(0,many.length,i=>many[i],100,3);
assert.deepEqual(different.map(c=>c.indices.length),[3,3,3]);
assert(loft.bounds && loft.bounds.padding>=.46);
const tallLoft=planLoft([0],i=>i===0?{
    ...bead([0,0,1.2],[10,0,1.2],0,0,1),w:.12,h:.9,line:17
}:null,0,1,true);
assert(tallLoft.bounds.padding>=.9); // culling must include full downward stretch

// Playback must not expose a missing end cap when the next G1 is hidden.
const viewer=readFileSync(join(root,"preview","viewer.js"),"utf-8");
assert(viewer.includes("vLayerPixels=h*projectionMatrix[1][1]*uViewportHeight"));
assert(viewer.includes("float detail=smoothstep(1.5,8.0,vLayerPixels)"));
assert(viewer.includes("float specDetail=smoothstep(1.0,4.5,vLayerPixels)"));
assert(viewer.includes("float normalVariation=dot(fwidth(n),fwidth(n))"));
assert(viewer.includes("vec3 sheen="));

// Dimmed underlayers must be solid, not alpha-blended, and retain relief.
assert(viewer.includes("uMuted:{value:faded?1:0}"));
assert(viewer.includes("transparent:false,depthWrite:true"));
assert(viewer.includes("float darkShade="));
assert(viewer.includes("frontierCap.material.uniforms.uMuted.value="));

// Three.js X/Z/Y without a Y sign reversal mirrors the slicer's XY paths.
assert(viewer.includes("scene.scale.z=-1"));
assert(viewer.includes("controls.target.set(cx,cz,-cy)"));
assert(viewer.includes("a.set(p.a[0],p.a[2],-p.a[1]).project(camera)"));
assert(viewer.includes("b.set(p.b[0],p.b[2],-p.b[1]).project(camera)"));
assert(viewer.includes('axis==="y"?-position:position'));

// The selected model must never be published before its binary tracks arrive.
const loadStart=viewer.indexOf("async function loadModel(index) {");
const loadEnd=viewer.indexOf('modelSelect.addEventListener("change"',loadStart);
const loadBody=viewer.slice(loadStart,loadEnd);
assert(loadStart>=0&&loadEnd>loadStart);
assert(loadBody.includes("const generation=++modelLoadGeneration"));
assert(loadBody.includes("if(generation!==modelLoadGeneration)return"));
assert(loadBody.indexOf("model=requested;")>loadBody.indexOf("nextData.byteLength"));
assert(loadBody.indexOf("data=nextData;")>loadBody.indexOf("model=requested;"));
assert(loadBody.indexOf("hopData=nextHops;")>loadBody.indexOf("data=nextData;"));

// Keyboard shortcuts: layers and animation, without hijacking form controls.
assert(viewer.includes('document.addEventListener("keydown", event=>{'));
assert(viewer.includes('chooseLayer(current+(event.key==="ArrowUp"?1:-1))'));
assert(viewer.includes('event.key==="ArrowRight"?1:-1'));
assert(viewer.includes('"textarea,select,[contenteditable],input:not([type=range]):not([type=checkbox]):not([type=radio])"'));

const visibilityStart=viewer.indexOf("function updateVisibility() {");
const visibilityEnd=viewer.indexOf("function ",visibilityStart+1);
const visibilityBody=viewer.slice(visibilityStart,visibilityEnd);
assert(visibilityStart>=0);
assert(visibilityBody.includes("model.layers[i].z"));
// Catches the regression that broke the entire viewer/timeline during
// faded-material reconstruction (no local 'info' exists in this scope).
assert(!/\binfo\.z\b/.test(visibilityBody));
// Z-hops follow only the currently selected layer; "Darunter anzeigen"
// and the underlayer dimming setting affect deposited material, not hops.
assert(visibilityBody.includes("const hopLayer=model.layers[current]"));
assert(visibilityBody.includes("if(showHops&&hopLayer.hopCount)"));
assert(visibilityBody.includes("hopCache.get(current)"));
assert(visibilityBody.includes("Math.min(hopLayer.hopCount,hopsFinished-hopLayer.hopStart)"));
assert(visibilityBody.includes("if(!showHops||i!==current)"));
assert(visibilityBody.includes("if(startIndex===current)"));
assert(!visibilityBody.includes('keep.add("h"+i)'));



assert(viewer.includes("vec3 p=basePoint + joinVector*lateral"));
assert(viewer.includes('const base=sectionMesh(4,"mantle")'));
assert(viewer.includes('const capBase=sectionMesh(4,"cap")'));
assert(viewer.includes('const headBase=sectionMesh(4,"closed")'));
assert(viewer.includes("spatialChunks(first,last,cachedRead,40,6000)"));
assert(viewer.includes("bindLoftAttributes(capGeometry,plan,plan.caps)"));
assert(viewer.includes("mesh.frustumCulled=true"));
assert(viewer.includes("updateLoftLods();"));
assert(viewer.includes("const mantleLods=[sectionMesh(2,"));
assert(viewer.includes("const capLods=[sectionMesh(2,"));
assert(viewer.includes("geometry.boundingSphere=new THREE.Sphere("));
assert(viewer.includes("new Set(o.userData.lodVariants?.filter(Boolean)"));
assert(viewer.includes("renderTip(headMesh,p,alpha,finished)"));
assert(viewer.includes("frontierCap.visible=false"));
assert(viewer.includes("renderFrontierCap(last)"));
assert(!viewer.includes("aCap>0.5 && join.z>0.5"));
const sharp=[
    bead([0,0,.2],[10,0,.2]),
    bead([10,0,.2],[9,.1,.2],0,1,2)
];
const sharpReader=i=>sharp[i]||null;
assert(joinContinuity(sharpReader,0,true));
assert.equal(computeJoin(sharpReader,0,0,true),null);
const sharpJoin=buildJoinAttributes([0],sharpReader,0);
assert.equal(sharpJoin.fallbacks.length,1);
assert.equal(sharpJoin.fallbacks[0].index,0);
assert.equal(sharpJoin.joinsEnd[2],0);

const varied=[
    bead([0,0,.2],[10,0,.2]),
    {...bead([10,0,.2],[15,2,.2],0,1,2),w:1.1}
];
assert.equal(computeJoin(i=>varied[i]||null,0,0,true),null);
assert.equal(buildJoinAttributes([0],i=>varied[i]||null,0).fallbacks.length,1);
assert.equal(buildJoinAttributes([0],reader,0).fallbacks.length,0);

// Sparse support probing splits a single G1 into different-width pieces.
// Both ends MUST use one common cross section, even in closed-bead mode.
const split=[
    {...bead([0,0,.35],[1,0,.35],0,0,.05),line:123,w:.6,h:.2},
    {...bead([1,0,.35],[2,0,.35],0,.05,.1),line:123,w:.4,h:.3}
];
const splitReader=i=>split[i]||null;
const sharedA=sharedSplitSection(splitReader,0,true);
const sharedB=sharedSplitSection(splitReader,1,false);
assert(sharedA && sharedB);
assert.deepEqual(sharedA,sharedB);
assert(sharedA[0]>sharedA[1]);
const splitAttribute=buildJoinAttributes([0,1],splitReader,0);
assert(Math.abs(splitAttribute.sizesEnd[0]-splitAttribute.sizesStart[2])<1e-6);
assert(Math.abs(splitAttribute.sizesEnd[1]-splitAttribute.sizesStart[3])<1e-6);
assert.equal(splitAttribute.fallbacks.length,0);
const area=(p)=>profileArea(p.w,p.h);
assert(Math.abs(area({w:sharedA[0],h:sharedA[1]})-(area(split[0])+area(split[1]))*.5)<1e-6);
assert(viewer.includes("const first=model.layers[visualLayer],boundary=loftBoundary("));
// Do not taper unrelated G-code moves merely because they touch in XY.
assert.equal(sharedSplitSection(i=>[
    {...split[0],line:123},{...split[1],line:124}
][i]||null,0,true),null);

// Color transitions must not break a physically connected GPU loft.
const gradientRecords=[
    {...bead([0,0,1.04],[10,0,1.04],0,0,1),line:12998},
    {...bead([10,0,1.04],[20,0,1.281],4,1,2),line:12999},
    {...bead([20,0,1.281],[30,0,1.04],4,2,3),line:13000},
    {...bead([30,0,1.04],[40,0,1.04],0,3,4),line:13001}
];
const gradientRead=i=>gradientRecords[i]||null;
assert(joinContinuity(gradientRead,0,true));
assert(joinContinuity(gradientRead,2,true));
assert.equal(planLoft([0,1,2,3],gradientRead,0,4,true).caps.length,2);
assert(viewer.includes("float zAtRing=mix(iStart.z,iStop.z,aEnd)"));
assert(viewer.includes("vBeadColor=mix(uColor,mix(uRegularColor,uColor,raised),uNonPlanarBlend)"));

// A physical continuation with a different feature must not be welded.
const mixed=[
    bead([0,0,.2],[10,0,.2]),
    {...bead([10,0,.2],[20,0,.2],0,1,2),role:3}
];
assert.equal(joinContinuity(i=>mixed[i]||null,0,true),null);
assert.equal(computeJoin(i=>mixed[i]||null,0,0,true),null);
assert.equal(buildJoinAttributes([0],i=>mixed[i]||null,0).fallbacks.length,0);

// A width spike on the neighboring path must not inflate the corner
// transition. Both full beads remain independently closed.
const extreme=[
    bead([0,0,.2],[10,0,.2]),
    {...bead([10,0,.2],[12,2,.2],0,1,2),w:8,h:.035}
];
const bounded=buildJoinAttributes([0],i=>extreme[i]||null,0);
assert.equal(bounded.fallbacks.length,1);
assert(bounded.fallbacks[0].w<=.46+1e-6);
assert(bounded.fallbacks[0].h<=.035+1e-6);
assert(junctionVertexShader.includes("min(min(iSize.x,iSize.y*1.5),0.8)"));
assert(junctionVertexShader.includes("iTangent.x,iTangent.z,iTangent.y"));
assert(junctionVertexShader.includes("tangent*(span*longitudinal)"));
assert(!junctionVertexShader.includes("iCenter.x+radius*aCircle.x"));
assert(!junctionVertexShader.includes("(iSize.x-h)*0.5"));

assert(junctionVertexShader.includes("clipping_planes_vertex"));
class StubGeometry {
    constructor(){this.attrs={};}
    setAttribute(name,value){this.attrs[name]=value;}
    setIndex(indices){this.indices=indices;}
}
class StubAttribute {
    constructor(values,size){this.values=values;this.size=size;}
}
const THREEStub={
    InstancedBufferGeometry:StubGeometry,
    Float32BufferAttribute:StubAttribute,
    InstancedBufferAttribute:StubAttribute
};
const geom=makeJunctionGeometry(THREEStub,sharpJoin.fallbacks);
assert.equal(geom.instanceCount,1);
assert(geom.indices.length>100);
assert.equal(geom.attrs.iSize.values.length,2);
assert.equal(geom.attrs.iTangent.values.length,3);
assert.equal(geom.attrs.iSpan.values.length,1);
// Reversal falls back to the incoming XY travel direction, not vertical.
assert(Math.abs(geom.attrs.iTangent.values[0]-1)<1e-6);
assert(Math.abs(geom.attrs.iTangent.values[1])<1e-6);
assert(Math.abs(geom.attrs.iTangent.values[2])<1e-6);
assert(geom.attrs.iSpan.values[0]>0);
assert(geom.attrs.iSpan.values[0]<=sharpJoin.fallbacks[0].h*.5+1e-6);
const giantJunction=makeJunctionGeometry(THREEStub,bounded.fallbacks);
assert(giantJunction.attrs.iSize.values[0]<=.46+1e-6);
assert(giantJunction.attrs.iSize.values[1]<=.035+1e-6);

// The G-code inspector must point to the actual source line even when
// the preview omits a full extrusion, rather than guessing by segment index.
const sourceBuffer=new ArrayBuffer(4*48),sourceView=new DataView(sourceBuffer);
for(const [index,line] of [15,20,20,42].entries())
    sourceView.setUint32(index*48+36,line,true);
assert.equal(nearestSourceRecord(sourceView,48,36,4,20),1);
assert.equal(nearestSourceRecord(sourceView,48,36,4,40),3);
assert.equal(nearestSourceRecord(sourceView,48,36,4,17),0);
assert.equal(nearestSourceRecord(sourceView,48,36,4,1),0);
assert.equal(nearestSourceRecord(sourceView,48,36,0,20),-1);
assert(rawClass(1).includes("ausgelassen"));
assert(rawClass(2).includes("G2/G3"));
const rawLine=new DataView(new ArrayBuffer(40));
rawLine.setUint32(28,369,true);
assert.equal(nearestSourceRecord(rawLine,40,28,1,369),0);
const page=readFileSync(join(root,"preview","index.html"),"utf-8");
assert.equal((page.match(/<details class="side-group"/g)||[]).length,8);
assert(page.includes('id="sourceDrawer"'));
assert(viewer.includes("openSourceLine(pick.line)"));
assert(viewer.includes("function auditGPUInstance(index, expected)"));
assert(viewer.includes("syncSourceHighlights()"));
assert(viewer.includes("sourceLineAtLayerEnd(moveData,model.moveCount"));
assert(viewer.includes('row.classList.toggle("active",line===selectedSourceLine)'));
assert(viewer.includes('row.classList.toggle("playing",line===next)'));
assert(viewer.includes("browsingLayer=true"));
assert(viewer.includes("browsingLayer=false"));
assert(viewer.includes("model.layers[visualLayer].start+model.layers[visualLayer].count"));
assert(viewer.includes("model.layers[visualLayer].hopStart+model.layers[visualLayer].hopCount"));
const belowMarkup=readFileSync(join(root,"preview","index.html"),"utf-8");
assert(belowMarkup.includes('<option value="1" selected>Aktuelle + darunter</option>'));
assert(!belowMarkup.includes('<option value="3" selected>'));
assert(belowMarkup.includes('id="selectedSource"'));
assert(belowMarkup.includes('id="playingSource"'));
assert(belowMarkup.includes(".source-line.playing"));
assert(viewer.includes("local<mesh.geometry.instanceCount"));
assert(viewer.includes("pick.segmentIndex=i"));
assert(viewer.includes("GPU-Instanz gefunden"));
// Regression: a float32 record can end *after* float64 metadata duration,
// so the final G1 must not be hidden when playback reaches 100%.
const float64Duration=10000.0006;
const float32End=Math.fround(float64Duration);
assert(float32End>float64Duration);
assert(viewer.includes("currentTime>=model.duration-0.0001?model.count"));
assert(viewer.includes("currentTime>=model.duration-0.0001?model.rawCount"));
assert(viewer.includes("loadRawMoves()"));

function packed(records){
    const buffer=new ArrayBuffer(records.length*48),view=new DataView(buffer);
    records.forEach((record,i)=>{
        const o=i*48;
        [...record.a,...record.b,record.w,record.h].forEach((value,j)=>
            view.setFloat32(o+j*4,value,true));
        view.setUint32(o+32,record.role,true);
        view.setUint32(o+36,i+1,true);
        view.setFloat32(o+40,record.start,true);
        view.setFloat32(o+44,record.end,true);
    });
    return view;
}
const ordinary=packed([bead([0,0,.2],[10,0,.2])]);
const wide=packed([{...bead([0,0,.2],[10,0,.2]),w:2.2}]);
const layer={start:0,count:1,height:.2,timeStart:0,timeEnd:1,z:.2};
const result=analysePrint(wide,new DataView(new ArrayBuffer(0)),[layer]);
assert.equal(result.counts["Überbreite"],1);
assert(result.findings[0].position.length===3);
assert.equal(analysePrint(ordinary,new DataView(new ArrayBuffer(0)),[layer]).total,0);
const compare=compareLayerVolumes(wide,layer,ordinary,layer);
assert(compare.deltaVolume>0 && compare.diffs.length>0);
assert(Math.abs(compare.originalVolume-profileArea(.46,.2)*10)<1e-5);
assert.equal(compareLayerVolumes(ordinary,layer,ordinary,layer).diffs.length,0);
assert(buildCutContours(ordinary,[layer],"x",5,1).vertices.length>0);
assert(buildCutContours(ordinary,[layer],"z",.1,1).vertices.length>0);
assert.equal(buildCutContours(ordinary,[layer],"x",30,1).contours,0);
const limited=buildCutContours(ordinary,[layer],"z",.1,1,{maxSegments:1});
assert(limited.vertices.length>0 && limited.contours===1);
console.log("PASS: seamless joins, print diagnostics, spatial comparison and 3D cut contours");
