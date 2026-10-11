import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
import {dirname,join} from "node:path";

const root=dirname(dirname(fileURLToPath(import.meta.url)));
async function load(file){
    const code=readFileSync(join(root,"preview",file));
    return import("data:text/javascript;base64,"+code.toString("base64"));
}
const {computeJoin,buildJoinAttributes}=await load("extrusion_joins.js");
const {analysePrint}=await load("diagnostics.js");
const {compareLayerVolumes}=await load("comparison.js");
const {buildCutContours}=await load("section.js");

const bead=(a,b,role=0,start=0,end=1)=>({a,b,w:.46,h:.20,role,start,end});
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
assert(Math.abs(compare.originalVolume-(.46*.2-(1-Math.PI/4)*.2**2)*10)<1e-5);
assert.equal(compareLayerVolumes(ordinary,layer,ordinary,layer).diffs.length,0);
assert(buildCutContours(ordinary,[layer],"x",5,1).vertices.length>0);
assert(buildCutContours(ordinary,[layer],"z",.1,1).vertices.length>0);
assert.equal(buildCutContours(ordinary,[layer],"x",30,1).contours,0);
const limited=buildCutContours(ordinary,[layer],"z",.1,1,{maxSegments:1});
assert(limited.vertices.length>0 && limited.contours===1);
console.log("PASS: seamless joins, print diagnostics, spatial comparison and 3D cut contours");
