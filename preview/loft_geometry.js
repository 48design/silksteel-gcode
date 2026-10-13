import {computeJoin,sharedSplitSection,joinContinuity} from "./extrusion_joins.js";

// A GPU loft is a sequence of two-ring mantle instances. Adjacent instances
// must use the EXACT same ring transform, including at G1 subdivisions.
// Only discontinuities get real cap instances; playback adds a temporary tip.
const normalOf=p=>{
    const dx=p.b[0]-p.a[0],dy=p.b[1]-p.a[1];
    const len=Math.hypot(dx,dy);
    return len>1e-8?[-dy/len,dx/len]:[1,0];
};
function boundary(read,id,atEnd,first,last,connected){
    const p=read(id),normal=normalOf(p);
    const local={join:[normal[0],normal[1],0],size:[p.w,p.h],
        point:atEnd?p.b:p.a,connected:false};
    const neighbor=atEnd?id+1:id-1;
    if(!connected||neighbor<first||neighbor>=last)return local;
    const pair=joinContinuity(read,id,atEnd);
    if(!pair)return local;
    // Source endpoints can differ by <0.012 mm due to G-code rounding.
    // Snap both GPU rings to the same midpoint; otherwise an otherwise
    // connected loft can contain tiny cracks despite identical normals.
    const point=pair.a.b.map((v,i)=>(v+pair.b.a[i])*.5);
    // Same-line support subdivision must have a shared cross section even
    // if the source width/height changes too much for a normal miter.
    const split=sharedSplitSection(read,id,atEnd);
    if(split){
        const n1=normalOf(pair.a),n2=normalOf(pair.b);
        const sx=n1[0]+n2[0],sy=n1[1]+n2[1],len=Math.hypot(sx,sy);
        if(len<1e-5)return local;
        return {join:[sx/len,sy/len,0],size:split,point,connected:true};
    }
    const miter=computeJoin(read,id,p.role,atEnd);
    if(!miter)return local;
    return {join:[...miter.miter,0],size:miter.size,point,connected:true};
}
export function loftBoundary(read,id,atEnd,first,last,seamless=true){
    return boundary(read,id,atEnd,first,last,seamless);
}
export function planLoft(indices,read,first,last,seamless=true){
    const n=indices.length;
    const starts=new Float32Array(n*3),ends=new Float32Array(n*3),
          sizes=new Float32Array(n*2),joinsStart=new Float32Array(n*3),
          joinsEnd=new Float32Array(n*3),sizesStart=new Float32Array(n*2),
          sizesEnd=new Float32Array(n*2);
    const caps=[];
    let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    let padding=0;
    for(let i=0;i<n;i++){
        const id=indices[i],p=read(id);
        const left=boundary(read,id,false,first,last,seamless);
        const right=boundary(read,id,true,first,last,seamless);
        starts.set(left.point,i*3);ends.set(right.point,i*3);
        sizes.set([p.w,p.h],i*2);
        joinsStart.set(left.join,i*3);
        joinsEnd.set(right.join,i*3);
        sizesStart.set(left.size,i*2);
        sizesEnd.set(right.size,i*2);
        if(!left.connected)caps.push({index:id,end:0,local:i});
        if(!right.connected)caps.push({index:id,end:1,local:i});
        for(const point of [p.a,p.b]){
            for(let d=0;d<3;d++){
                lo[d]=Math.min(lo[d],point[d]);hi[d]=Math.max(hi[d],point[d]);
            }
        }
        padding=Math.max(padding,left.size[0]*Math.hypot(left.join[0],left.join[1]),
            right.size[0]*Math.hypot(right.join[0],right.join[1]),
            left.size[1],right.size[1]);
    }
    return {starts,ends,sizes,joinsStart,joinsEnd,sizesStart,sizesEnd,caps,
        // The bead extends a full h BELOW nozzle Z, not h/2.
        // Never shrink this padding: optimistic spheres may cull valid strands.
        bounds:n?{lo,hi,padding:padding+.08}:null};
}
// Cells are deliberately 3D-position based, not sequenced by G-code role:
// adjacent segments remain topologically joined even across chunk boundaries.
export function spatialChunks(layerStart,layerEnd,read,cellSize=40,maxItems=6000){
    const cells=new Map();
    for(let id=layerStart;id<layerEnd;id++){
        const p=read(id);
        const x=(p.a[0]+p.b[0])*.5,y=(p.a[1]+p.b[1])*.5;
        const key=p.role+":"+Math.floor(x/cellSize)+":"+Math.floor(y/cellSize);
        let bins=cells.get(key);
        if(!bins){bins=[[]];cells.set(key,bins);}
        let bucket=bins[bins.length-1];
        if(bucket.length>=maxItems){bucket=[];bins.push(bucket);}
        bucket.push(id);
    }
    return [...cells.entries()].flatMap(([key,bins])=>bins.map(indices=>({
        key,role:read(indices[0]).role,indices
    })));
}
