import {joinContinuity} from "./extrusion_joins.js";
import {profileArea,profileWidth} from "./bead_profile.js";

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
        point:atEnd?p.b:p.a,connected:false,shared:false};
    const neighbor=atEnd?id+1:id-1;
    if(!connected||neighbor<first||neighbor>=last)return local;
    const pair=joinContinuity(read,id,atEnd);
    if(!pair)return local;
    // Snap rounded G-code endpoints to exactly the same cross-section center.
    const point=pair.a.b.map((v,i)=>(v+pair.b.a[i])*.5);
    const n1=normalOf(pair.a),n2=normalOf(pair.b);
    const sx=n1[0]+n2[0],sy=n1[1]+n2[1],len=Math.hypot(sx,sy);
    const h=Math.sqrt(Math.max(1e-9,pair.a.h*pair.b.h));
    const area=(profileArea(pair.a.w,pair.a.h)+
                profileArea(pair.b.w,pair.b.h))*.5;
    const size=[profileWidth(area,h),h];
    // A diagonal cut through the bead stretches the ring by 1/cos(turn/2).
    // Its longitudinal extent must fit the neighboring path segments.
    const cosine=Math.min(1,len*.5);
    const reach=cosine>1e-6
        ?size[0]*.5*Math.sqrt(1/(cosine*cosine)-1):Infinity;
    if(!(len>1e-6)||reach>.45*Math.min(pair.l1,pair.l2)){
        // The full miter would spike or fold. Keep the two original profile
        // rings and join them with a short rounded sweep, not separate caps.
        return {...local,point,connected:true};
    }
    const scale=1/cosine;
    return {join:[sx/len*scale,sy/len*scale,0],
        size,point,connected:true,shared:true};
}
export function loftBoundary(read,id,atEnd,first,last,seamless=true){
    return boundary(read,id,atEnd,first,last,seamless);
}
function packBevels(entries,bounds){
    if(!entries.length)return null;
    const n=entries.length;
    const starts=new Float32Array(n*3),ends=new Float32Array(n*3),
        sizes=new Float32Array(n*2),joinsStart=new Float32Array(n*3),
        joinsEnd=new Float32Array(n*3),sizesStart=new Float32Array(n*2),
        sizesEnd=new Float32Array(n*2);
    for(let i=0;i<n;i++){
        const e=entries[i];
        starts.set(e.point,i*3);ends.set(e.point,i*3);
        sizes.set(e.sizeStart,i*2);
        joinsStart.set(e.start,i*3);joinsEnd.set(e.end,i*3);
        sizesStart.set(e.sizeStart,i*2);sizesEnd.set(e.sizeEnd,i*2);
    }
    return {starts,ends,sizes,joinsStart,joinsEnd,sizesStart,sizesEnd,
        indices:Uint32Array.from(entries.map(e=>e.index)),bounds};
}
export function planLoft(indices,read,first,last,seamless=true){
    const n=indices.length;
    const starts=new Float32Array(n*3),ends=new Float32Array(n*3),
          sizes=new Float32Array(n*2),joinsStart=new Float32Array(n*3),
          joinsEnd=new Float32Array(n*3),sizesStart=new Float32Array(n*2),
          sizesEnd=new Float32Array(n*2);
    const caps=[],bevelEntries=[];
    let lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
    let padding=0,previousId=-2,previousRight=null;
    for(let i=0;i<n;i++){
        const id=indices[i],p=read(id);
        // A connected junction has identical data on both sides. Reuse the
        // previous outgoing ring when successive IDs share a source junction.
        const left=id===previousId+1 && previousRight?.shared
            ? previousRight : boundary(read,id,false,first,last,seamless);
        const right=boundary(read,id,true,first,last,seamless);
        if(left.connected&&!left.shared){
            // Sweeping normal profile rings about the same snapped endpoint
            // creates a continuous rounded join, even for exact U-turns.
            const before=boundary(read,id-1,true,first,last,seamless);
            const startAngle=Math.atan2(before.join[1],before.join[0]);
            const endAngle=Math.atan2(left.join[1],left.join[0]);
            let turn=Math.atan2(Math.sin(endAngle-startAngle),
                                Math.cos(endAngle-startAngle));
            if(Math.abs(Math.abs(turn)-Math.PI)<1e-6)turn=Math.PI;
            const count=Math.max(1,Math.ceil(Math.abs(turn)/(Math.PI/6)));
            for(let j=0;j<count;j++){
                const t0=j/count,t1=(j+1)/count;
                const angle0=startAngle+turn*t0,angle1=startAngle+turn*t1;
                const lerpSize=t=>before.size.map((v,k)=>
                    v+(left.size[k]-v)*t);
                bevelEntries.push({
                    index:id,point:left.point,
                    start:j===0?before.join:[Math.cos(angle0),Math.sin(angle0),0],
                    end:j===count-1?left.join:[Math.cos(angle1),Math.sin(angle1),0],
                    sizeStart:lerpSize(t0),sizeEnd:lerpSize(t1)
                });
            }
        }
        previousId=id;
        previousRight=right;
        starts.set(left.point,i*3);ends.set(right.point,i*3);
        sizes[i*2]=p.w;sizes[i*2+1]=p.h;
        joinsStart.set(left.join,i*3);
        joinsEnd.set(right.join,i*3);
        sizesStart.set(left.size,i*2);
        sizesEnd.set(right.size,i*2);
        if(!left.connected)caps.push({index:id,end:0,local:i});
        if(!right.connected)caps.push({index:id,end:1,local:i});
        for(let d=0;d<3;d++){
            lo[d]=Math.min(lo[d],p.a[d],p.b[d]);
            hi[d]=Math.max(hi[d],p.a[d],p.b[d]);
        }
        padding=Math.max(padding,left.size[0]*Math.hypot(left.join[0],left.join[1]),
            right.size[0]*Math.hypot(right.join[0],right.join[1]),
            left.size[1],right.size[1]);
    }
    // Include the entire rounded sweep in the conservative bounds.
    const bounds=n?{lo,hi,padding:padding+.08}:null;
    return {starts,ends,sizes,joinsStart,joinsEnd,sizesStart,sizesEnd,caps,
        bevels:packBevels(bevelEntries,bounds),bounds};
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

// Merge nearby 40-mm logical cells into spatial GPU batches. Within each
// batch, source IDs remain ordered so playback prefix counts stay correct.
export function spatialBatches(first,last,read,cellSize=40,span=3,maxItems=6000){
    const groups=new Map();
    for(let id=first;id<last;id++){
        const p=read(id);
        const x=(p.a[0]+p.b[0])*.5,y=(p.a[1]+p.b[1])*.5;
        const cx=Math.floor(x/cellSize),cy=Math.floor(y/cellSize);
        const key=p.role+":"+Math.floor(cx/span)+":"+Math.floor(cy/span);
        let bins=groups.get(key);
        if(!bins){bins=[[]];groups.set(key,bins);}
        let bucket=bins[bins.length-1];
        if(bucket.length>=maxItems){bucket=[];bins.push(bucket);}
        bucket.push(id);
    }
    return [...groups.entries()].flatMap(([key,bins])=>bins.map(indices=>({
        key,role:read(indices[0]).role,indices
    })));
}
