import {CORNER_FRACTION,profileOutline} from "./bead_profile.js";
// Approximate cross-section contours of rounded-rectangle beads.
// The GPU clipping plane reveals hidden geometry; these lines outline its cut.
function bead(record,o) {
    const a=[0,4,8].map(d=>record.getFloat32(o+d,true));
    const b=[12,16,20].map(d=>record.getFloat32(o+d,true));
    return {a,b,w:record.getFloat32(o+24,true),h:record.getFloat32(o+28,true)};
}
const interpolate=(a,b,t)=>a+(b-a)*t;
export function buildCutContours(data,layers,axis,position,finished,{maxSegments=6500,stride=48}={}) {
    const lines=[],countMax=Math.max(1,maxSegments);
    const axisIndex=axis==="x"?0:axis==="y"?1:2;
    const push=(a,b)=>lines.push(...a,...b);
    let count=0;
    const minIndex=Math.max(0,layers[0]?.start||0);
    const endIndex=Math.min(finished,layers.at(-1)?.start+layers.at(-1)?.count||0);
    for(let i=minIndex;i<endIndex && count<countMax;i++){
        const p=bead(data,i*stride);
        const dx=p.b[0]-p.a[0],dy=p.b[1]-p.a[1];
        const len=Math.hypot(dx,dy);
        if(len<.001)continue;
        if(axisIndex===2){
            // The upper surface follows G-code Z, so a horizontal cut must
            // evaluate the rounded-rectangle width at each endpoint.
            const halfWidth=z=>{
                const depth=z-position,r=CORNER_FRACTION*Math.min(p.w,p.h);
                if(depth<0||depth>p.h)return 0;
                const d=depth<r ? r-depth :
                    depth>p.h-r ? depth-(p.h-r) : 0;
                return p.w*.5-r+Math.sqrt(Math.max(0,r*r-d*d));
            };
            const left=halfWidth(p.a[2]),right=halfWidth(p.b[2]);
            if(left<=0&&right<=0)continue;
            const pa=[-dy/len*left,dx/len*left];
            const pb=[-dy/len*right,dx/len*right];
            const verts=[
                [p.a[0]+pa[0],position,p.a[1]+pa[1]],
                [p.b[0]+pb[0],position,p.b[1]+pb[1]],
                [p.b[0]-pb[0],position,p.b[1]-pb[1]],
                [p.a[0]-pa[0],position,p.a[1]-pa[1]]
            ];
            for(let j=0;j<4;j++)push(verts[j],verts[(j+1)%4]);
            count++;continue;
        }
        const da=p.b[axisIndex]-p.a[axisIndex];
        if(Math.abs(da)<.001)continue;
        const t=(position-p.a[axisIndex])/da;
        if(t<0||t>1)continue;
        const x=interpolate(p.a[0],p.b[0],t);
        const y=interpolate(p.a[1],p.b[1],t);
        const z=interpolate(p.a[2],p.b[2],t);
        const dir=Math.min(2.5,len/Math.abs(da));
        const w=p.w*dir,h=p.h;
        const perimeter=profileOutline(w,h,5).map(([lateral,vertical])=>
            axisIndex===0 ? [position,z+vertical,y+lateral] :
                            [x+lateral,z+vertical,position]);
        for(let j=0;j<perimeter.length;j++)push(perimeter[j],perimeter[(j+1)%perimeter.length]);
        count++;
    }
    return {vertices:new Float32Array(lines),contours:count,truncated:count>=countMax};
}
