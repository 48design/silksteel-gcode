// Approximate cross-section contours of deposited stadium beads.
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
            const center=(p.a[2]+p.b[2])*.5-p.h*.5;
            const dz=position-center;
            if(Math.abs(dz)>p.h*.5)continue;
            const half=Math.max(0,(p.w-p.h)*.5+Math.sqrt(Math.max(0,(p.h*.5)**2-dz**2)));
            const px=-dy/len*half,py=dx/len*half;
            const verts=[
                [p.a[0]+px,position,p.a[1]+py],
                [p.b[0]+px,position,p.b[1]+py],
                [p.b[0]-px,position,p.b[1]-py],
                [p.a[0]-px,position,p.a[1]-py]
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
        const w=Math.max(p.h,p.w*dir),h=p.h;
        const perimeter=[];
        const n=16,inner=(w-h)*.5;
        for(let j=0;j<=n;j++){
            const theta=-Math.PI/2+j*Math.PI/n;
            const lateral=inner+Math.cos(theta)*h*.5;
            const vertical=Math.sin(theta)*h*.5-h*.5;
            perimeter.push(axisIndex===0 ?
                [position,z+vertical,y+lateral] : [x+lateral,z+vertical,position]);
        }
        for(let j=0;j<=n;j++){
            const theta=Math.PI/2+j*Math.PI/n;
            const lateral=-inner+Math.cos(theta)*h*.5;
            const vertical=Math.sin(theta)*h*.5-h*.5;
            perimeter.push(axisIndex===0 ?
                [position,z+vertical,y+lateral] : [x+lateral,z+vertical,position]);
        }
        for(let j=0;j<perimeter.length;j++)push(perimeter[j],perimeter[(j+1)%perimeter.length]);
        count++;
    }
    return {vertices:new Float32Array(lines),contours:count,truncated:count>=countMax};
}
