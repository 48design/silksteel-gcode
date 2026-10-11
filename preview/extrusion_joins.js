// Seamless visual joins for ordered, GPU-instanced G-code bead segments.
// Independent of Three.js; preserves existing record offsets and playback.
export function computeJoin(read, index, role, atEnd) {
    const current = read(index);
    const neighborIndex = index + (atEnd ? 1 : -1);
    if (neighborIndex < 0) return null;
    const neighbor = read(neighborIndex);
    if (!neighbor || neighbor.role !== role) return null;
    const a = atEnd ? current : neighbor;
    const b = atEnd ? neighbor : current;
    const delta = Math.hypot(a.b[0]-b.a[0], a.b[1]-b.a[1], a.b[2]-b.a[2]);
    if (delta > .012) return null;
    // Only continuously printed paths join. Retracted, lifted or separately
    // deposited tracks must keep their real end caps.
    if (b.start - a.end > .005) return null;
    const dx1=a.b[0]-a.a[0], dy1=a.b[1]-a.a[1];
    const dx2=b.b[0]-b.a[0], dy2=b.b[1]-b.a[1];
    const l1=Math.hypot(dx1,dy1), l2=Math.hypot(dx2,dy2);
    if (l1<.0001 || l2<.0001) return null;
    const n1=[-dy1/l1,dx1/l1], n2=[-dy2/l2,dx2/l2];
    const cosine=n1[0]*n2[0]+n1[1]*n2[1];
    if (cosine < -.50) return null; // Extreme reverse turn: avoid huge miter.
    if (Math.abs(a.w-b.w)>Math.max(.06,(a.w+b.w)*.20) ||
        Math.abs(a.h-b.h)>Math.max(.035,(a.h+b.h)*.20)) return null;
    const mx=n1[0]+n2[0],my=n1[1]+n2[1], ml=Math.hypot(mx,my);
    if (ml<.001) return null;
    const nx=mx/ml,ny=my/ml;
    const scale=1/Math.max(.40,nx*n2[0]+ny*n2[1]);
    if(scale>2.5) return null;
    return {miter:[nx*scale,ny*scale],size:[(a.w+b.w)*.5,(a.h+b.h)*.5]};
}

export function buildJoinAttributes(indices, read, role) {
    const n=indices.length;
    const joinsStart=new Float32Array(n*3), joinsEnd=new Float32Array(n*3);
    const sizesStart=new Float32Array(n*2), sizesEnd=new Float32Array(n*2);
    for(let i=0;i<n;i++){
        const id=indices[i],p=read(id);
        const dx=p.b[0]-p.a[0],dy=p.b[1]-p.a[1],l=Math.hypot(dx,dy)||1;
        const normal=[-dy/l,dx/l];
        const start=computeJoin(read,id,role,false);
        const end=computeJoin(read,id,role,true);
        joinsStart.set([...(start?.miter||normal),start?1:0],i*3);
        joinsEnd.set([...(end?.miter||normal),end?1:0],i*3);
        sizesStart.set(start?.size||[p.w,p.h],i*2);
        sizesEnd.set(end?.size||[p.w,p.h],i*2);
    }
    return {joinsStart,joinsEnd,sizesStart,sizesEnd};
}
