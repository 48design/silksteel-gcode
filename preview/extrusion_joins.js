// Shared-radius bead joins. Every extrusion remains closed at both ends,
// including when its successor has not yet been printed by the playback.
export function joinContinuity(read,index,atEnd) {
    const a=read(atEnd?index:index-1),b=read(atEnd?index+1:index);
    if(!a || !b)return null;
    if(!a.b || !b.a || a.role!==b.role)return null;
    if(Math.hypot(a.b[0]-b.a[0],a.b[1]-b.a[1],a.b[2]-b.a[2])>.012)return null;
    // Tolerate Float32 timestamp quantization on long prints.
    if(b.start-a.end>.08 || b.start<a.end-.12)return null;
    const v1=[a.b[0]-a.a[0],a.b[1]-a.a[1]];
    const v2=[b.b[0]-b.a[0],b.b[1]-b.a[1]];
    const l1=Math.hypot(...v1),l2=Math.hypot(...v2);
    if(l1<.0001 || l2<.0001)return null;
    const dot=(v1[0]*v2[0]+v1[1]*v2[1])/(l1*l2);
    return {a,b,dot,l1,l2};
}

export function sharedSplitSection(read, index, atEnd) {
    // Subsections originating from the SAME G1 must share the same physical
    // boundary, even when width changes because the support grid changes.
    const match=joinContinuity(read,index,atEnd);
    if(!match || !Number.isInteger(match.a.line) ||
       match.a.line!==match.b.line || match.dot<.999)return null;
    const a=match.a,b=match.b;
    const h=(a.h+b.h)*.5;
    if(!(h>0))return null;
    const k=1-Math.PI/4;
    const area=((a.w*a.h-k*a.h*a.h)+(b.w*b.h-k*b.h*b.h))*.5;
    return [Math.max(h,area/h+k*h),h];
}

export function computeJoin(read,index,role,atEnd) {
    const continuity=joinContinuity(read,index,atEnd);
    if(!continuity)return null;
    const {a,b,dot,l1,l2}=continuity;
    // Never miter sharp reversals or sections whose bead cross-section changes
    // significantly. Such junctions receive a separate rounded volume filler.
    if(dot<-.10)return null;
    if(Math.abs(a.w-b.w)>Math.max(.045,(a.w+b.w)*.14) ||
       Math.abs(a.h-b.h)>Math.max(.025,(a.h+b.h)*.14))return null;
    const n1=[-(a.b[1]-a.a[1])/l1,(a.b[0]-a.a[0])/l1];
    const n2=[-(b.b[1]-b.a[1])/l2,(b.b[0]-b.a[0])/l2];
    const nx=n1[0]+n2[0],ny=n1[1]+n2[1],norm=Math.hypot(nx,ny);
    if(norm<.001)return null;
    const cosine=(nx*n2[0]+ny*n2[1])/norm;
    if(cosine<.67)return null;
    const scale=1/cosine;
    const size=[(a.w+b.w)*.5,(a.h+b.h)*.5];
    // Prevent miter spikes from folding a mesh over a very short segment.
    const extent=size[0]*.5*Math.sqrt(Math.max(0,scale*scale-1));
    if(extent>Math.min(l1,l2)*.45)return null;
    return {miter:[nx/norm*scale,ny/norm*scale],size};
}

// Returns both stable GPU data and missing-corner locations, without
// duplicating fallback vertices on either side of a joined segment.
export function buildJoinAttributes(indices,read,role) {
    const n=indices.length;
    const joinsStart=new Float32Array(n*3),joinsEnd=new Float32Array(n*3);
    const sizesStart=new Float32Array(n*2),sizesEnd=new Float32Array(n*2);
    const fallbacks=[];
    for(let i=0;i<n;i++){
        const id=indices[i],p=read(id);
        const dx=p.b[0]-p.a[0],dy=p.b[1]-p.a[1],l=Math.hypot(dx,dy)||1;
        const normal=[-dy/l,dx/l];
        const start=computeJoin(read,id,role,false);
        const end=computeJoin(read,id,role,true);
        joinsStart.set([...(start?.miter||normal),start?1:0],i*3);
        joinsEnd.set([...(end?.miter||normal),end?1:0],i*3);
        sizesStart.set(sharedSplitSection(read,id,false)||start?.size||[p.w,p.h],i*2);
        sizesEnd.set(sharedSplitSection(read,id,true)||end?.size||[p.w,p.h],i*2);
        if(!end){
            const match=joinContinuity(read,id,true);
            if(match && !sharedSplitSection(read,id,true) &&
               (match.dot<.995 || Math.abs(match.a.w-match.b.w)>.015 ||
                Math.abs(match.a.h-match.b.h)>.015)){
                // Junction fillers must NEVER inherit the largest width from
                // an adjacent over-extruded track: that created giant flat
                // discs at bends. The still-closed tracks already preserve
                // the full volume and footprint at each end.
                const h=Math.max(.02,Math.min(match.a.h,match.b.h));
                const w=Math.max(h,Math.min(match.a.w,match.b.w));
                // Orient corner fillers along the actual G-code path.
                // A vertically oriented disc is not a deposited filament.
                const a=match.a, b=match.b;
                const u=[a.b[0]-a.a[0],a.b[1]-a.a[1],a.b[2]-a.a[2]];
                const v=[b.b[0]-b.a[0],b.b[1]-b.a[1],b.b[2]-b.a[2]];
                const ul=Math.hypot(...u),vl=Math.hypot(...v);
                // At a near-U-turn, averaging opposite directions makes
                // the tangent point *sideways*. Retain the incoming direction.
                const tangent=match.dot < -0.5
                    ? u.map(x=>x/(ul||1))
                    : u.map((x,k)=>x/(ul||1)+v[k]/(vl||1));
                const tl=Math.hypot(...tangent);
                const unit=(tl>1e-6?tangent.map(x=>x/tl):u.map(x=>x/(ul||1)));
                const span=Math.max(0.001,Math.min(.5*h,match.l1*.25,match.l2*.25));
                fallbacks.push({index:id,pos:[...p.b],w,h,role,tangent:unit,span});
            }
        }
    }
    return {joinsStart,joinsEnd,sizesStart,sizesEnd,fallbacks};
}
