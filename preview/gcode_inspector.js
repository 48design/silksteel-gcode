// Source-aware lookup for rendered and omitted G-code extrusion records.
export function nearestSourceRecord(view,stride,lineOffset,count,line) {
    if(!view || count<1 || !Number.isFinite(line))return -1;
    let lo=0,hi=count;
    while(lo<hi){
        const mid=(lo+hi)>>>1;
        if(view.getUint32(mid*stride+lineOffset,true)<line)lo=mid+1;
        else hi=mid;
    }
    if(lo>=count)return count-1;
    if(lo===0)return 0;
    const a=view.getUint32((lo-1)*stride+lineOffset,true);
    const b=view.getUint32(lo*stride+lineOffset,true);
    return Math.abs(line-a)<Math.abs(b-line)?lo-1:lo;
}
export function rawClass(kind) {
    return kind===1?"Im G-Code extrudiert, aber nicht als 3D-Strang dargestellt":
        kind===2?"G2/G3-Bogen: nur Start/Ende verfügbar":
        kind===3?"Reine Z-Extrusion":
        "Gerenderte Extrusion";
}
