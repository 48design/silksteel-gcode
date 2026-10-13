// Bounded, heuristic print diagnostics. No machine-specific collision claims.
const STRIDE=48, HOP_STRIDE=40;
const vec=(dv,o)=>[dv.getFloat32(o,true),dv.getFloat32(o+4,true),dv.getFloat32(o+8,true)];
const distance=(a,b)=>Math.hypot(...a.map((v,i)=>v-b[i]));
export function analysePrint(data,hops,layers,{maxPerType=40}={}) {
    const findings=[],counts={};
    const caps=new Map();
    let layerIndex=0,previous=null;
    function report(type,text,index,position,time,priority=1){
        counts[type]=(counts[type]||0)+1;
        const used=caps.get(type)||0;
        if(used>=maxPerType)return;
        // Omit near-duplicates from the same extrusion source line.
        if(findings.length && findings[findings.length-1].type===type &&
           findings[findings.length-1].index>=index-2)return;
        caps.set(type,used+1);
        findings.push({type,text,index,time,position,layer:layerIndex,priority});
    }
    const n=data.byteLength/STRIDE;
    for(let i=0;i<n;i++){
        const o=i*STRIDE;
        while(layerIndex+1<layers.length && i>=layers[layerIndex+1].start)layerIndex++;
        const a=vec(data,o),b=vec(data,o+12);
        const w=data.getFloat32(o+24,true),h=data.getFloat32(o+28,true);
        const role=data.getUint32(o+32,true),line=data.getUint32(o+36,true);
        const time=data.getFloat32(o+44,true);
        const nominal=layers[layerIndex]?.height||.2;
        const length=distance(a,b);
        const center=a.map((v,k)=>(v+b[k])*.5);
        if(w>Math.max(1.8, nominal*7.5)){
            report("Überbreite",`Zeile ${line}: Strangbreite ${w.toFixed(2)} mm`,i,center,time,2);
        }
        if(Math.abs(b[2]-a[2])>Math.max(.35, nominal*1.8) &&
           length>Math.max(.1, nominal)){
            report("Z-Sprung",`Zeile ${line}: ΔZ ${(b[2]-a[2]).toFixed(2)} mm bei Extrusion`,i,center,time,2);
        }
        if(h>w*1.2 && role!==6 &&
           center[2]>nominal+.12 && length>1){
            report("Freitragend",`Zeile ${line}: vertikal gedehnter Strang ohne klare Auflage`,i,center,time);
        }
        if(previous && previous.role===role && distance(previous.b,a)<.02 &&
           Math.min(w,previous.w)>0 && Math.max(w,previous.w)/Math.min(w,previous.w)>2.3){
            report("Flow-Sprung",`Zeile ${line}: abrupter Querschnittswechsel`,i,center,time,2);
        }
        previous={b,w,role};
    }
    let lift=null,travel=false;
    for(let i=0;i<hops.byteLength/HOP_STRIDE;i++){
        const o=i*HOP_STRIDE,kind=hops.getUint32(o+24,true);
        const a=vec(hops,o),b=vec(hops,o+12);
        if(kind===0){lift={a,b,line:hops.getUint32(o+28,true),time:hops.getFloat32(o+36,true)};travel=false;}
        else if(kind===1 && lift && Math.hypot(b[0]-a[0],b[1]-a[1])>.02)travel=true;
        else if(kind===2 && lift){
            if(!travel && Math.hypot(b[0]-lift.a[0],b[1]-lift.a[1])<.04){
                const before=layerIndex;
                layerIndex=layers.findIndex((layer,i)=>i===layers.length-1||
                    layer.timeEnd>=lift.time);
                report("In-Place-Z-Hop",`Zeile ${lift.line}: Heben/Senken ohne XY-Verfahrt`,0,lift.a,lift.time,2);
                layerIndex=before;
            }
            lift=null;
        }
    }
    findings.sort((a,b)=>b.priority-a.priority||a.time-b.time);
    return {findings,counts,total:Object.values(counts).reduce((a,b)=>a+b,0)};
}
