import {profileArea} from "./bead_profile.js";
// Coarse, explicitly approximate volumetric comparison for one visible layer.
// Bounding memory to one layer keeps large G-code files usable.
const STRIDE=48;
export function closestLayer(layers,z) {
    let best=0,delta=Infinity;
    layers.forEach((layer,i)=>{
        const d=Math.abs(layer.z-z);
        if(d<delta){best=i;delta=d;}
    });
    return best;
}
function sampleLayer(data,layer,{xy=.65,z=.20,maxSamples=60000}={}) {
    const cells=new Map();
    let volume=0,samples=0;
    const start=layer.start,stop=start+layer.count;
    const sampleEvery=Math.max(1,Math.ceil(layer.count/maxSamples));
    for(let i=start;i<stop;i++){
        const o=i*STRIDE;
        const a=[0,4,8].map(d=>data.getFloat32(o+d,true));
        const b=[12,16,20].map(d=>data.getFloat32(o+d,true));
        const width=data.getFloat32(o+24,true),height=data.getFloat32(o+28,true);
        const length=Math.hypot(b[0]-a[0],b[1]-a[1],b[2]-a[2]);
        if(length<1e-7)continue;
        const area=profileArea(width,height);
        const vol=area*length;
        volume+=vol;
        if((i-start)%sampleEvery!==0)continue;
        const n=Math.max(1,Math.min(64,Math.ceil(length/xy)));
        // Sample budget prevents adversarial G-code from exhausting browser RAM.
        const stride=Math.max(1,Math.ceil(layer.count*n/maxSamples));
        for(let t=0;t<n;t+=stride){
            const mid=(t+Math.min(n,t+stride))/2/n;
            const p=a.map((v,d)=>v+(b[d]-v)*mid);
            const key=[Math.floor(p[0]/xy),Math.floor(p[1]/xy),Math.floor(p[2]/z)].join(",");
            let cell=cells.get(key);
            if(!cell){cell={vol:0,pos:p};cells.set(key,cell);}
            cell.vol+=vol*sampleEvery*(Math.min(n,t+stride)-t)/n;
            samples++;
        }
    }
    return {cells,volume,samples};
}
export function compareLayerVolumes(processed,processedLayer,original,originalLayer,settings={}) {
    const current=sampleLayer(processed,processedLayer,settings);
    const baseline=sampleLayer(original,originalLayer,settings);
    const diffs=[],maxPoints=2200;
    let added=0,removed=0,changed=0;
    const keys=new Set([...current.cells.keys(),...baseline.cells.keys()]);
    const ranked=[];
    for(const key of keys){
        const a=current.cells.get(key),b=baseline.cells.get(key);
        const delta=(a?.vol||0)-(b?.vol||0);
        const threshold=Math.max(.007,(Math.abs(a?.vol||0)+Math.abs(b?.vol||0))*.15);
        if(Math.abs(delta)<=threshold)continue;
        if(!b)added+=delta;
        else if(!a)removed-=delta;
        else changed+=Math.abs(delta);
        ranked.push({pos:(delta>0?a:b).pos,delta});
    }
    ranked.sort((a,b)=>Math.abs(b.delta)-Math.abs(a.delta));
    diffs.push(...ranked.slice(0,maxPoints));
    return {processedVolume:current.volume,originalVolume:baseline.volume,
        deltaVolume:current.volume-baseline.volume,
        added,removed,changed,cells:keys.size,diffs,approximate:true};
}
