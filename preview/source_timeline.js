// Compact 12-byte timeline of *all* timed source G-code moves.
// Unlike extrusion instances this includes travel and G4 dwell.
export const MOVE_SIZE=12;
export function sourceLineAtTime(view,count,seconds){
    if(!view || count<=0)return 0;
    let low=0,high=count;
    while(low<high){
        const mid=(low+high)>>>1;
        if(view.getFloat32(mid*MOVE_SIZE+4,true)<=seconds)low=mid+1;
        else high=mid;
    }
    return view.getUint32(Math.min(count-1,low)*MOVE_SIZE+8,true);
}
export function sourceLineAtLayerEnd(view,count,layer){
    if(!view||!layer)return 0;
    const start=layer.moveStart||0;
    const last=Math.min(count-1,start+(layer.moveCount||0)-1);
    if(last<start)return 0;
    return view.getUint32(last*MOVE_SIZE+8,true);
}
