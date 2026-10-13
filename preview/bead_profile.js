// Canonical geometric rounded-rectangle profile for preview inspection.
// G-code Z is the fixed TOP; cross-section area equals E volume / path length.
export const CORNER_FRACTION=0.4;
export const CORNER_LOSS=(4-Math.PI)*CORNER_FRACTION**2;
export function profileArea(width,height){
    if(!(width>0&&height>0))return 0;
    return width*height-CORNER_LOSS*Math.min(width,height)**2;
}
export function profileWidth(area,height){
    if(!(area>0&&height>0))return 0;
    const flat=area/height+CORNER_LOSS*height;
    if(flat>=height)return flat;
    const disc=Math.max(0,height*height-4*CORNER_LOSS*area);
    return 2*area/(height+Math.sqrt(disc));
}
export function profileOutline(width,height,steps=5){
    const r=CORNER_FRACTION*Math.min(width,height),points=[];
    for(const [sx,sy,start] of [[1,1,0],[-1,1,Math.PI/2],
                                [-1,-1,Math.PI],[1,-1,3*Math.PI/2]]){
        for(let i=0;i<=steps;i++){
            const angle=start+i*Math.PI/(2*steps);
            points.push([
                sx*(width/2-r)+r*Math.cos(angle),
                sy*(height/2-r)+r*Math.sin(angle)-height/2
            ]);
        }
    }
    return points;
}
