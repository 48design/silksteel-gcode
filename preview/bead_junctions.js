// Instanced, rounded-end volume patches for non-miterable extrusion joins.
// Profile has flat upper/lower faces and round vertical shoulders.
export const junctionVertexShader=`
precision highp float;
attribute vec2 aCircle, aProfile;
attribute vec3 iCenter, iTangent;
attribute vec2 iSize;
attribute float iSpan;
varying vec3 vNormal;
varying vec3 vPosition;
#include <clipping_planes_pars_vertex>
void main(){
  // Orient the rounded junction along the local G-code path, NOT Z.
  // Width is lateral, height is vertical, and span is along travel.
  vec3 tangent=normalize(vec3(iTangent.x,iTangent.z,iTangent.y));
  float horizontal=length(tangent.xz);
  vec3 lateral=horizontal>0.0001 ?
      vec3(-tangent.z/horizontal,0.0,tangent.x/horizontal) : vec3(1.0,0.0,0.0);
  vec3 vertical=normalize(cross(lateral,tangent));
  // Closed adjacent beads own the material volume. This filler must stay
  // strictly local even when one of those beads is abnormally over-wide.
  float radiusW=max(0.0001,min(min(iSize.x,iSize.y*1.5),0.8)*0.5);
  float radiusH=max(0.0001,min(iSize.y,0.8)*0.5);
  float span=max(0.0001,min(iSpan,0.4));
  float longitudinal=aProfile.y;
  float circle=aProfile.x;
  vec3 p=vec3(iCenter.x,iCenter.z,iCenter.y)
      +tangent*(span*longitudinal)
      +circle*(lateral*(radiusW*aCircle.x)+vertical*(radiusH*aCircle.y));
  vNormal=normalize(tangent*(longitudinal/span)
      +circle*(lateral*(aCircle.x/radiusW)+vertical*(aCircle.y/radiusH)));
  vPosition=p;
  vec4 mvPosition=modelViewMatrix*vec4(p,1.0);
  gl_Position=projectionMatrix*mvPosition;
  #include <clipping_planes_vertex>
}
`;
export function makeJunctionGeometry(THREE,fallbacks){
    if(!fallbacks.length)return null;
    const azimuth=12, profileSteps=8,unit=[],profile=[],indices=[];
    function vertex(u,v,c,s){unit.push(u,v);profile.push(c,s);return unit.length/2-1;}
    const rings=[];
    for(let i=0;i<=profileSteps;i++){
        const angle=-Math.PI/2+i*Math.PI/profileSteps, ring=[];
        for(let j=0;j<azimuth;j++){
            const theta=j*2*Math.PI/azimuth;
            ring.push(vertex(Math.cos(theta),Math.sin(theta),Math.cos(angle),Math.sin(angle)));
        }
        rings.push(ring);
    }
    for(let i=0;i<profileSteps;i++)for(let j=0;j<azimuth;j++){
        const k=(j+1)%azimuth,a=rings[i][j],b=rings[i][k],
              c=rings[i+1][j],d=rings[i+1][k];
        indices.push(a,c,b,b,c,d);
    }
    for(const [ring,side] of [[rings[0],-1],[rings.at(-1),1]]){
        const center=vertex(0,0,0,side);
        for(let j=0;j<azimuth;j++){
            const k=(j+1)%azimuth;
            if(side<0)indices.push(center,ring[j],ring[k]);
            else indices.push(center,ring[k],ring[j]);
        }
    }
    const geometry=new THREE.InstancedBufferGeometry();
    geometry.setAttribute("aCircle",new THREE.Float32BufferAttribute(unit,2));
    geometry.setAttribute("aProfile",new THREE.Float32BufferAttribute(profile,2));
    geometry.setIndex(indices);
    const centers=new Float32Array(fallbacks.length*3);
    const sizes=new Float32Array(fallbacks.length*2);
    const tangents=new Float32Array(fallbacks.length*3);
    const spans=new Float32Array(fallbacks.length);
    for(let i=0;i<fallbacks.length;i++){
        centers.set(fallbacks[i].pos,i*3);
        sizes.set([fallbacks[i].w,fallbacks[i].h],i*2);
        tangents.set(fallbacks[i].tangent,i*3);
        spans[i]=fallbacks[i].span;
    }
    geometry.setAttribute("iCenter",new THREE.InstancedBufferAttribute(centers,3));
    geometry.setAttribute("iSize",new THREE.InstancedBufferAttribute(sizes,2));
    geometry.setAttribute("iTangent",new THREE.InstancedBufferAttribute(tangents,3));
    geometry.setAttribute("iSpan",new THREE.InstancedBufferAttribute(spans,1));
    geometry.instanceCount=fallbacks.length;
    return geometry;
}
