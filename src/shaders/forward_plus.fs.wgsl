// TODO-2: implement the Forward+ fragment shader

// See naive.fs.wgsl for basic fragment shader setup; this shader should use light clusters instead of looping over all lights

// ------------------------------------
// Shading process:
// ------------------------------------
// Determine which cluster contains the current fragment.
// Retrieve the number of lights that affect the current fragment from the cluster’s data.
// Initialize a variable to accumulate the total light contribution for the fragment.
// For each light in the cluster:
//     Access the light's properties using its index.
//     Calculate the contribution of the light based on its position, the fragment’s position, and the surface normal.
//     Add the calculated contribution to the total light accumulation.
// Multiply the fragment’s diffuse color by the accumulated light contribution.
// Return the final color, ensuring that the alpha component is set appropriately (typically to 1).
@group(${bindGroup_scene}) @binding(0) var<uniform> u_Camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_lightCull}) @binding(0) var depthTexture: texture_2d<f32>;
@group(${bindGroup_lightCull}) @binding(2) var<storage, read_write> lightIndices: array<i32>;
@group(${bindGroup_lightCull}) @binding(3) var<storage, read_write> lightGrid: array<i32>;
@group(${bindGroup_lightCull}) @binding(4) var<storage, read_write> tileMinMax: array<i32>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read_write> lightCountTotal: atomic<i32>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read> gridSize: vec2i;
@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput
{
    @builtin(position) fragPos: vec4f,
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) viewPos: vec3f
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    let maxGridSize = ${MAX_GRID_SIZE};
    var viewportSize = vec2f(u_Camera.viewportSize);
    var tileIndex = vec2i(i32(in.fragPos.x / ${TILESIZE_X}), i32(in.fragPos.y / ${TILESIZE_Y}));
    var gridId = tileIndex.x + maxGridSize * tileIndex.y;
    var lightCount = lightGrid[2*gridId + 1];
    var lightIdxOffset = lightGrid[2*gridId];
    var tileMin = f32(tileMinMax[2*gridId]) / f32(${DEPTH_INTEGER_SCALE});
    var tileMax = f32(tileMinMax[2*gridId+1]) / f32(${DEPTH_INTEGER_SCALE});
    var tilePos_Pixel = vec4f(vec4i(i32(tileIndex.x)*${TILESIZE_X},i32(tileIndex.y)*${TILESIZE_Y},
        i32(tileIndex.x+1)*${TILESIZE_X},i32(tileIndex.y+1)*${TILESIZE_Y}));
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }

    var totalLightContrib = vec3f(0, 0, 0);
    for (var lightIdx = lightIdxOffset; lightIdx < lightIdxOffset + lightCount; lightIdx++) {
        let light = lightSet.lights[lightIndices[lightIdx]];
        totalLightContrib += calculateLightContrib(light, in.pos, normalize(in.nor));
    }

    var finalColor = diffuseColor.rgb * totalLightContrib;
    if((finalColor!=finalColor).x){
        finalColor = vec3f(1.,0.,1.);
    }
    //return vec4f(in.fragPos.xy / viewportSize, 0.0,1.0);
    //return vec4(f32(lightCount)*0.1, -in.viewPos.z*0.1, 0., 1);

    // var ndc = in.fragPos.xy / u_Camera.viewportSize * 2. - 1.;
    // ndc.y *=-1;
    // var viewPos = normalize(vec3f(ndc * u_Camera.cameraParams, 1.f));
    // var viewPos2 = in.viewPos;
    // viewPos2.z *= -1;
    // return vec4(normalize(viewPos2).xy, 0., 1.);
    //return vec4(normalize(viewPos), 1.);
    
    return vec4(finalColor, 1);
}
