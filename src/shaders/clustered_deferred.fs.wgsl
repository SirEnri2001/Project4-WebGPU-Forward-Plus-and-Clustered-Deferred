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
@group(${bindGroup_lightCull}) @binding(6) var<storage, read_write> gridSize: vec3i;
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
    var viewportSize = vec2f(u_Camera.viewportSize);
    var tile_x = viewportSize.x / ${X_SLICES};
    var tile_y = viewportSize.y / ${Y_SLICES};
    var tileIndex = vec2i(i32(in.fragPos.x / tile_x), i32(in.fragPos.y / tile_y));
    var gridDimXY = i32(gridSize.x * gridSize.y);
    var tilePos_Pixel = vec4f(f32(tileIndex.x)*tile_x,f32(tileIndex.y)*tile_y,
        f32(tileIndex.x+1)*tile_x,f32(tileIndex.y+1)*tile_y);
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }
    var totalLightContrib = vec3f(0, 0, 0);
    var totalLightCount = 0;
    for(var z_index = 0; z_index < ${Z_SLICES};z_index++){
        var gridId = tileIndex.x + gridSize.x * tileIndex.y + z_index * ${Z_SLICES} * gridDimXY;
        var lightCount = lightGrid[2*gridId + 1];
        var lightIdxOffset = lightGrid[2*gridId];
        for (var lightIdx = lightIdxOffset; lightIdx < lightIdxOffset + lightCount; lightIdx++) {
            let light = lightSet.lights[lightIndices[lightIdx]];
            totalLightContrib += calculateLightContrib(light, in.pos, normalize(in.nor));
        }
        totalLightCount+=lightCount;
    }

    
    

    var finalColor = diffuseColor.rgb * totalLightContrib;
    if((finalColor!=finalColor).x){
        finalColor = vec3f(1.,0.,1.);
    }
    //return vec4f(in.fragPos.xy / viewportSize, 0.0,1.0);
    //return vec4(f32(totalLightCount)*0.005, 0.0, 0., 1);

    // var ndc = in.fragPos.xy / u_Camera.viewportSize * 2. - 1.;
    // ndc.y *=-1;
    // var viewPos = normalize(vec3f(ndc * u_Camera.cameraParams, 1.f));
    // var viewPos2 = in.viewPos;
    // viewPos2.z *= -1;
    // return vec4(normalize(viewPos2).xy, 0., 1.);
    //return vec4(normalize(viewPos), 1.);
    
    return vec4(finalColor, 1);
}
