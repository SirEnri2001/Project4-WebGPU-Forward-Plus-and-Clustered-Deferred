// TODO-3: implement the Clustered Deferred fullscreen fragment shader

@group(${bindGroup_scene}) @binding(0) var<uniform> u_Camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
// Similar to the Forward+ fragment shader, but with vertex information coming from the G-buffer instead.
@group(${bindGroup_lightCull}) @binding(2) var<storage, read_write> lightIndices: array<i32>;
@group(${bindGroup_lightCull}) @binding(3) var<storage, read_write> lightGrid: array<i32>;
@group(${bindGroup_lightCull}) @binding(4) var<storage, read_write> tileMinMax: array<i32>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read_write> lightCountTotal: atomic<i32>;
@group(${bindGroup_lightCull}) @binding(6) var<storage, read_write> gridSize: vec3i;


@group(${bindGroup_deferredLighting}) @binding(0) var gbufferBaseColor: texture_2d<f32>;
@group(${bindGroup_deferredLighting}) @binding(1) var gbufferNormal: texture_2d<f32>;
@group(${bindGroup_deferredLighting}) @binding(2) var gbufferDepth: texture_2d<f32>;



struct FragmentInput
{
    @builtin(position) fragPos: vec4f,
}

@fragment
fn main(in: FragmentInput) -> @location(0) vec4f
{
    var viewportSize = vec2f(u_Camera.viewportSize);
    var uv = in.fragPos.xy / viewportSize;
    var pixelPos = vec2u(in.fragPos.xy);
    var tile_x = viewportSize.x / ${X_SLICES};
    var tile_y = viewportSize.y / ${Y_SLICES};
    var tileIndex = vec2i(i32(in.fragPos.x / tile_x), i32(in.fragPos.y / tile_y));
    var gridDimXY = i32(gridSize.x * gridSize.y);
    var tilePos_Pixel = vec4f(f32(tileIndex.x)*tile_x,f32(tileIndex.y)*tile_y,
        f32(tileIndex.x+1)*tile_x,f32(tileIndex.y+1)*tile_y);
    let baseColor = textureLoad(gbufferBaseColor, pixelPos, 0).xyz;
    let normal = textureLoad(gbufferNormal, pixelPos, 0).xyz;
    let depth = textureLoad(gbufferDepth, pixelPos, 0).x;
    var pos_ndc = uv*2.-1.;
    pos_ndc.y *=-1;
    var pos_view = vec3f(pos_ndc * u_Camera.cameraParams, -1.) * -depth; 
    var normal_view = (u_Camera.viewMat * vec4f(normal, 0.)).xyz;
    var totalLightContrib = vec3f(0, 0, 0);
    var totalLightCount = 0;
    for(var z_index = 0; z_index < ${Z_SLICES};z_index++){
        var gridId = tileIndex.x + gridSize.x * tileIndex.y + z_index * ${Z_SLICES} * gridDimXY;
        var lightCount = lightGrid[2*gridId + 1];
        var lightIdxOffset = lightGrid[2*gridId];
        for (var lightIdx = lightIdxOffset; lightIdx < lightIdxOffset + lightCount; lightIdx++) {
            let light = lightSet.lights[lightIndices[lightIdx]];
            var light_pos_view = (u_Camera.viewMat * vec4f(light.pos, 1.)).xyz;
            totalLightContrib += calculateLightContrib_View(light, light_pos_view, pos_view, normalize(normal_view));
        }
        totalLightCount+=lightCount;
    }

    var finalColor = baseColor * totalLightContrib;
    if((finalColor!=finalColor).x){
        finalColor = vec3f(1.,0.,1.);
    }
    //return vec4f(abs(pos_view)*0.1, 1.);
    //return vec4f(in.fragPos.xy / viewportSize, 0.0,1.0);
    //return vec4(f32(totalLightCount)*0.05, 0.0, 0., 1);

    // var ndc = in.fragPos.xy / u_Camera.viewportSize * 2. - 1.;
    // ndc.y *=-1;
    // var viewPos = normalize(vec3f(ndc * u_Camera.cameraParams, 1.f));
    // var viewPos2 = in.viewPos;
    // viewPos2.z *= -1;
    // return vec4(normalize(viewPos2).xy, 0., 1.);
    //return vec4(normalize(viewPos), 1.);
    
    return vec4(finalColor, 1);
}
