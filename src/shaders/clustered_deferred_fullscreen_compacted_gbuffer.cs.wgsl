// TODO-3: implement the Clustered Deferred fullscreen fragment shader

@group(${bindGroup_scene}) @binding(0) var<uniform> u_Camera: CameraUniforms;
@group(${bindGroup_scene}) @binding(1) var<storage, read> lightSet: LightSet;
// Similar to the Forward+ fragment shader, but with vertex information coming from the G-buffer instead.
@group(${bindGroup_lightCull}) @binding(2) var<storage, read_write> lightIndices: array<i32>;
@group(${bindGroup_lightCull}) @binding(3) var<storage, read_write> lightGrid: array<i32>;
@group(${bindGroup_lightCull}) @binding(4) var<storage, read_write> tileMinMax: array<i32>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read_write> lightCountTotal: atomic<i32>;
@group(${bindGroup_lightCull}) @binding(6) var<storage, read_write> gridSize: vec3i;

@group(${bindGroup_deferredLighting}) @binding(0) var gbufferPacked: texture_2d<u32>;
@group(${bindGroup_deferredLighting}) @binding(1) var frameBuffer: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(16, 16, 1)
fn deferredShadingCS(
    @builtin(global_invocation_id) index_u: vec3u, 
    @builtin(local_invocation_id) tid_u: vec3u, 
    @builtin(workgroup_id) blockId_u: vec3u 
)
{
    var index = vec3i(index_u);
    var tid = vec3i(tid_u);
    var blockId = vec3i(blockId_u);
    var viewportSize = vec2f(u_Camera.viewportSize);
    var fragPos = vec2f(f32(blockId.x*16+tid.x), f32(blockId.y*16+tid.y));
    var uv = fragPos.xy / viewportSize;
    var pixelPos = vec2u(fragPos.xy);
    var tile_x = viewportSize.x / ${X_SLICES};
    var tile_y = viewportSize.y / ${Y_SLICES};
    var invert_y_fragPos = fragPos.xy;
    invert_y_fragPos.y = viewportSize.y - invert_y_fragPos.y;
    var tileIndex = vec2i(i32(invert_y_fragPos.x / tile_x), i32(invert_y_fragPos.y / tile_y));
    var gridDimXY = i32(gridSize.x * gridSize.y);
    var tilePos_Pixel = vec4f(f32(tileIndex.x)*tile_x,f32(tileIndex.y)*tile_y,
        f32(tileIndex.x+1)*tile_x,f32(tileIndex.y+1)*tile_y);
    let packedNormalDepth = textureLoad(gbufferPacked, pixelPos, 0);
    let baseColor = unpack4x8snorm(packedNormalDepth.y).xyz;
    var unpackedNormalDepth = unpack4x8snorm(packedNormalDepth.x);
    var normal = unpackedNormalDepth.xyz;
    let depth = -f32(packedNormalDepth.z)/${DEPTH_INTEGER_SCALE};
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
            totalLightCount+=1;
        }
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
    //textureStore(frameBuffer, pixelPos, vec4f(normal, 1.));
    textureStore(frameBuffer, pixelPos, vec4f(finalColor, 1.));
}
