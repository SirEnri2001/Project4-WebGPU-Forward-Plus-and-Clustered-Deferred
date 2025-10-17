@group(${bindGroup_lightCull}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_lightCull}) @binding(2) var<storage, read_write> lightIndices_ST: array<i32>;
@group(${bindGroup_lightCull}) @binding(3) var<storage, read_write> lightGrid_ST: array<i32>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read_write> lightCountTotal_ST: atomic<i32>;
@group(${bindGroup_lightCull}) @binding(6) var<storage, read_write> gridSize: vec3i;
@group(${bindGroup_scene}) @binding(0) var<uniform> u_Camera: CameraUniforms;

// workgroup shared mems
var<workgroup> lightIndexArray_WG: array<i32, ${MAX_LIGHTS_PER_CLUSTER}>;
var<workgroup> lightCounter_WG: atomic<i32>;

fn planeDistance(p: vec3f, planePoint: vec3f, planeNormal: vec3f)-> f32{
    var v = p - planePoint;
    return dot(planeNormal, v);
}

@compute @workgroup_size(${LIGHTS_BATCH_SIZE})
fn computeTileVisibleLightIndex(
    @builtin(global_invocation_id) index_u: vec3u, 
    @builtin(local_invocation_id) tid_u: vec3u, 
    @builtin(workgroup_id) blockId_u: vec3u 
) {
    var index = vec3i(index_u);
    var tid = vec3i(tid_u);
    var blockId = vec3i(blockId_u);
    // blockId.xy represents viewport grid coords, blockId.z is which batch of light we are currently appending
    let i = index.x;
    let gridCounts = gridSize.x * gridSize.y;
    var gridId = i32(blockId.x + gridSize.x * blockId.y + blockId.z* ${Z_SLICES} * gridCounts);
    var viewportSize = vec2f(u_Camera.viewportSize);
    var tile_x = viewportSize.x / ${X_SLICES};
    var tile_y = viewportSize.y / ${Y_SLICES};
    var tilePos_Pixel = vec4f(f32(blockId.x)*tile_x,f32(blockId.y)*tile_y,
        f32(blockId.x+1)*tile_x,f32(blockId.y+1)*tile_y);
    var minZ = -0.1;
    var maxZ = -2000.;
    var tileDepthMin = maxZ / ${Z_SLICES} * f32(blockId.z+1);
    var tileDepthMax = maxZ / ${Z_SLICES} * f32(blockId.z);
    var isLightValid: bool = false;
    var debugVal: i32 = 0;

    if(tid.x==0){
        atomicStore(&lightCounter_WG, 0);
        lightGrid_ST[2*gridId] = 0;
        lightGrid_ST[2*gridId + 1] = 0;
    }
    workgroupBarrier();

    for(var lightIdx = i32(tid.x); lightIdx < i32(lightSet.numLights); lightIdx += ${LIGHTS_BATCH_SIZE}) {
        // do the culling
        let light = lightSet.lights[lightIdx];
    
        var lightPos_View = (u_Camera.viewMat * vec4f(light.pos, 1.)).xyz;
        var lightRadius = f32(${lightRadius});
    
        isLightValid = 
             lightPos_View.z - lightRadius< 0.
             && lightPos_View.z + lightRadius > tileDepthMin 
             && lightPos_View.z - lightRadius < tileDepthMax;
        if(isLightValid){
            var p_bottomleft_ndc = vec2f(tilePos_Pixel.x / viewportSize.x, tilePos_Pixel.y / viewportSize.y)*2.-1.;
            var p_bottomright_ndc = vec2f(tilePos_Pixel.z / viewportSize.x, tilePos_Pixel.y / viewportSize.y)*2.-1.;
            var p_topleft_ndc = vec2f(tilePos_Pixel.x / viewportSize.x, tilePos_Pixel.w / viewportSize.y)*2.-1.;
            var p_topright_ndc = vec2f(tilePos_Pixel.z / viewportSize.x, tilePos_Pixel.w / viewportSize.y)*2.-1.;
            var p_bottomleft_View = vec3f(p_bottomleft_ndc * u_Camera.cameraParams, -1.) * -lightPos_View.z; 
            var p_bottomright_View = vec3f(p_bottomright_ndc * u_Camera.cameraParams, -1.) * -lightPos_View.z; 
            var p_topleft_View = vec3f(p_topleft_ndc * u_Camera.cameraParams, -1.) * -lightPos_View.z; 
            var p_topright_View = vec3f(p_topright_ndc * u_Camera.cameraParams, -1.) * -lightPos_View.z;
            var viewPos_View = vec3f(0.,0.,0.);

            if(planeDistance(lightPos_View, viewPos_View, normalize(cross(p_bottomleft_View, p_bottomright_View)))>lightRadius){
                isLightValid = false;
            }
            else if(planeDistance(lightPos_View, viewPos_View, normalize(cross(p_topright_View, p_topleft_View)))>lightRadius){
                isLightValid = false;
            }
            else if(planeDistance(lightPos_View, viewPos_View, normalize(cross(p_bottomright_View, p_topright_View)))>lightRadius){
                isLightValid = false;
            }
            else if(planeDistance(lightPos_View, viewPos_View, normalize(cross(p_topleft_View, p_bottomleft_View)))>lightRadius){
                isLightValid = false;
            }
        }
        if isLightValid {
            let arrayIndex = atomicAdd(&lightCounter_WG, 1);
            if arrayIndex < 2048 {
                lightIndexArray_WG[arrayIndex] = lightIdx;
            }
        }
    }

    workgroupBarrier();
    if(tid.x==0){
        var totalLightCountInTile = min(i32(atomicLoad(&lightCounter_WG)), ${MAX_LIGHTS_PER_CLUSTER});
        var lightOffset = atomicAdd(&lightCountTotal_ST, totalLightCountInTile);
        lightGrid_ST[2*gridId] = i32(lightOffset);
        lightGrid_ST[2*gridId + 1] = i32(totalLightCountInTile);
        for (var index = lightOffset;index<lightOffset + totalLightCountInTile;index++){
            lightIndices_ST[index] = lightIndexArray_WG[index - lightOffset];
        }
    }
}

