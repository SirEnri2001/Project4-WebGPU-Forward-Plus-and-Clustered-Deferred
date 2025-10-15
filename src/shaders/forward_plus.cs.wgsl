

@group(${bindGroup_lightCull}) @binding(0) var depthTexture: texture_2d<f32>;
@group(${bindGroup_lightCull}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_lightCull}) @binding(2) var<storage, read_write> lightIndices_ST: array<i32>;
@group(${bindGroup_lightCull}) @binding(3) var<storage, read_write> lightGrid_ST: array<i32>;
@group(${bindGroup_lightCull}) @binding(4) var<storage, read_write> tileMinMax: array<atomic<i32>>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read_write> lightCountTotal_ST: atomic<i32>;
@group(${bindGroup_lightCull}) @binding(6) var<storage, read_write> gridSize: vec3i;
@group(${bindGroup_scene}) @binding(0) var<uniform> u_Camera: CameraUniforms;

// workgroup shared mems
var<workgroup> lightIndexArray_WG: array<i32, 1024>;
var<workgroup> lightCounter_WG: atomic<i32>;

@compute @workgroup_size(${TILESIZE_X}, ${TILESIZE_Y}, 1)
fn computeDepthMinMax(
    @builtin(global_invocation_id) index: vec3u, 
    @builtin(local_invocation_id) tid: vec3u, 
    @builtin(workgroup_id) blockId: vec3u
) {
    let i = index.x;
    var gridId = blockId.x + u32(gridSize.x) * blockId.y;
    var tileSizeX = u32(${TILESIZE_X});
    var tileSizeY = u32(${TILESIZE_Y});
    var viewportSize = vec2u(u_Camera.viewportSize);
    let uv = vec2u(
        (tid.x + blockId.x * tileSizeX), 
        (tid.y + blockId.y * tileSizeY)
    );
    let depthValue = i32(textureLoad(depthTexture, uv, 0).x*${DEPTH_INTEGER_SCALE});
    if(tid.x==0 && tid.y==0){
        atomicStore(&tileMinMax[2*gridId], 0xfffffff);
        atomicStore(&tileMinMax[2*gridId+1], -0xfffffff);
    }
    workgroupBarrier();
    let minDepthValue = atomicMin(&tileMinMax[2*gridId], i32(depthValue));
    let maxDepthValue = atomicMax(&tileMinMax[2*gridId+1], i32(depthValue));
    workgroupBarrier();
}

fn planeDistance(p: vec3f, planePoint: vec3f, planeNormal: vec3f)-> f32{
    var v = p - planePoint;
    return dot(planeNormal, v);
}

@compute @workgroup_size(${LIGHTS_BATCH_SIZE})
fn computeTileVisibleLightIndex(
    @builtin(global_invocation_id) index: vec3u, 
    @builtin(local_invocation_id) tid: vec3u, 
    @builtin(workgroup_id) blockId: vec3u 
) {
    // blockId.xy represents viewport grid coords, blockId.z is which batch of light we are currently appending
    let i = index.x;
    var gridId = i32(blockId.x + u32(gridSize.x) * blockId.y);
    let gridCounts = gridSize.x * gridSize.y;
    var viewportSize = vec2f(u_Camera.viewportSize);
    var tilePos_Pixel = vec4f(vec4i(i32(blockId.x)*${TILESIZE_X},i32(blockId.y)*${TILESIZE_Y},
        i32(blockId.x+1)*${TILESIZE_X},i32(blockId.y+1)*${TILESIZE_Y}));
    tilePos_Pixel.y = viewportSize.y - tilePos_Pixel.y;
    tilePos_Pixel.w = viewportSize.y - tilePos_Pixel.w;
    var tileDepthMin = f32(atomicLoad(&tileMinMax[2*gridId])) / f32(${DEPTH_INTEGER_SCALE});
    var tileDepthMax = f32(atomicLoad(&tileMinMax[2*gridId+1])) / f32(${DEPTH_INTEGER_SCALE});
    let lightIdx = i32(tid.x + blockId.z * ${LIGHTS_BATCH_SIZE});
    let lightBatch = i32(blockId.z);
    var isLightValid: bool = true;
    var debugVal: i32 = 0;
    if(lightIdx>=i32(lightSet.numLights)){
        isLightValid = false;
    }else{
        // do the culling
        let light = lightSet.lights[lightIdx];
    
        var lightPos_View = (u_Camera.viewMat * vec4f(light.pos, 1.)).xyz;
        var lightRadius = f32(${lightRadius});
    
        isLightValid = 
            isLightValid && lightPos_View.z - lightRadius< 0.
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
    }
    if(tid.x==0){
        atomicStore(&lightCounter_WG, 0);
        lightGrid_ST[2*gridId + 2*lightBatch*gridCounts] = 0;
        lightGrid_ST[2*gridId + 1 + 2*lightBatch*gridCounts] = 0;
    }
    workgroupBarrier();
    if isLightValid {
        let arrayIndex = atomicAdd(&lightCounter_WG, 1);
        if arrayIndex < 1024 {
            lightIndexArray_WG[arrayIndex] = lightIdx;
        }
    }
    workgroupBarrier();
    if(tid.x==0){
        var totalLightCountInTile = i32(atomicLoad(&lightCounter_WG));
        var lightOffset = atomicAdd(&lightCountTotal_ST, totalLightCountInTile);
        lightGrid_ST[2*gridId + 2*lightBatch*gridCounts] = i32(lightOffset);
        lightGrid_ST[2*gridId + 1 + 2*lightBatch*gridCounts] = i32(totalLightCountInTile);
        for (var index = lightOffset;index<lightOffset + totalLightCountInTile;index++){
            lightIndices_ST[index] = lightIndexArray_WG[index - lightOffset];
        }
    }
}

