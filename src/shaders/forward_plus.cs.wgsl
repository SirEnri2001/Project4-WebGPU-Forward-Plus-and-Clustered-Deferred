

@group(${bindGroup_lightCull}) @binding(0) var depthTexture: texture_2d<f32>;
@group(${bindGroup_lightCull}) @binding(1) var<storage, read> lightSet: LightSet;
@group(${bindGroup_lightCull}) @binding(2) var<storage, read_write> lightIndices_ST: array<i32>;
@group(${bindGroup_lightCull}) @binding(3) var<storage, read_write> lightGrid_ST: array<i32>;
@group(${bindGroup_lightCull}) @binding(4) var<storage, read_write> tileMinMax: array<atomic<i32>>;
@group(${bindGroup_lightCull}) @binding(5) var<storage, read_write> lightCountTotal_ST: atomic<i32>;
@group(${bindGroup_lightCull}) @binding(6) var<storage, read_write> gridSize: vec3i;
@group(${bindGroup_scene}) @binding(0) var<uniform> u_Camera: CameraUniforms;

// workgroup shared mems
var<workgroup> lightIndexArray_WG: array<i32, ${MAX_LIGHTS_PER_TILE}>;
var<workgroup> lightCounter_WG: atomic<i32>;

@compute @workgroup_size(16, 16, 1)
fn computeDepthMinMax(
    @builtin(global_invocation_id) index_u: vec3u, 
    @builtin(local_invocation_id) tid_u: vec3u, 
    @builtin(workgroup_id) blockId_u: vec3u
) {
    var index = vec3i(index_u);
    var tid = vec3i(tid_u);
    var blockId = vec3i(blockId_u);
    var gridId = blockId.x + gridSize.x * blockId.y;
    var viewportSize = vec2f(u_Camera.viewportSize);
    var tileSizeX = i32(ceil(viewportSize.x / ${X_SLICES}));
    var tileSizeY = i32(ceil(viewportSize.y / ${Y_SLICES}));
    var pos_pixel = vec2i(
        i32(f32(tid.x) + f32(blockId.x) / ${X_SLICES} * viewportSize.x), 
        i32(f32(tid.y) + f32(blockId.y) / ${Y_SLICES} * viewportSize.y)
    );
    var pos_pixel_end = vec2i(
        (pos_pixel.x + tileSizeX), 
        (pos_pixel.y + tileSizeY)
    );
    var tileCountX = i32(ceil(f32(tileSizeX) / 16));
    var tileCountY = i32(ceil(f32(tileSizeY) / 16));
    if(tid.x==0 && tid.y==0){
        atomicStore(&tileMinMax[2*gridId], 0xfffffff);
        atomicStore(&tileMinMax[2*gridId+1], -0xfffffff);
    }
    var minDepth = 0xfffffff;
    var maxDepth = -0xfffffff;
    for(var y = 0; y < tileSizeY; y+=16){
        // if(pos_pixel.y+y*16>pos_pixel_end.y || pos_pixel.y+y*16>i32(viewportSize.y)){
        //     continue;
        // }
        for(var x = 0; x < tileSizeX;x+=16){
            // if(pos_pixel.x+x*16>pos_pixel_end.x || pos_pixel.x+x*16>i32(viewportSize.x)){
            //     continue;
            // }
            var curPos_pixel = vec2i(pos_pixel.x+x, pos_pixel.y+y);
            curPos_pixel.y = i32(viewportSize.y) - curPos_pixel.y;
            let depthValue = i32(textureLoad(depthTexture, curPos_pixel, 0).x*${DEPTH_INTEGER_SCALE});
            if(depthValue==0){
                continue;
            }
            if(depthValue>maxDepth){
                maxDepth = depthValue;
            }
            if(depthValue<minDepth){
                minDepth = depthValue;
            }
        }
        
    }
    workgroupBarrier();
    let minDepthValue = atomicMin(&tileMinMax[2*gridId], i32(minDepth));
    let maxDepthValue = atomicMax(&tileMinMax[2*gridId+1], i32(maxDepth));
    workgroupBarrier();
}

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
    var gridId = blockId.x + gridSize.x * blockId.y;
    let gridCounts = gridSize.x * gridSize.y;
    var viewportSize = vec2f(u_Camera.viewportSize);
    var tile_x = viewportSize.x / ${X_SLICES};
    var tile_y = viewportSize.y / ${Y_SLICES};
    var tilePos_Pixel = vec4f(f32(blockId.x)*tile_x,f32(blockId.y)*tile_y,
        f32(blockId.x+1)*tile_x,f32(blockId.y+1)*tile_y);
    var tileDepthMin = f32(atomicLoad(&tileMinMax[2*gridId])) / f32(${DEPTH_INTEGER_SCALE});
    var tileDepthMax = f32(atomicLoad(&tileMinMax[2*gridId+1])) / f32(${DEPTH_INTEGER_SCALE});
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
        var lightRadius = f32(2);
    
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
            if arrayIndex < ${MAX_LIGHTS_PER_TILE} {
                lightIndexArray_WG[arrayIndex] = lightIdx;
            }
        }
    }

    workgroupBarrier();
    if(tid.x==0){
        var totalLightCountInTile = min(i32(atomicLoad(&lightCounter_WG)), ${MAX_LIGHTS_PER_TILE});
        var lightOffset = atomicAdd(&lightCountTotal_ST, totalLightCountInTile);
        lightGrid_ST[2*gridId] = i32(lightOffset);
        lightGrid_ST[2*gridId + 1] = i32(totalLightCountInTile);
        for (var index = lightOffset;index<lightOffset + totalLightCountInTile;index++){
            lightIndices_ST[index] = lightIndexArray_WG[index - lightOffset];
        }
    }
}

