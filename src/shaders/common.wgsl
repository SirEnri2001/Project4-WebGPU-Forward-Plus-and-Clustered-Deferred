// CHECKITOUT: code that you add here will be prepended to all shaders

struct Light {
    pos: vec3f,
    color: vec3f
}

struct LightSet {
    numLights: u32,
    lights: array<Light>
}

// TODO-2: you may want to create a ClusterSet struct similar to LightSet

struct CameraUniforms {
    viewProjMat : mat4x4f,
    viewMat: mat4x4f,
    viewportSize: vec2f,
    cameraParams: vec2f
}

// CHECKITOUT: this special attenuation function ensures lights don't affect geometry outside the maximum light radius
fn rangeAttenuation(distance: f32) -> f32 {
    return clamp(1.f - pow(distance / 2, 4.f), 0.f, 1.f) / (distance * distance);
}

fn calculateLightContrib(light: Light, posWorld: vec3f, nor: vec3f) -> vec3f {
    let vecToLight = light.pos - posWorld;
    let distToLight = length(vecToLight);

    let lambert = max(dot(nor, normalize(vecToLight)), 0.f);
    return light.color * lambert  / distToLight / distToLight * rangeAttenuation(distToLight);
}

fn calculateLightContrib_View(light: Light, light_pos_View: vec3f, pos_View: vec3f, nor: vec3f) -> vec3f {
    let vecToLight = light_pos_View - pos_View;
    let distToLight = length(vecToLight);

    let lambert = max(dot(nor, normalize(vecToLight)), 0.f);
    return light.color * lambert  / distToLight / distToLight * rangeAttenuation(distToLight);
}

fn getLightGridIndex(gridId: i32, gridCounts: i32, lightBatchId: i32, lightBatchSize: i32) -> i32{
    return 2*gridId + 2 * lightBatchId * gridCounts;
}