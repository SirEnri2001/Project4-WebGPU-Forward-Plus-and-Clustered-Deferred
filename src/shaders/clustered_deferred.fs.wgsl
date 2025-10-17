@group(${bindGroup_material}) @binding(0) var diffuseTex: texture_2d<f32>;
@group(${bindGroup_material}) @binding(1) var diffuseTexSampler: sampler;

struct FragmentInput
{
    @location(0) pos: vec3f,
    @location(1) nor: vec3f,
    @location(2) uv: vec2f,
    @location(3) viewPos: vec3f
}

struct RenderTargets{
    @location(0) baseColor: vec4f,
    @location(1) normal: vec4f,
    @location(2) depth: vec4f,
}

@fragment
fn main(in: FragmentInput) -> RenderTargets
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }
    var rt: RenderTargets;
    rt.baseColor = diffuseColor;
    rt.normal = vec4f(in.nor, 1.);
    rt.depth = vec4f(in.viewPos.z, 0.,0., 1.);
    return rt;
}
