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
    @location(0) packedVal: vec4u,
}

@fragment
fn main(in: FragmentInput) -> RenderTargets
{
    let diffuseColor = textureSample(diffuseTex, diffuseTexSampler, in.uv);
    if (diffuseColor.a < 0.5f) {
        discard;
    }
    var rt: RenderTargets;
    var normal = vec4(normalize(in.nor.xyz), in.viewPos.z / 16.);
    rt.packedVal = vec4u(pack4x8snorm(normal), pack4x8snorm(diffuseColor), u32(-in.viewPos.z*${DEPTH_INTEGER_SCALE}) , 1u);
    return rt;
}
