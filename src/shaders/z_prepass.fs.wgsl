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
    return vec4(in.viewPos.z, 0.,0., 1);
}
