// TODO-3: implement the Clustered Deferred fullscreen vertex shader

// This shader should be very simple as it does not need all of the information passed by the the naive vertex shader.

struct VertexInput
{
    @location(0) pos: vec3f,
}

struct VertexOutput
{
    @builtin(position) fragPos: vec4f,
}

@vertex
fn main(in: VertexInput) -> VertexOutput
{
    var out: VertexOutput;
    out.fragPos = vec4f(in.pos, 1.);
    return out;
}
