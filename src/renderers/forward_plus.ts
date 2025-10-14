import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class ForwardPlusRenderer extends renderer.Renderer {
    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;
    lightCullingBindGroupLayout: GPUBindGroupLayout;
    lightCullingBindGroup: GPUBindGroup;

    depthRTTexture: GPUTexture;
    depthRTTextureView: GPUTextureView;
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    lightIndices: GPUBuffer; // output of compute shader
    lightIndicesArray : Int32Array;
    lightGrid: GPUBuffer; // output of compute shader
    lightGridArray : Int32Array;
    tileMinMax: GPUBuffer;
    tileMinMaxArray:Int32Array;

    
    lightCountTotalArray: Int32Array;
    lightCountTotal: GPUBuffer;
    gridSizeArray:Int32Array;
    gridSize: GPUBuffer;

    zPrepassGraphicsPipeline: GPURenderPipeline;
    computeDepthMinMaxComputePipeline: GPUComputePipeline;
    computeTileVisibleLightIndexComputePipeline: GPUComputePipeline;
    forwardPlusGraphicsPipeline: GPURenderPipeline;

    // shader modules
    forwardPlusCSModule: GPUShaderModule;

    constructor(stage: Stage) {
        super(stage);
        this.lightIndicesArray  = new Int32Array(this.lights.numLights);
        this.lightIndicesArray.set(Array(this.lights.numLights).fill(0));
        this.lightIndices = renderer.device.createBuffer({
            label: "lightIndices",
            size: this.lightIndicesArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.lightGridArray  = new Int32Array(2*shaders.constants.MAX_GRID_SIZE*shaders.constants.MAX_GRID_SIZE);
        this.lightGridArray.set(Array(2*shaders.constants.MAX_GRID_SIZE*shaders.constants.MAX_GRID_SIZE));
        this.lightGrid = renderer.device.createBuffer({
            label: "lightGrid",
            size: this.lightGridArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.tileMinMaxArray = new Int32Array(2*shaders.constants.MAX_GRID_SIZE*shaders.constants.MAX_GRID_SIZE);
        this.tileMinMax = renderer.device.createBuffer({
            label: "tileMinMax",
            size: this.tileMinMaxArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });
        this.lightCountTotalArray = new Int32Array(1);
        this.lightCountTotalArray.set([0], 0);
        this.lightCountTotal = renderer.device.createBuffer({
            label:"lightCountTotal",
            size: this.lightCountTotalArray.byteLength, // sizeof(i32)==4
            usage: GPUBufferUsage.STORAGE  | GPUBufferUsage.COPY_DST| GPUBufferUsage.COPY_SRC
        });
        this.gridSizeArray = new Int32Array(2);
        this.gridSize = renderer.device.createBuffer({
            label:"gridSize",
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // create depth texture RT
        this.depthRTTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "r32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthRTTextureView = this.depthRTTexture.createView();
        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        // Create bind groups and layouts
        this.sceneUniformsBindGroupLayout = renderer.device.createBindGroupLayout({
            label: "scene uniforms bind group layout",
            entries: [
                {
                    binding:0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer:{type:"uniform"}
                },
                // add an entry for camera uniforms at binding 0, visible to only the vertex shader, and of type "uniform"
                { // lightSet
                    binding: 1,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT | GPUShaderStage.COMPUTE,
                    buffer: { type: "read-only-storage" }
                }
            ]
        });

        this.sceneUniformsBindGroup = renderer.device.createBindGroup({
            label: "scene uniforms bind group",
            layout: this.sceneUniformsBindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: stage.camera.uniformsBuffer }},
                // dd an entry for camera uniforms at binding 0
                // you can access the camera using `this.camera`
                // if you run into TypeScript errors, you're probably trying to upload the host buffer instead
                {
                    binding: 1,
                    resource: { buffer: this.lights.lightSetStorageBuffer }
                }
            ]
        });
        
        this.lightCullingBindGroupLayout = renderer.device.createBindGroupLayout({
            label:"Light Culling CS Layout",
            entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT ,
                texture: {
                sampleType: "unfilterable-float", // "r32float" do not support textureSample, nor it is "float" sample type
                viewDimension: "2d",
                multisampled: false,
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {
                type: "read-only-storage",
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {
                type: "storage",
                },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {
                type: "storage",
                },
            },
            {
                binding: 4,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {
                type: "storage",
                },
            },
            {
                binding: 5,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                buffer: {
                type: "storage",
                },
            },
            ],
        });
        this.lightCullingBindGroup = renderer.device.createBindGroup({
            label:"Light Culling CS Bind",
            layout: this.lightCullingBindGroupLayout,
            entries:[
                {
                    binding:0,
                    resource: this.depthRTTextureView
                },
                {
                    binding:1,
                    resource:{buffer: this.lights.lightSetStorageBuffer}
                },
                {
                    binding:2,
                    resource:{buffer: this.lightIndices}
                },
                {
                    binding:3,
                    resource:{buffer: this.lightGrid}
                },
                {
                    binding:4,
                    resource:{buffer: this.tileMinMax}
                },
                {
                    binding:5,
                    resource:{buffer: this.lightCountTotal}
                }
            ]
        });

        this.zPrepassGraphicsPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "forward plus z prepass pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "forward plus vert shader",
                    code: shaders.naiveVertSrc
                }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "forward plus z prepass frag shader",
                    code: shaders.zPrepassFragSrc,
                }),
                targets: [
                    {
                        format: "r32float",
                    }
                ]
            }
        });

        this.forwardPlusCSModule = renderer.device.createShaderModule({
            label: 'Light Culling CS shader',
            code: shaders.forwardPlusCSRawSrc
        });

        
        this.computeDepthMinMaxComputePipeline = renderer.device.createComputePipeline({
            label: 'Light Culling CS Pipeline',
            layout: renderer.device.createPipelineLayout({
                label: "Compute Shader Pipeline Layout",
                bindGroupLayouts: [
                        this.sceneUniformsBindGroupLayout,
                        null,
                        this.lightCullingBindGroupLayout],
            }),
            compute: {
                module: this.forwardPlusCSModule,
                entryPoint:"computeDepthMinMax"
            },
        });

        this.computeTileVisibleLightIndexComputePipeline = renderer.device.createComputePipeline({
            label: 'Light Culling CS Pipeline',
            layout: renderer.device.createPipelineLayout({
                label: "Compute Shader Pipeline Layout",
                bindGroupLayouts: [
                        this.sceneUniformsBindGroupLayout,
                        null,
                        this.lightCullingBindGroupLayout],
            }),
            compute: {
                module: this.forwardPlusCSModule,
                entryPoint:"computeTileVisibleLightIndex"
            },
        });

        this.forwardPlusGraphicsPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "forward plus shading pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    this.lightCullingBindGroupLayout,
                    renderer.materialBindGroupLayout
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "forward plus vert shader",
                    code: shaders.naiveVertSrc
                }),
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "forward plus frag shader",
                    code: shaders.forwardPlusFragSrc,
                }),
                targets: [
                    {
                        format: renderer.canvasFormat,
                    }
                ]
            }
        });
    }

    createLayouts() : void{

    }

    async LightCulling() : Promise<void> {
        const encoder = renderer.device.createCommandEncoder();
        var gridX = Math.floor((renderer.canvas.width + shaders.constants.TILESIZE_X - 1) / shaders.constants.TILESIZE_X);
        var gridY = Math.floor((renderer.canvas.height + shaders.constants.TILESIZE_Y - 1) / shaders.constants.TILESIZE_Y);
        renderer.device.queue.writeBuffer(this.lightCountTotal, 0, this.lightCountTotalArray.buffer);
        renderer.device.queue.writeBuffer(this.lightIndices, 0, this.lightIndicesArray.buffer);
        renderer.device.queue.writeBuffer(this.lightGrid, 0, this.lightGridArray.buffer);
        const computePass1 = encoder.beginComputePass();
        computePass1.setPipeline(this.computeDepthMinMaxComputePipeline);
        computePass1.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
        computePass1.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        computePass1.dispatchWorkgroups(
            gridX, 
            gridY, 
            1
        );
        computePass1.end();

        const computePass2 = encoder.beginComputePass();
        computePass2.setPipeline(this.computeTileVisibleLightIndexComputePipeline);
        computePass2.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
        computePass2.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        computePass2.dispatchWorkgroups(
            gridX, 
            gridY, 
            1
        );
        computePass2.end();


        const StagingBuffer = renderer.device.createBuffer({
            size: this.lightCountTotalArray.byteLength,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
        });
        encoder.copyBufferToBuffer(
            this.lightCountTotal /* source buffer */,
            0 /* source offset */,
            StagingBuffer /* destination buffer */,
            0 /* destination offset */,
            this.lightCountTotalArray.byteLength /* size */
        );
        renderer.device.queue.submit([encoder.finish()]);
        await StagingBuffer.mapAsync(GPUMapMode.READ);
        const res = new Int32Array(StagingBuffer.getMappedRange());
        console.log("Active lights all grids: " + res[0]);
        // console.log("Grid X "+ gridX);
        // console.log("Grid Y "+gridY);
    }

    zPrepass(): void{
        const encoder = renderer.device.createCommandEncoder();
        // z prepass
        const renderPass = encoder.beginRenderPass({
            label: "forward p render pass",
            colorAttachments: [
                {
                    view: this.depthRTTextureView,
                    clearValue: [0, 0, 0, 0],
                    loadOp: "clear",
                    storeOp: "store"
                }
            ],
            depthStencilAttachment: {
                view: this.depthTextureView,
                depthClearValue: 1.0,
                depthLoadOp: "clear",
                depthStoreOp: "store"
            }
        });
        renderPass.setPipeline(this.zPrepassGraphicsPipeline);

        // bind `this.sceneUniformsBindGroup` to index `shaders.constants.bindGroup_scene`
        renderPass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        this.sceneUniformsBindGroup
        this.scene.iterate(node => {
            renderPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
        }, material => {
            renderPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
        }, primitive => {
            renderPass.setVertexBuffer(0, primitive.vertexBuffer);
            renderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
            renderPass.drawIndexed(primitive.numIndices);
        });

        renderPass.end();

        renderer.device.queue.submit([encoder.finish()]);
    }

    override draw() {
        this.zPrepass();

        // light culling by tiles
        var res = this.LightCulling();
        res.then(()=>{
            // actual shading
            {
                const canvasTextureView = renderer.context.getCurrentTexture().createView();
                const encoder = renderer.device.createCommandEncoder();
                const renderPass = encoder.beginRenderPass({
                    label: "forward p render pass",
                    colorAttachments: [
                        {
                            view: canvasTextureView,
                            clearValue: [0, 0, 0, 0],
                            loadOp: "clear",
                            storeOp: "store"
                        }
                    ],
                    depthStencilAttachment: {
                        view: this.depthTextureView,
                        depthClearValue: 1.0,
                        depthLoadOp: "clear",
                        depthStoreOp: "store"
                    }
                });
                renderPass.setPipeline(this.forwardPlusGraphicsPipeline);
                renderPass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
                renderPass.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
                this.scene.iterate(node => {
                    renderPass.setBindGroup(shaders.constants.bindGroup_model, node.modelBindGroup);
                }, material => {
                    renderPass.setBindGroup(shaders.constants.bindGroup_material, material.materialBindGroup);
                }, primitive => {
                    renderPass.setVertexBuffer(0, primitive.vertexBuffer);
                    renderPass.setIndexBuffer(primitive.indexBuffer, 'uint32');
                    renderPass.drawIndexed(primitive.numIndices);
                });

                renderPass.end();

                renderer.device.queue.submit([encoder.finish()]);
            }
        })
    }
}
