import * as renderer from '../renderer';
import * as shaders from '../shaders/shaders';
import { Stage } from '../stage/stage';

export class ClusteredDeferredRenderer extends renderer.Renderer {
    sceneUniformsBindGroupLayout: GPUBindGroupLayout;
    sceneUniformsBindGroup: GPUBindGroup;
    lightCullingBindGroupLayout: GPUBindGroupLayout;
    lightCullingBindGroup: GPUBindGroup;
    gbuffersBindGroupLayout: GPUBindGroupLayout;
    gbuffersBindGroup: GPUBindGroup;
    gbuffersPackedBindGroupLayout: GPUBindGroupLayout;
    gbuffersPackedBindGroup: GPUBindGroup;

    GBufferBaseColor: GPUTexture;
    GBufferBaseColorView: GPUTextureView;
    GBufferNormal: GPUTexture;
    GBufferNormalView: GPUTextureView;
    GBufferDepth: GPUTexture;
    GBufferDepthView: GPUTextureView;
    GBufferPacked: GPUTexture;
    GBufferPackedView: GPUTextureView;
    depthTexture: GPUTexture;
    depthTextureView: GPUTextureView;

    FrameBuffer: GPUTexture;
    FrameBufferView: GPUTextureView;

    useCSPipeline: boolean = true;
    usePackedGBuffer: boolean = true;

    lightIndices: GPUBuffer; // output of compute shader
    lightIndicesArray : Int32Array;
    lightGrid: GPUBuffer; // output of compute shader
    lightGridArray : Int32Array;
    tileMinMax: GPUBuffer;
    tileMinMaxArray:Int32Array;

    fullscreenTriangleArray: Float32Array;
    fullscreenTriangleBuffer: GPUBuffer;
    
    lightCountTotalArray: Int32Array;
    lightCountTotal: GPUBuffer;
    gridSizeArray:Int32Array;
    gridSize: GPUBuffer;

    BasepassGraphicsPipeline: GPURenderPipeline;
    BasepassPackedGraphicsPipeline: GPURenderPipeline;
    computeTileVisibleLightIndexComputePipeline: GPUComputePipeline;
    deferredLightingGraphicsPipeline: GPURenderPipeline;
    deferredLightingComputePipeline: GPUComputePipeline;
    deferredLightingPackedComputePipeline: GPUComputePipeline;

    // shader modules
    forwardPlusCSModule: GPUShaderModule;

    lightCullingBatchSize = 64;
    avgLightsPerCluster = shaders.constants.AVG_LIGHTS_PER_CLUSTER;

    constructor(stage: Stage) {
        super(stage);
        var gridX = shaders.constants.X_SLICES;
        var gridY = shaders.constants.Y_SLICES;
        var gridZ = shaders.constants.Z_SLICES;
        var gridCounts = gridX * gridY * gridZ;
        
        this.fullscreenTriangleArray = new Float32Array([
            -1, 3, 0,
            -1, -1, 0, 
            3, -1, 0,
        ]);
        this.fullscreenTriangleBuffer = renderer.device.createBuffer({
            label:"Fullscreen Triangle",
            size: this.fullscreenTriangleArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST
        });
        renderer.device.queue.writeBuffer(this.fullscreenTriangleBuffer, 0,  this.fullscreenTriangleArray.buffer);

        this.lightIndicesArray  = new Int32Array(gridCounts*this.avgLightsPerCluster);
        this.lightIndicesArray.set(Array(gridCounts*this.avgLightsPerCluster).fill(0));
        this.lightIndices = renderer.device.createBuffer({
            label: "lightIndices",
            size: this.lightIndicesArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.lightGridArray  = new Int32Array(2*gridCounts);
        this.lightGridArray.set(Array(2*gridCounts).fill(0));
        this.lightGrid = renderer.device.createBuffer({
            label: "lightGrid",
            size: this.lightGridArray.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.tileMinMaxArray = new Int32Array(2*gridCounts);
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

        this.gridSizeArray = new Int32Array(3);
        this.gridSize = renderer.device.createBuffer({
            label:"gridSize",
            size: 16,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        
        this.gridSizeArray[0] = shaders.constants.X_SLICES;
        this.gridSizeArray[1] = shaders.constants.Y_SLICES;
        this.gridSizeArray[2] = shaders.constants.Z_SLICES;
        
        renderer.device.queue.writeBuffer(this.gridSize, 0, this.gridSizeArray.buffer);

        this.GBufferBaseColor = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.GBufferBaseColorView = this.GBufferBaseColor.createView();

        this.GBufferNormal = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "rgba16float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.GBufferNormalView = this.GBufferNormal.createView();

        this.GBufferDepth = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "r32float",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.GBufferDepthView = this.GBufferDepth.createView();

        this.GBufferPacked = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "rgba32uint",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.GBufferPackedView = this.GBufferPacked.createView();

        this.depthTexture = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: "depth24plus",
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING
        });
        this.depthTextureView = this.depthTexture.createView();

        this.FrameBuffer = renderer.device.createTexture({
            size: [renderer.canvas.width, renderer.canvas.height],
            format: 'rgba32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING
        });
        this.FrameBufferView = this.FrameBuffer.createView();

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
            // {
            //     binding: 0,
            //     visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT ,
            //     texture: {
            //     sampleType: "unfilterable-float", // "r32float" do not support textureSample, nor it is "float" sample type
            //     viewDimension: "2d",
            //     multisampled: false,
            //     },
            // },
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
            {
                binding: 6,
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
                },
                {
                    binding:6,
                    resource:{buffer: this.gridSize}
                }
            ]
        });

        this.gbuffersBindGroupLayout = renderer.device.createBindGroupLayout({
            label:"GBuffer Layout",
            entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                texture: {
                sampleType: "unfilterable-float", // "r32float" do not support textureSample, nor it is "float" sample type
                viewDimension: "2d",
                multisampled: false,
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                texture: {
                sampleType: "unfilterable-float", // "r32float" do not support textureSample, nor it is "float" sample type
                viewDimension: "2d",
                multisampled: false,
                },
            },
            {
                binding: 2,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                texture: {
                sampleType: "unfilterable-float", // "r32float" do not support textureSample, nor it is "float" sample type
                viewDimension: "2d",
                multisampled: false,
                },
            },
            {
                binding: 3,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                access:	"write-only",
                format:"rgba8unorm",
                viewDimension:"2d"
                },
            },
            ],
        });
        this.gbuffersBindGroup = renderer.device.createBindGroup({
            label:"GBuffer Bind",
            layout: this.gbuffersBindGroupLayout,
            entries:[
                {
                    binding:0,
                    resource: this.GBufferBaseColorView
                },
                {
                    binding:1,
                    resource: this.GBufferNormalView
                },
                {
                    binding:2,
                    resource: this.GBufferDepthView
                },
                {
                    binding:3,
                    resource: renderer.context.getCurrentTexture().createView()
                },
            ]
        });

        this.gbuffersPackedBindGroupLayout = renderer.device.createBindGroupLayout({
            label:"GBufferPacked Layout",
            entries: [
            {
                binding: 0,
                visibility: GPUShaderStage.COMPUTE | GPUShaderStage.FRAGMENT,
                texture: {
                sampleType: "uint", // "r32float" do not support textureSample, nor it is "float" sample type
                viewDimension: "2d",
                multisampled: false,
                },
            },
            {
                binding: 1,
                visibility: GPUShaderStage.COMPUTE,
                storageTexture: {
                access:	"write-only",
                format:"rgba8unorm",
                viewDimension:"2d"
                },
            },
            ],
        });
        this.gbuffersPackedBindGroup = renderer.device.createBindGroup({
            label:"GBufferPacked Bind",
            layout: this.gbuffersPackedBindGroupLayout,
            entries:[
                {
                    binding:0,
                    resource: this.GBufferPackedView
                },
                {
                    binding:1,
                    resource: renderer.context.getCurrentTexture().createView()
                },
            ]
        });

        this.BasepassGraphicsPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "basepass pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    null,
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
                    label: "deferred basepass vert shader",
                    code: shaders.naiveVertSrc
                }),
                entryPoint:"main",
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "deferred basepass frag shader",
                    code: shaders.clusteredDeferredFragSrc,
                }),
                entryPoint:"main",
                targets: [
                    {
                        format: "rgba16float",
                    },
                    {
                        format: "rgba16float",
                    },
                    {
                        format: "r32float",
                    }
                ]
            }
        });

        this.BasepassPackedGraphicsPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "basepass packed pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    renderer.modelBindGroupLayout,
                    null,
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
                    label: "deferred basepass vert shader",
                    code: shaders.naiveVertSrc
                }),
                entryPoint:"main",
                buffers: [ renderer.vertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "deferred packed basepass frag shader",
                    code: shaders.clusteredDeferredPackedFragSrc,
                }),
                entryPoint:"main",
                targets: [
                    {
                        format: "rgba32uint",
                    }
                ]
            }
        });

        this.forwardPlusCSModule = renderer.device.createShaderModule({
            label: 'Light Culling CS shader',
            code: shaders.clusteringComputeSrc
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

        this.deferredLightingComputePipeline = renderer.device.createComputePipeline({
            label: 'Deferred Lighting CS Pipeline',
            layout: renderer.device.createPipelineLayout({
                label: "Compute Shader Pipeline Layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    null,
                    this.lightCullingBindGroupLayout,
                    this.gbuffersBindGroupLayout,
                ],
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    label: "Deferred lighting cs shader",
                    code: shaders.clusteredDeferredFullscreenCSSrc,
                }),
                entryPoint:"deferredShadingCS"
            },
        });

        this.deferredLightingPackedComputePipeline = renderer.device.createComputePipeline({
            label: 'Deferred Lighting Packed CS Pipeline',
            layout: renderer.device.createPipelineLayout({
                label: "Packed Compute Shader Pipeline Layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    null,
                    this.lightCullingBindGroupLayout,
                    this.gbuffersPackedBindGroupLayout,
                ],
            }),
            compute: {
                module: renderer.device.createShaderModule({
                    label: "Deferred lighting cs shader",
                    code: shaders.clusteredDeferredFullscreenPackedCSSrc,
                }),
                entryPoint:"deferredShadingCS"
            },
        });

        this.deferredLightingGraphicsPipeline = renderer.device.createRenderPipeline({
            layout: renderer.device.createPipelineLayout({
                label: "deferred lighting pipeline layout",
                bindGroupLayouts: [
                    this.sceneUniformsBindGroupLayout,
                    null,
                    this.lightCullingBindGroupLayout,
                    this.gbuffersBindGroupLayout,
                ]
            }),
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: "less",
                format: "depth24plus"
            },
            vertex: {
                module: renderer.device.createShaderModule({
                    label: "Lighting full screen vert shader",
                    code: shaders.clusteredDeferredFullscreenVertSrc,
                }),
                entryPoint:"main",
                buffers: [ renderer.fullscreenVertexBufferLayout ]
            },
            fragment: {
                module: renderer.device.createShaderModule({
                    label: "Lighting full screen frag shader",
                    code: shaders.clusteredDeferredFullscreenFragSrc,
                }),
                entryPoint:"main",
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
        var gridX = shaders.constants.X_SLICES;
        var gridY = shaders.constants.Y_SLICES;
        var gridZ = shaders.constants.Z_SLICES;
        renderer.device.queue.writeBuffer(this.lightCountTotal, 0, this.lightCountTotalArray.buffer);
        renderer.device.queue.writeBuffer(this.lightIndices, 0, this.lightIndicesArray.buffer);
        renderer.device.queue.writeBuffer(this.lightGrid, 0, this.lightGridArray.buffer);
        const computePass2 = encoder.beginComputePass();
        computePass2.setPipeline(this.computeTileVisibleLightIndexComputePipeline);
        computePass2.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
        computePass2.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
        computePass2.dispatchWorkgroups(
            gridX, 
            gridY, 
            gridZ
        );
        computePass2.end();
        var StagingBuffer = renderer.device.createBuffer({
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
        var allGridLights = res[0];
        if(allGridLights / this.gridSizeArray[0] / this.gridSizeArray[1]>shaders.constants.AVG_LIGHTS_PER_CLUSTER){
            console.warn("AVG_LIGHTS_PER_CLUSTER cannot support all lights, screen may flicker")
        }
        if(allGridLights>this.lightIndicesArray.length){
            console.warn("Light Index cannot hold all light instances. Black tile may appear");
        }
        StagingBuffer.unmap();
        StagingBuffer.destroy();
        // console.log("Grid X "+ gridX);
        // console.log("Grid Y "+gridY);
    }

    BasePass(): void{
        const encoder = renderer.device.createCommandEncoder();
        // z prepass
        var desc:GPURenderPassDescriptor;
        if(this.usePackedGBuffer){
            desc = {
                label: "forward p base pass",
                colorAttachments: [
                    {
                        view: this.GBufferPackedView,
                        clearValue: [0, 0, 0, 0],
                        loadOp: "clear",
                        storeOp: "store"
                    },
                ],
                depthStencilAttachment: {
                    view: this.depthTextureView,
                    depthClearValue: 1.0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store"
                }
            };
        }
        else
        {
            desc = {
                label: "forward p base pass",
                colorAttachments: [
                    {
                        view: this.GBufferBaseColorView,
                        clearValue: [0, 0, 0, 0],
                        loadOp: "clear",
                        storeOp: "store"
                    },
                    {
                        view: this.GBufferNormalView,
                        clearValue: [0, 0, 0, 0],
                        loadOp: "clear",
                        storeOp: "store"
                    },
                    {
                        view: this.GBufferDepthView,
                        clearValue: [0, 0, 0, 0],
                        loadOp: "clear",
                        storeOp: "store"
                    },

                ],
                depthStencilAttachment: {
                    view: this.depthTextureView,
                    depthClearValue: 1.0,
                    depthLoadOp: "clear",
                    depthStoreOp: "store"
                }
            };
        }
        const renderPass = encoder.beginRenderPass(desc);
        if(this.usePackedGBuffer){
            renderPass.setPipeline(this.BasepassPackedGraphicsPipeline);
        }else{
            renderPass.setPipeline(this.BasepassGraphicsPipeline);
        }
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
        renderer.device.queue.writeBuffer(this.lightCountTotal, 0, this.lightCountTotalArray.buffer);

        this.BasePass();

        // light culling by tiles
        var res = this.LightCulling();
        
        res.then(()=>{
            // actual shading
            if (this.usePackedGBuffer){
                const encoder = renderer.device.createCommandEncoder();
                var gridX = Math.ceil(renderer.canvas.width/16);
                var gridY = Math.ceil(renderer.canvas.height/16);
                const computePass2 = encoder.beginComputePass();
                computePass2.setPipeline(this.deferredLightingPackedComputePipeline);
                computePass2.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
                computePass2.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
                this.gbuffersPackedBindGroup = renderer.device.createBindGroup({ // ... create bind group every frame since 
                    label:"GBuffer Packed Bind",
                    layout: this.gbuffersPackedBindGroupLayout,
                    entries:[
                        {
                            binding:0,
                            resource: this.GBufferPackedView
                        },
                        {
                            binding:1,
                            resource: renderer.context.getCurrentTexture().createView()
                        },
                    ]
                });
                computePass2.setBindGroup(shaders.constants.bindGroup_deferredLighting, this.gbuffersPackedBindGroup);
                computePass2.dispatchWorkgroups(
                    gridX, 
                    gridY, 
                    1
                );
                computePass2.end();
                // encoder.copyTextureToTexture(
                //     // details of the source texture
                //     { texture:this.FrameBuffer },
                    
                //     // details of the destination texture
                //    { texture: renderer.context.getCurrentTexture()},
                    
                //     // size:
                //     [ renderer.canvas.width, renderer.canvas.height, 1 ]
                // );
                renderer.device.queue.submit([encoder.finish()]);
            }
            else if(this.useCSPipeline){
                const encoder = renderer.device.createCommandEncoder();
                var gridX = Math.ceil(renderer.canvas.width/16);
                var gridY = Math.ceil(renderer.canvas.height/16);
                const computePass2 = encoder.beginComputePass();
                computePass2.setPipeline(this.deferredLightingComputePipeline);
                computePass2.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
                computePass2.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
                this.gbuffersBindGroup = renderer.device.createBindGroup({ // ... create bind group every frame since 
                    label:"GBuffer Bind",
                    layout: this.gbuffersBindGroupLayout,
                    entries:[
                        {
                            binding:0,
                            resource: this.GBufferBaseColorView
                        },
                        {
                            binding:1,
                            resource: this.GBufferNormalView
                        },
                        {
                            binding:2,
                            resource: this.GBufferDepthView
                        },
                        {
                            binding:3,
                            resource: renderer.context.getCurrentTexture().createView()
                        },
                    ]
                });
                computePass2.setBindGroup(shaders.constants.bindGroup_deferredLighting, this.gbuffersBindGroup);
                computePass2.dispatchWorkgroups(
                    gridX, 
                    gridY, 
                    1
                );
                computePass2.end();
                // encoder.copyTextureToTexture(
                //     // details of the source texture
                //     { texture:this.FrameBuffer },
                    
                //     // details of the destination texture
                //    { texture: renderer.context.getCurrentTexture()},
                    
                //     // size:
                //     [ renderer.canvas.width, renderer.canvas.height, 1 ]
                // );
                renderer.device.queue.submit([encoder.finish()]);
            }
            else{        
                const canvasTextureView = renderer.context.getCurrentTexture().createView();
                const encoder = renderer.device.createCommandEncoder();
                const renderPass = encoder.beginRenderPass({
                    label: "deferred shading pass",
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
                renderPass.setPipeline(this.deferredLightingGraphicsPipeline);
                renderPass.setBindGroup(shaders.constants.bindGroup_scene, this.sceneUniformsBindGroup);
                renderPass.setBindGroup(shaders.constants.bindGroup_lightCull, this.lightCullingBindGroup);
                renderPass.setBindGroup(shaders.constants.bindGroup_deferredLighting, this.gbuffersBindGroup);
                renderPass.setVertexBuffer(0, this.fullscreenTriangleBuffer);
                renderPass.draw(3, 1, 0, 0);
                renderPass.end();

                renderer.device.queue.submit([encoder.finish()]);
            }
        });
                
    }
}
