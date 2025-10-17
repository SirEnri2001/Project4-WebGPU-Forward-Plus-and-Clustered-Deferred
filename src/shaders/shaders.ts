// CHECKITOUT: this file loads all the shaders and preprocesses them with some common code

import { Camera } from '../stage/camera';

import commonRaw from './common.wgsl?raw';

import naiveVertRaw from './naive.vs.wgsl?raw';
import naiveFragRaw from './naive.fs.wgsl?raw';

import forwardPlusFragRaw from './forward_plus.fs.wgsl?raw';
import zPrepassFragRaw from './z_prepass.fs.wgsl?raw';
import forwardPlusCSRaw from './forward_plus.cs.wgsl?raw'

import clusteredDeferredFragRaw from './clustered_deferred.fs.wgsl?raw';
import clusteredDeferredFullscreenVertRaw from './clustered_deferred_fullscreen.vs.wgsl?raw';
import clusteredDeferredFullscreenFragRaw from './clustered_deferred_fullscreen.fs.wgsl?raw';

import moveLightsComputeRaw from './move_lights.cs.wgsl?raw';
import clusteringComputeRaw from './clustering.cs.wgsl?raw';

// CONSTANTS (for use in shaders)
// =================================

// CHECKITOUT: feel free to add more constants here and to refer to them in your shader code

// Note that these are declared in a somewhat roundabout way because otherwise minification will drop variables
// that are unused in host side code.
export const constants = {
    bindGroup_scene: 0,
    bindGroup_model: 1,
    bindGroup_lightCull: 2,
    bindGroup_material: 3,
    bindGroup_deferredLighting: 3,
    moveLightsWorkgroupSize: 128,

    lightRadius: 2,

    WORKGROUP_SIZE: 128,
    TILESIZE_X: 16,
    TILESIZE_Y: 16,
    LIGHTS_BATCH_SIZE: 64,
    DEPTH_INTEGER_SCALE: 20480,
    AVG_LIGHTS_PER_TILE: 256,
    MAX_LIGHTS_PER_TILE: 512,

    X_SLICES: 32,
    Y_SLICES: 32,
    Z_SLICES: 32,
    AVG_LIGHTS_PER_CLUSTER: 512,
    MAX_LIGHTS_PER_CLUSTER: 1024,
};

// =================================

function evalShaderRaw(raw: string) {
    return eval('`' + raw.replaceAll('${', '${constants.') + '`');
}

const commonSrc: string = evalShaderRaw(commonRaw);

function processShaderRaw(raw: string) {
    return commonSrc + evalShaderRaw(raw);
}

export const naiveVertSrc: string = processShaderRaw(naiveVertRaw);
export const naiveFragSrc: string = processShaderRaw(naiveFragRaw);

export const forwardPlusFragSrc: string = processShaderRaw(forwardPlusFragRaw);
export const zPrepassFragSrc: string = processShaderRaw(zPrepassFragRaw);
export const forwardPlusCSRawSrc: string = processShaderRaw(forwardPlusCSRaw);

export const clusteredDeferredFragSrc: string = processShaderRaw(clusteredDeferredFragRaw);
export const clusteredDeferredFullscreenVertSrc: string = processShaderRaw(clusteredDeferredFullscreenVertRaw);
export const clusteredDeferredFullscreenFragSrc: string = processShaderRaw(clusteredDeferredFullscreenFragRaw);

export const moveLightsComputeSrc: string = processShaderRaw(moveLightsComputeRaw);
export const clusteringComputeSrc: string = processShaderRaw(clusteringComputeRaw);
