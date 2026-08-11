#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_void};

pub type cudaError_t = c_int;
pub type cudaStream_t = *mut c_void;

pub const CUDA_SUCCESS: cudaError_t = 0;
pub const CUDA_MEMCPY_HOST_TO_DEVICE: c_int = 1;
pub const CUDA_MEMCPY_DEVICE_TO_HOST: c_int = 2;

unsafe extern "C" {
    pub fn cuda_operator_flash_attention_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        output: *mut f32,
        batch_size: i32,
        head_num: i32,
        seq_len: i32,
        head_dim: i32,
        causal: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn cudaGetErrorString(error: cudaError_t) -> *const c_char;
    pub fn cudaGetDevice(device: *mut c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: c_int) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    pub fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> cudaError_t;
    pub fn cudaFree(ptr: *mut c_void) -> cudaError_t;
    pub fn cudaMemset(ptr: *mut c_void, value: c_int, count: usize) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut c_void,
        src: *const c_void,
        count: usize,
        kind: c_int,
    ) -> cudaError_t;
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
}
