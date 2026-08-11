//! Safe Rust interface for the CUDA Operator Flash Attention kernel.

use std::error::Error as StdError;
use std::ffi::{CStr, c_void};
use std::fmt;
use std::ptr::{self, NonNull};

use cuda_operator_sys as sys;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    InvalidArgument(String),
    Cuda {
        operation: &'static str,
        code: i32,
        message: String,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(message) => formatter.write_str(message),
            Self::Cuda {
                operation,
                code,
                message,
            } => write!(
                formatter,
                "{operation} failed with CUDA error {code}: {message}"
            ),
        }
    }
}

impl StdError for Error {}

fn cuda_error(operation: &'static str, code: i32) -> Error {
    let message = unsafe {
        let pointer = sys::cudaGetErrorString(code);
        if pointer.is_null() {
            "unknown CUDA error".to_owned()
        } else {
            CStr::from_ptr(pointer).to_string_lossy().into_owned()
        }
    };
    Error::Cuda {
        operation,
        code,
        message,
    }
}

fn check_cuda(code: i32, operation: &'static str) -> Result<()> {
    if code == sys::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(cuda_error(operation, code))
    }
}

fn element_count(shape: [usize; 4]) -> Result<usize> {
    if shape.contains(&0) {
        return Err(Error::InvalidArgument(format!(
            "tensor dimensions must be greater than zero, got {shape:?}"
        )));
    }

    shape.into_iter().try_fold(1usize, |count, dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| Error::InvalidArgument(format!("tensor shape is too large: {shape:?}")))
    })
}

fn current_device() -> Result<i32> {
    let mut device = 0;
    check_cuda(unsafe { sys::cudaGetDevice(&mut device) }, "cudaGetDevice")?;
    Ok(device)
}

struct DeviceGuard {
    previous: i32,
    changed: bool,
}

impl DeviceGuard {
    fn new(device: i32) -> Result<Self> {
        let previous = current_device()?;
        let changed = previous != device;
        if changed {
            check_cuda(unsafe { sys::cudaSetDevice(device) }, "cudaSetDevice")?;
        }
        Ok(Self { previous, changed })
    }
}

impl Drop for DeviceGuard {
    fn drop(&mut self) {
        if self.changed {
            unsafe {
                sys::cudaSetDevice(self.previous);
            }
        }
    }
}

/// A contiguous FP32 tensor in CUDA device memory with `[B, H, N, D]` layout.
pub struct Tensor {
    pointer: NonNull<f32>,
    shape: [usize; 4],
    len: usize,
    device: i32,
}

impl Tensor {
    /// Copies a host slice into a tensor on the current CUDA device.
    pub fn from_slice(values: &[f32], shape: [usize; 4]) -> Result<Self> {
        let expected = element_count(shape)?;
        if values.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "data length does not match shape {shape:?}: expected {expected}, got {}",
                values.len()
            )));
        }

        let tensor = Self::allocate(shape, current_device()?, false)?;
        check_cuda(
            unsafe {
                sys::cudaMemcpy(
                    tensor.pointer.as_ptr().cast(),
                    values.as_ptr().cast(),
                    tensor.byte_len(),
                    sys::CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "cudaMemcpy host to device",
        )?;
        Ok(tensor)
    }

    /// Allocates a zero-initialized tensor on the current CUDA device.
    pub fn zeros(shape: [usize; 4]) -> Result<Self> {
        Self::allocate(shape, current_device()?, true)
    }

    fn allocate(shape: [usize; 4], device: i32, zeroed: bool) -> Result<Self> {
        let len = element_count(shape)?;
        let byte_len = len.checked_mul(size_of::<f32>()).ok_or_else(|| {
            Error::InvalidArgument(format!("tensor shape is too large: {shape:?}"))
        })?;
        let _guard = DeviceGuard::new(device)?;
        let mut raw = ptr::null_mut::<c_void>();
        check_cuda(unsafe { sys::cudaMalloc(&mut raw, byte_len) }, "cudaMalloc")?;
        let pointer = NonNull::new(raw.cast::<f32>()).ok_or_else(|| {
            Error::InvalidArgument("cudaMalloc returned a null pointer".to_owned())
        })?;

        let tensor = Self {
            pointer,
            shape,
            len,
            device,
        };
        if zeroed {
            check_cuda(unsafe { sys::cudaMemset(raw, 0, byte_len) }, "cudaMemset")?;
        }
        Ok(tensor)
    }

    pub fn shape(&self) -> [usize; 4] {
        self.shape
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn as_device_ptr(&self) -> *const f32 {
        self.pointer.as_ptr()
    }

    fn as_mut_device_ptr(&mut self) -> *mut f32 {
        self.pointer.as_ptr()
    }

    fn byte_len(&self) -> usize {
        self.len * size_of::<f32>()
    }

    /// Waits for device work and copies the tensor back to host memory.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        let _guard = DeviceGuard::new(self.device)?;
        check_cuda(
            unsafe { sys::cudaDeviceSynchronize() },
            "cudaDeviceSynchronize",
        )?;

        let mut values = vec![0.0; self.len];
        check_cuda(
            unsafe {
                sys::cudaMemcpy(
                    values.as_mut_ptr().cast(),
                    self.pointer.as_ptr().cast(),
                    self.byte_len(),
                    sys::CUDA_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "cudaMemcpy device to host",
        )?;
        Ok(values)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("len", &self.len)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        let Ok(_guard) = DeviceGuard::new(self.device) else {
            return;
        };
        unsafe {
            sys::cudaFree(self.pointer.as_ptr().cast());
        }
    }
}

/// An owned CUDA stream on the device that was current when it was created.
pub struct CudaStream {
    raw: sys::cudaStream_t,
    device: i32,
}

impl CudaStream {
    pub fn new() -> Result<Self> {
        let device = current_device()?;
        let mut raw = ptr::null_mut();
        check_cuda(
            unsafe { sys::cudaStreamCreate(&mut raw) },
            "cudaStreamCreate",
        )?;
        Ok(Self { raw, device })
    }

    pub fn device(&self) -> i32 {
        self.device
    }

    pub fn synchronize(&self) -> Result<()> {
        let _guard = DeviceGuard::new(self.device)?;
        check_cuda(
            unsafe { sys::cudaStreamSynchronize(self.raw) },
            "cudaStreamSynchronize",
        )
    }
}

impl fmt::Debug for CudaStream {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaStream")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        let Ok(_guard) = DeviceGuard::new(self.device) else {
            return;
        };
        unsafe {
            sys::cudaStreamDestroy(self.raw);
        }
    }
}

/// Runs FP32 Flash Attention on the CUDA default stream.
///
/// Inputs must be contiguous tensors with the same `[B, H, N, 64]` shape.
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor, causal: bool) -> Result<Tensor> {
    launch_flash_attention(q, k, v, causal, ptr::null_mut(), q.device)
}

/// Runs FP32 Flash Attention on an explicitly owned CUDA stream.
pub fn flash_attention_on_stream(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    stream: &CudaStream,
) -> Result<Tensor> {
    launch_flash_attention(q, k, v, causal, stream.raw, stream.device)
}

fn launch_flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
    stream: sys::cudaStream_t,
    stream_device: i32,
) -> Result<Tensor> {
    if q.shape != k.shape || q.shape != v.shape {
        return Err(Error::InvalidArgument(format!(
            "q, k and v must have the same shape, got q={:?}, k={:?}, v={:?}",
            q.shape, k.shape, v.shape
        )));
    }
    if q.device != k.device || q.device != v.device {
        return Err(Error::InvalidArgument(format!(
            "q, k and v must be on the same CUDA device, got q={}, k={}, v={}",
            q.device, k.device, v.device
        )));
    }
    if q.device != stream_device {
        return Err(Error::InvalidArgument(format!(
            "tensor device {} does not match stream device {stream_device}",
            q.device
        )));
    }

    let [batch_size, head_num, seq_len, head_dim] = q.shape;
    if head_dim != 64 {
        return Err(Error::InvalidArgument(format!(
            "flash_attention currently requires head_dim=64, got {head_dim}"
        )));
    }
    let batch_heads = batch_size.checked_mul(head_num).ok_or_else(|| {
        Error::InvalidArgument("batch size multiplied by head count overflowed".to_owned())
    })?;
    if batch_heads > 65535 {
        return Err(Error::InvalidArgument(format!(
            "batch_size * head_num must not exceed 65535, got {batch_heads}"
        )));
    }

    let to_i32 = |dimension| {
        i32::try_from(dimension).map_err(|_| {
            Error::InvalidArgument(format!(
                "tensor dimensions exceed the range supported by the kernel: {:?}",
                q.shape
            ))
        })
    };
    let batch_size = to_i32(batch_size)?;
    let head_num = to_i32(head_num)?;
    let seq_len = to_i32(seq_len)?;
    let head_dim = to_i32(head_dim)?;

    let _guard = DeviceGuard::new(q.device)?;
    let mut output = Tensor::allocate(q.shape, q.device, false)?;
    check_cuda(
        unsafe {
            sys::cuda_operator_flash_attention_f32(
                q.as_device_ptr(),
                k.as_device_ptr(),
                v.as_device_ptr(),
                output.as_mut_device_ptr(),
                batch_size,
                head_num,
                seq_len,
                head_dim,
                i32::from(causal),
                stream,
            )
        },
        "cuda_operator_flash_attention_f32",
    )?;
    Ok(output)
}
