# Rust Interface

该模块通过 C ABI 从 Rust 调用 `flash_attention_v0`。CUDA C++ 继续负责 kernel 计算，Rust 负责构建链接、显存所有权、shape/device 检查、CUDA stream 生命周期和错误传播。

Flash Attention kernel 的算法、分块和 online softmax 状态见 [Flash Attention README](../READ_FLASH_ATTENTION.md)。本文只讨论 Rust 与 CUDA C++ 之间的接口设计。

## 设计概览

Rust Interface 采用两层 crate：

```text
cuda-operator-sys  → 原始 C ABI 与 CUDA Runtime FFI
cuda-operator      → Tensor、CudaStream、Result 和参数检查
```

核心设计：

1. CUDA kernel 不使用 Rust 重写，由 NVCC 继续编译 `.cu` 文件。
2. C++ launch 函数通过 `extern "C"` 包装为稳定的跨语言调用边界。
3. `cuda-operator-sys` 集中保存裸指针和 `unsafe extern "C"` 声明。
4. `cuda-operator` 使用所有权、借用、RAII 和 `Result` 提供安全接口。
5. Cargo 构建脚本自动调用 NVCC 并链接 CUDA Runtime，不要求用户预先运行 CMake。

用户接口与 Python Extension 保持相近：

```rust
let output = flash_attention(&q, &k, &v, true)?;
```

Q、K、V 使用共享借用传入，函数返回一个拥有独立 CUDA 显存的 `Tensor`。

## 目录结构

```text
rust/
├── Cargo.toml                         # workspace
├── Cargo.lock
├── cuda-operator-sys/
│   ├── build.rs                       # NVCC 编译与 native linking
│   └── src/lib.rs                     # C ABI / CUDA Runtime 原始声明
└── cuda-operator/
    ├── src/lib.rs                     # 安全 Tensor、stream 和算子 API
    ├── examples/flash_attention.rs    # 最小调用示例
    └── tests/flash_attention.rs       # CPU reference 与 GPU 测试

../flash_attention_ffi.h               # C ABI 头文件
../flash_attention_ffi.cu              # C ABI 实现
../flash_attention_v0.cu               # CUDA kernel
```

## 整体调用链

```text
Rust 用户代码
flash_attention(&q, &k, &v, causal)
    ↓
cuda-operator
    ├── 检查 shape、device 和 kernel 限制
    ├── 在目标 GPU 上分配 output
    └── 提取 Q/K/V/output device pointer
    ↓
cuda-operator-sys
unsafe extern "C" declaration
    ↓
cuda_operator_flash_attention_f32(...)
    ├── C ABI 参数检查
    ├── launch_flash_attention_v0(...)
    └── cudaGetLastError()
    ↓
flash_attention_v0 CUDA kernel
    ↓
Result<Tensor>
```

安全 Rust 代码不直接解引用设备指针。裸指针只用于跨 FFI 传参，实际显存访问发生在 CUDA kernel 或 CUDA Runtime 中。

## 为什么保留 CUDA C++ kernel

该项目的 kernel 已使用 CUDA C++ 实现，并依赖 NVCC、CUDA 关键字和 CUDA intrinsics。Rust Interface 的目标是提供新的工程接入方式，而不是改变 kernel 实现语言。

因此职责划分为：

```text
CUDA C++：block/grid、shared memory、warp primitive、kernel launch
Rust：资源所有权、接口约束、错误处理、构建与上层调用
```

这种结构使独立 CUDA 程序、PyTorch Extension 和 Rust Interface 可以复用同一个 `launch_flash_attention_v0()`。

## 为什么需要 C ABI

普通 C++ 函数存在 name mangling，且 C++ `bool`、异常、类和模板不适合作为跨语言 ABI。Rust 与 CUDA C++ 之间只传递稳定的 C 类型：

| 参数 | ABI 类型 | 语义 |
|---|---|---|
| Q/K/V | `const float*` | 只读 CUDA device pointer |
| Output | `float*` | 可写 CUDA device pointer |
| B/H/N/D | `int32_t` | Kernel 维度 |
| Causal | `int32_t` | 0 表示 false，非 0 表示 true |
| Stream | `cudaStream_t` | Kernel 提交目标 stream |
| Return | `cudaError_t` | 参数或 launch error |

C ABI 函数使用：

```text
cuda_operator_flash_attention_f32(...)
```

命名包含算子名和 dtype，为以后增加 FP16/BF16 或其他算子保留区分方式。

C ABI 层再次检查空指针、正维度、`head_dim = 64` 和 `B * H <= 65535`。Rust 安全层已经执行同类检查，这里的重复检查用于保护其他可能直接调用 C ABI 的语言或程序。

## `cuda-operator-sys`：不安全边界

`cuda-operator-sys` 不提供业务抽象，只声明两类原始函数：

1. `cuda_operator_flash_attention_f32` C ABI
2. 当前安全层需要的 CUDA Runtime API

CUDA Runtime 声明包括：

```text
device     : cudaGetDevice / cudaSetDevice
memory     : cudaMalloc / cudaFree / cudaMemset / cudaMemcpy
sync       : cudaDeviceSynchronize
stream     : cudaStreamCreate / cudaStreamDestroy / cudaStreamSynchronize
diagnostic : cudaGetErrorString
```

这些函数位于 `unsafe extern "C"` block 中。编译器只能检查 Rust 侧声明的类型，不能证明以下条件：

- 指针是否来自正确的 CUDA device
- 指针指向的空间是否足够
- 内存是否仍然有效
- stream 是否属于正确设备
- C/CUDA 侧函数签名是否与声明一致

因此 `-sys` crate 只描述 ABI，不宣称调用安全。上述约束由 `cuda-operator` 的安全层维护。

## `cuda-operator`：安全接口

安全 crate 对外提供：

```text
Tensor
CudaStream
Error / Result
flash_attention
flash_attention_on_stream
```

其目标是让普通调用不需要编写 `unsafe`、`cudaMalloc`、`cudaFree` 或裸 FFI。

## Tensor 抽象

当前 `Tensor` 是针对该 kernel 的最小 CUDA Tensor，不是通用张量框架。

```text
Tensor
├── NonNull<f32> device pointer
├── [usize; 4] shape
├── element count
└── CUDA device index
```

它维护以下不变量：

- 指针非空
- 存储位于 CUDA device memory
- dtype 固定为 FP32
- shape 固定为四维 `[B, H, N, D]`
- 每个维度大于 0
- 元素数量等于四个维度的乘积
- 元素数量和字节数计算不能溢出

### 从 host 创建

```rust
let q = Tensor::from_slice(&q_host, [1, 8, 257, 64])?;
```

调用过程：

```text
验证 host slice length
    ↓
读取 current CUDA device
    ↓
cudaMalloc
    ↓
cudaMemcpy HostToDevice
    ↓
返回拥有该 device pointer 的 Tensor
```

### 创建零 Tensor

```rust
let tensor = Tensor::zeros([1, 1, 32, 64])?;
```

该接口执行 `cudaMalloc` 和 `cudaMemset`。Flash Attention 的 output 不使用 zero initialization，因为 kernel 会写入全部有效输出元素。

### 复制回 host

```rust
let output_host = output.to_vec()?;
```

当前实现先调用 `cudaDeviceSynchronize()`，然后执行 DeviceToHost `cudaMemcpy`。这样可以等待 default stream 和显式 stream 中可能生成该 Tensor 的 kernel，但同步粒度是整个设备。

## 所有权与 RAII

`Tensor` 拥有 CUDA pointer，且没有实现复制语义。变量离开作用域时，`Drop` 在 Tensor 所属 device 上调用 `cudaFree`：

```text
Tensor::from_slice / Tensor::zeros
    ↓
Tensor 获得显存所有权
    ↓
在作用域内借用 Tensor
    ↓
Tensor Drop
    ↓
cudaFree
```

`flash_attention` 接收 `&Tensor`：

```rust
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    causal: bool,
) -> Result<Tensor>
```

共享借用表达 Q、K、V 只读；函数不会取得或转移输入的所有权。输出是新建的 `Tensor`，其所有权交给调用者。

相比直接暴露 `cudaMalloc`/`cudaFree`，该设计减少以下错误：

- 正常返回路径忘记释放显存
- 提前释放仍由 Rust 对象持有的显存
- 多个普通 Rust 对象重复拥有同一 device pointer
- 将可变输出指针暴露为普通安全接口

析构函数不能返回 `Result`，因此 `Drop` 中的 CUDA 错误不会向调用者传播；需要观测异步执行错误时，应显式调用 `synchronize()` 或 `to_vec()`。

## DeviceGuard

CUDA Runtime 的 current device 属于线程执行上下文。Tensor 可能在 GPU 1 上创建，但后续调用发生时 current device 已切换到 GPU 0。

内部 `DeviceGuard` 执行：

```text
cudaGetDevice() 保存原 device
    ↓
必要时 cudaSetDevice(target)
    ↓
执行 allocation / copy / launch / free
    ↓
Drop 时恢复原 device
```

这与 PyTorch Extension 中的 `c10::cuda::CUDAGuard` 作用相同，但 Rust Interface 不依赖 PyTorch，因此使用 CUDA Runtime 自行实现。

## 参数检查

`flash_attention` 在进入 FFI 前检查：

| 检查项 | 约束 |
|---|---|
| Shape | Q、K、V 的 `[B, H, N, D]` 完全一致 |
| Device | Q、K、V 位于同一 CUDA device |
| Stream device | 显式 stream 与 Tensor 位于同一 device |
| Head dimension | `D = 64` |
| Grid limit | `B * H <= 65535` |
| Integer conversion | B、H、N、D 可以转换为 kernel 使用的 `i32` |
| Allocation size | 元素数量与字节数计算不溢出 |

安全层检查用于生成包含具体 shape/device 的 Rust 错误；C ABI 检查作为第二层防御。

## CUDA Stream

### Default stream

```rust
let output = flash_attention(&q, &k, &v, false)?;
```

该接口向 C ABI 传入空 stream handle，对应 CUDA default stream。

### 显式 stream

```rust
let stream = CudaStream::new()?;
let output = flash_attention_on_stream(&q, &k, &v, false, &stream)?;
stream.synchronize()?;
```

`CudaStream` 记录 raw stream handle 和创建时的 CUDA device。创建时调用 `cudaStreamCreate`，离开作用域时在正确 device 上调用 `cudaStreamDestroy`。

Kernel launch 保持异步：

```text
flash_attention_on_stream 返回
    ≠
GPU 已经完成计算
```

需要在 host 读取结果时，可以显式 `stream.synchronize()`，或使用当前实现的 `output.to_vec()` 进行设备同步后复制。

## 错误模型

安全 crate 定义两类错误：

```text
InvalidArgument(String)
Cuda { operation, code, message }
```

`InvalidArgument` 用于 shape、device、长度或 kernel 约束错误。`Cuda` 保存：

- 失败的 CUDA 操作名称
- 原始 `cudaError_t` 数值
- `cudaGetErrorString` 返回的错误文本

因此所有公开的可失败操作返回：

```rust
Result<T, Error>
```

调用者可以使用 `?` 向上传播，而不是检查每个 CUDA 整数返回码。

C ABI 在 launch 后调用 `cudaGetLastError()`，用于捕获本次 kernel launch 的立即错误。异步执行阶段的错误会在后续 stream/device synchronization 或 CUDA Runtime 调用时出现。

## Cargo 与 NVCC 构建链

Rust 编译器不编译 `.cu` 文件，因此 `cuda-operator-sys/build.rs` 在 Cargo 构建期间调用 NVCC：

```text
cargo build
    ↓
build.rs
    ├── nvcc 编译 flash_attention_v0.cu
    ├── nvcc 编译 flash_attention_ffi.cu
    ├── nvcc --lib 生成静态库
    └── 输出 native link 配置
            ├── cuda_operator_flash_attention
            ├── cudart
            └── stdc++
    ↓
rustc 编译两个 Rust crate
    ↓
链接最终 executable / test binary
```

Build script 的架构选择顺序：

- `CUDA_ARCH` 指定目标架构，默认 `86`
- `CUDA_HOME` 或 `CUDA_PATH` 指定 CUDA Toolkit
- `NVCC` 可以指定 NVCC 路径

示例：

```bash
CUDA_ARCH=89 cargo build --release
NVCC=/usr/local/cuda/bin/nvcc cargo build --release
```

Build script 使用 `rerun-if-changed` 跟踪 kernel、C ABI 和相关头文件；这些文件变化时 Cargo 会重新编译 native library。

仓库的 CMake 同时提供 `cuda_operator_flash_attention` 静态库 target，供 CMake 工程使用。Cargo 构建不依赖已有的 `build/` 目录，而是独立生成自己的静态库。

## 构建与运行

默认目标架构与项目验证设备一致，为 `sm_86`：

```bash
cd flash_attention/rust
cargo build --release
cargo run --release --example flash_attention
cargo test --release
```

其他 GPU 架构通过 `CUDA_ARCH` 指定：

```bash
CUDA_ARCH=89 cargo test --release
```

验证环境为 Rust 1.96、CUDA Toolkit 12.4 和 RTX 3060 Laptop GPU（`sm_86`）。该 workspace 不依赖第三方 Rust crate。

## 使用

```rust
use cuda_operator::{Result, Tensor, flash_attention};

fn run(q_host: &[f32], k_host: &[f32], v_host: &[f32]) -> Result<Vec<f32>> {
    let shape = [1, 8, 257, 64];

    let q = Tensor::from_slice(q_host, shape)?;
    let k = Tensor::from_slice(k_host, shape)?;
    let v = Tensor::from_slice(v_host, shape)?;

    let output = flash_attention(&q, &k, &v, true)?;
    output.to_vec()
}
```

对应的数据流：

```text
Vec<f32> / &[f32]
    ↓ HostToDevice
Tensor on CUDA
    ↓ Flash Attention
Tensor on CUDA
    ↓ DeviceToHost
Vec<f32>
```

如果多个算子连续执行，可以将中间结果保留为 `Tensor`，避免在每一步都复制回 host。当前项目只有 Flash Attention 接受该 Tensor；后续算子接入同一安全层后可以形成全 device 数据流。

## 正确性测试

```bash
cargo test --release
```

Rust 测试实现了独立 CPU attention reference，没有依赖 PyTorch。Reference 显式计算：

```text
scores = QK^T / sqrt(D)
probability = softmax(scores + causal_mask)
output = probability V
```

逐元素容差为：

```text
tolerance = 2e-4 + 2e-4 * abs(expected)
```

四项测试：

| 测试 | 覆盖内容 |
|---|---|
| non-causal | `[1, 2, 17, 64]` 与 CPU reference 对比 |
| causal | 不规则 `N=17` 的 causal mask 与 CPU reference 对比 |
| invalid head dimension | `[1, 1, 2, 32]` 在 Rust 安全层返回错误 |
| non-default stream | 显式创建 stream、提交 kernel、同步并与 reference 对比 |

测试使用不规则序列长度，使 kernel 同时经过完整 tile 和边界 tile。

## 与 PyTorch Extension 的区别

| 维度 | PyTorch Extension | Rust Interface |
|---|---|---|
| Tensor | `at::Tensor` / `torch.Tensor` | 项目内最小 FP32 `Tensor` |
| 显存分配 | PyTorch allocator | `cudaMalloc` / `cudaFree` |
| Device guard | `c10::cuda::CUDAGuard` | 自定义 RAII `DeviceGuard` |
| 默认 stream | PyTorch current stream | CUDA default stream |
| 显式 stream | `torch.cuda.Stream` 上下文 | `CudaStream` 参数 |
| 错误 | `TORCH_CHECK` / PyTorch exception | `Result<T, Error>` |
| 算子注册 | PyTorch dispatcher | 普通 Rust 函数 |
| Autograd | 当前拒绝 | 不包含 autograd 概念 |
| Native 边界 | ATen C++ API | C ABI + raw FFI |

两者共享同一个 CUDA kernel，但上层 runtime 不同：PyTorch 版接入现有 Tensor runtime，Rust 版自行管理最小运行时资源。

## 设计取舍

### 为什么拆成 `-sys` 和安全 crate

`-sys` 层与 C ABI 一一对应，便于审计签名和链接问题；安全层维护 shape、device 和生命周期约束。新增算子时，底层绑定和用户 API 不会混在同一层传播 `unsafe`。

### 为什么不使用 bindgen

当前 C ABI 和 CUDA Runtime 函数数量较少，手写声明可以保持 workspace 无第三方依赖，并明确展示实际使用的 native surface。接口扩大后，可以再使用 bindgen 自动同步头文件。

### 为什么不把 host 数据直接传给 kernel

Kernel 接收 device pointer。由 `Tensor::from_slice` 显式完成 HostToDevice copy，可以让多个 kernel 复用同一份设备数据，并使数据位置在类型与 API 中可见。

### 为什么 `to_vec()` 使用 device synchronization

Tensor 可能由 default stream 或显式 stream 生成，但 Tensor 当前没有记录 producer stream/event。设备同步可以覆盖两种来源，代价是阻塞该设备上的其他工作。更细粒度的实现需要为 Tensor 记录事件或 stream 依赖。

## 当前限制

- Tensor 只支持连续 FP32 `[B, H, N, D]`，不是通用 ndarray/Tensor。
- Flash Attention 固定 `head_dim = 64`，只实现 forward。
- 不支持 FP16、BF16、dropout、backward 和 packed variable-length sequence。
- `Tensor::from_slice` 和 `to_vec` 使用同步 host/device copy，没有 pinned memory 或 async copy。
- `to_vec()` 使用 device-wide synchronization，没有 producer event 跟踪。
- 没有从外部 raw device pointer 安全接管 Tensor 的接口。
- `Drop` 不能传播 `cudaFree` 或 `cudaStreamDestroy` 错误。
- 未实现多 GPU 通信；一次算子的 Tensor 与 stream 必须位于同一设备。
