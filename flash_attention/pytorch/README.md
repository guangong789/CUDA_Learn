# PyTorch C++/CUDA Extension

本模块将 `flash_attention_v0` 接入 PyTorch dispatcher，使 CUDA kernel 可以接收和返回 `torch.Tensor`。Kernel 的计算逻辑保持在 `flash_attention_v0.cu` 中，Extension 负责构建、算子注册、参数检查、设备切换、输出分配和 CUDA stream 传递。

Flash Attention kernel 的分块、shared-memory 布局和 online softmax 合并公式见 [Flash Attention README](../READ_FLASH_ATTENTION.md)。本文只讨论 PyTorch 接入层。

## 设计概览

该 Extension 解决五个问题：

1. 使用 `CUDAExtension` 将 C++ binding 和 CUDA kernel 编译为 Python 可加载的动态库。
2. 使用 `TORCH_LIBRARY` 定义算子 schema，通过 dispatcher 分别注册 CPU 和 CUDA 实现。
3. 使用 `at::Tensor` 和 PyTorch allocator 管理输入、输出及显存生命周期。
4. 使用 `CUDAGuard` 和 PyTorch current CUDA stream 保持 device/stream 语义。
5. 使用 fake implementation、正确性测试、`opcheck` 和 benchmark 验证框架接入。

调用接口与普通 PyTorch 函数一致：

```python
output = flash_attention(q, k, v, causal=True)
```

输入和输出均为连续的 FP32 CUDA Tensor，布局为 `[B, H, N, D]`，当前固定要求 `D = 64`。

## 目录结构

```text
pytorch/
├── setup.py                         # CUDA Extension 构建配置
├── cuda_operator/
│   └── __init__.py                  # Python API 与 fake implementation
├── csrc/
│   └── flash_attention_binding.cpp  # dispatcher 注册与 CUDA 调用
├── test_flash_attention.py          # 正确性、stream、错误处理和 opcheck
└── benchmark.py                     # CUDA Event benchmark
```

CUDA 源码不在该目录复制一份，`setup.py` 直接编译上一级的 `flash_attention_v0.cu`。独立 CUDA 程序、PyTorch Extension 和 Rust Interface 因此调用同一个 kernel 实现。

## 调用链

```text
Python
cuda_operator.flash_attention(q, k, v, causal)
    ↓
torch.ops.cuda_operator.flash_attention(...)
    ↓
PyTorch dispatcher 根据 Tensor device 选择实现
    ├── CPU  → flash_attention_cpu()  → 抛出 CUDA-only 错误
    └── CUDA → flash_attention_cuda()
                  ↓
              参数与布局检查
                  ↓
              CUDAGuard 切换设备
                  ↓
              at::empty_like(q) 分配输出
                  ↓
              获取 PyTorch current CUDA stream
                  ↓
              launch_flash_attention_v0(...)
                  ↓
              C10_CUDA_KERNEL_LAUNCH_CHECK()
                  ↓
              返回 output Tensor
```

这条调用链中，Python 层不持有裸 CUDA 指针。只有 C++ binding 从 `at::Tensor` 中提取 `data_ptr<float>()`，并将设备指针传递给 kernel launch 函数。

## 构建层：CUDAExtension

`setup.py` 使用 `torch.utils.cpp_extension.CUDAExtension` 编译两个源文件：

```text
flash_attention_binding.cpp
flash_attention_v0.cu
```

C++ 部分使用 `-O3`，NVCC 部分使用 `-O3 -lineinfo`。`-lineinfo` 保留 kernel 行号信息，便于 Nsight Compute 或 Nsight Systems 将性能数据映射回 CUDA 源码。

构建产物为 `cuda_operator._C`。Python 初始化时：

```python
from . import _C
```

导入 `_C` 的主要作用不是调用 pybind 暴露的函数，而是加载动态库并执行其中的静态注册代码。`TORCH_LIBRARY` 和 `TORCH_LIBRARY_IMPL` 在动态库加载时将算子加入 PyTorch dispatcher。

## 算子注册：schema 与 dispatch

算子 schema 定义为：

```text
flash_attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor
```

注册过程分为三部分。

### Schema

`TORCH_LIBRARY(cuda_operator, m)` 声明算子名称、参数和返回值。注册后，算子具有统一入口：

```python
torch.ops.cuda_operator.flash_attention(q, k, v, causal)
```

### CPU dispatch

CPU dispatch 没有执行 kernel，而是返回明确的 CUDA-only 错误。单独注册 CPU 实现可以避免 dispatcher 只报告“找不到对应 backend”，使非法设备输入的错误信息包含修复方向。

### CUDA dispatch

CUDA dispatch 指向 `flash_attention_cuda()`。该函数不实现 Attention 数学，而是完成框架与 kernel 之间的适配：

```text
Tensor metadata → kernel dimensions
Tensor storage  → float device pointer
PyTorch device  → CUDA device guard
PyTorch stream  → cudaStream_t
CUDA error      → PyTorch RuntimeError
```

`PYBIND11_MODULE` 保留一个空模块，使编译产物可以作为 Python Extension 被导入；实际用户接口通过 dispatcher 而不是 pybind 函数提供。

## 参数检查

参数检查位于 C++ CUDA dispatch 中，发生在输出分配和 kernel launch 之前。

| 检查项 | 约束 | 原因 |
|---|---|---|
| Device | Q、K、V 均为 CUDA Tensor | Kernel 只接收 device pointer |
| Device 一致性 | Q、K、V 位于同一 GPU | 单次 kernel launch 不能直接读取其他设备的普通指针 |
| Dtype | 全部为 `torch.float32` | 当前 kernel 只实现 FP32 |
| Rank | 全部为 4 维 | 固定解释为 `[B, H, N, D]` |
| Shape | Q、K、V shape 完全一致 | 当前 kernel 只支持等长 Q/K/V |
| Layout | 全部 contiguous | Kernel 按连续 `[B, H, N, D]` 地址计算 |
| Dimension | B、H、N 大于 0，D 等于 64 | 对应 kernel 的 grid 和模板约束 |
| Grid | `B * H <= 65535` | `batch * head` 映射到 `grid.y` |
| Integer range | B、H、N 不超过 `INT_MAX` | Kernel 参数使用 `int` |
| Autograd | 输入不能 `requires_grad` | 当前只注册 forward，没有 backward |

这些检查构成框架接口的运行时契约。Kernel 内部仍保留 `head_dim` 检查，但主要错误应在进入 kernel 前由 binding 报告。

## Device 管理

PyTorch 程序可能同时使用多块 GPU，而且调用 Extension 时的 current device 不一定等于输入 Tensor 所在设备。

Binding 根据 `q.device()` 创建 `c10::cuda::CUDAGuard`：

```text
记录调用前的 current device
    ↓
切换到 q 所在 device
    ↓
分配输出并 launch kernel
    ↓
函数结束时恢复原 device
```

输出使用 `at::empty_like(q)` 分配，因此继承 Q 的：

- shape
- dtype
- device
- contiguous memory format

输出显存由 PyTorch allocator 管理，不需要 Extension 手工调用 `cudaMalloc` 或 `cudaFree`。

## CUDA Stream 语义

Extension 不使用写死的 default stream，而是读取输入设备上的 PyTorch current CUDA stream：

```text
torch.cuda.stream(stream) 上下文
    ↓
PyTorch current stream
    ↓
c10::cuda::getCurrentCUDAStream(...)
    ↓
cudaStream_t
    ↓
launch_flash_attention_v0(..., stream)
```

因此算子能够加入调用者当前 stream 的执行顺序，不需要在 binding 中调用 `cudaDeviceSynchronize()`。Kernel launch 保持异步，后续 PyTorch CUDA 算子可以继续在同一 stream 上消费 output。

测试中的 `test_non_default_stream` 会创建 `torch.cuda.Stream()`，在该 stream 上创建输入并调用算子，然后同步该 stream 并与 reference 对比。该测试用于检测错误地使用 default stream 的实现。

## Kernel launch 错误

Kernel launch 后调用：

```text
C10_CUDA_KERNEL_LAUNCH_CHECK()
```

它将 CUDA launch error 转换为 PyTorch 异常，例如非法 grid、非法 launch 配置等。由于 CUDA 执行是异步的，运行阶段错误可能在后续同步或其他 CUDA API 调用时出现；测试和 benchmark 会在结果比较或计时阶段产生同步。

## Python 包装与 fake implementation

Python 用户接口只做一层名称包装：

```python
def flash_attention(q, k, v, causal=False):
    return torch.ops.cuda_operator.flash_attention(q, k, v, causal)
```

计算和运行时检查仍由 dispatcher 与 C++ 实现完成。`__all__` 只导出 `flash_attention`，不直接暴露 `_C`。

模块还通过 `torch.library.register_fake` 注册 fake implementation。FakeTensor 不包含真实数据，fake implementation 只根据 metadata 检查 rank、shape、dtype 和 `head_dim`，然后返回 `torch.empty_like(q)` 形式的输出描述。

其作用包括：

- 为 `torch.library.opcheck` 提供 schema/fake-kernel 检查路径
- 在不执行真实 CUDA kernel 时完成输出 shape 和 dtype 推导
- 为依赖 FakeTensor 的 tracing/compile 工具提供算子元数据

运行时 C++ 检查仍是实际执行时的最终约束；fake implementation 不替代 CUDA dispatch。

## 构建与安装

首先确认当前 PyTorch 包含 CUDA 支持：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

在本目录执行 editable install：

```bash
cd flash_attention/pytorch
python -m pip install -e . --no-build-isolation
```

PyTorch 通常根据当前可见 GPU 选择编译架构。构建环境无法访问 GPU 时，可以显式设置，例如 RTX 30 系列：

```bash
TORCH_CUDA_ARCH_LIST=8.6 python -m pip install -e . --no-build-isolation
```

项目验证环境为 Python 3.10、PyTorch `2.6.0+cu124`、CUDA Toolkit 12.4 和 RTX 3060 Laptop GPU（`sm_86`）。

## 使用

```python
import torch
from cuda_operator import flash_attention

q = torch.randn((1, 8, 257, 64), device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)

with torch.inference_mode():
    output = flash_attention(q, k, v, causal=True)

print(output.shape)   # torch.Size([1, 8, 257, 64])
print(output.device)  # cuda:0
print(output.dtype)   # torch.float32
```

当前算子拒绝 `requires_grad=True` 的输入。推理代码可以使用 `torch.inference_mode()`，也可以显式传入 detached Tensor。

## 正确性测试

```bash
python test_flash_attention.py
```

测试固定随机种子为 2026，并将 Extension 输出同时与两个 reference 对比：

```text
显式 reference = softmax(QK^T / sqrt(D) + mask) V
PyTorch reference = scaled_dot_product_attention(...)
```

误差阈值为：

```text
atol = 3e-4
rtol = 3e-3
```

测试内容：

| 测试 | 覆盖内容 |
|---|---|
| regular and irregular shapes | `N=32/127/256/257/1000`，causal 与 non-causal |
| non-default stream | current stream 是否正确传给 kernel |
| operator registration | `torch.library.opcheck` 检查 schema、dispatch 和 fake implementation |
| wrong dtype | 拒绝 FP16 输入 |
| CPU input | CPU dispatch 返回 CUDA-only 错误 |
| non-contiguous input | 拒绝不符合地址计算方式的 layout |
| wrong head dimension | 拒绝 `D != 64` |
| shape mismatch | 拒绝 Q/K/V shape 不一致 |
| autograd input | 拒绝需要 backward 的输入 |

## Benchmark

```bash
python benchmark.py
python benchmark.py --causal
python benchmark.py --batch-size 1 --heads 8 --seq-len 257
```

Benchmark 的步骤为：

1. 创建固定 shape 的 FP32 CUDA Tensor。
2. 在 `torch.inference_mode()` 中分别运行自定义算子和 PyTorch SDPA。
3. 执行 warmup，避免首次加载和初始化影响结果。
4. 使用 CUDA Event 记录 GPU stream 上的执行时间。
5. 对多次迭代取平均时间。

TFLOPS 只计算两个矩阵乘法 `QK^T` 和 `PV` 的主要 FLOPs：

```text
non-causal score_count = N * N
causal score_count     = N * (N + 1) / 2
FLOPs                  = 4 * B * H * score_count * D
```

Softmax、mask 和指数运算没有计入该公式，因此该数值用于同一脚本内的相对比较。

在验证设备上使用 `B=4, H=8, N=1024, D=64` 的记录如下：

| 实现 | non-causal | causal |
|---|---:|---:|
| `cuda_operator` | 7.41 ms | 4.53 ms |
| PyTorch SDPA | 2.61 ms | 1.43 ms |

PyTorch SDPA 可以根据硬件和输入选择不同 backend。表中数据只对应记录时的软件、硬件和 benchmark 参数。

## 设计取舍

### 为什么使用 dispatcher，而不是只写 pybind 函数

Dispatcher 能根据 device/backend 选择实现，并使算子具有 schema、fake implementation 和 `torch.ops` 身份。Pybind 在这里仅负责让动态库可以被 Python 导入。

### 为什么输出使用 `empty_like`

Kernel 会覆盖输出的全部有效元素，不需要初始化为零。`empty_like` 同时保留输入的 dtype、shape 和 device，并使用 PyTorch allocator 管理存储。

### 为什么不在 binding 中同步

同步会阻断 PyTorch 的异步执行和跨算子流水。Binding 只提交 kernel，并把 output 返回给当前 stream 上的后续工作。

### 为什么显式拒绝 autograd

仅注册 forward 却允许 `requires_grad=True` 会产生不完整的梯度语义。当前接口在入口处拒绝该输入，直到 backward kernel 和 autograd registration 实现后再解除限制。

## 当前限制

- 仅支持 CUDA、FP32、连续 `[B, H, N, 64]` Tensor。
- 仅实现 forward，不支持 autograd。
- 不支持 dropout、FP16、BF16、Q 与 KV 序列长度不同的 attention，以及 packed variable-length sequence。
- Q、K、V 必须具有相同 shape 并位于同一设备。
- Extension 没有注册 autocast 和 deterministic-algorithm 专用实现。
