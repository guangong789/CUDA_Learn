# CUDA Operator

本项目使用 CUDA C++ 实现 Reduce、SGEMM、Softmax 和 Flash Attention 算子，并记录各版本的并行划分、访存方式和性能变化。算子通过 CPU、CUB、cuBLAS 或 PyTorch reference 进行正确性验证，使用 CUDA Event 进行性能测试，并使用 Nsight Compute 分析内存吞吐、warp stall、occupancy 和指令调度。

项目同时提供 PyTorch C++/CUDA Extension、C ABI 和 Rust 接口，用于验证 CUDA kernel 在独立程序及上层语言中的调用方式。

## 技术范围

- 开发语言：C++23、CUDA C++20、Python、Rust
- 构建工具：CMake、PyTorch C++ Extension、Cargo
- 优化方法：线程与数据分块、合并访存、`float4` 向量化、shared memory、register tiling、warp shuffle、double buffering 和 online softmax
- 性能分析：CUDA Event、Nsight Compute、Roofline、memory workload、scheduler 和 occupancy
- 对照实现：CPU、CUB、cuBLAS、PyTorch 显式 Attention 和 PyTorch SDPA

## 构建

工程默认将目标 GPU 架构设为 `sm_86`。在其他架构上构建时，需要修改根目录 `CMakeLists.txt` 中的 `CMAKE_CUDA_ARCHITECTURES`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

构建产物位于 `build/`。以下程序用于运行主要 benchmark：

```bash
./build/bm_sgemm
./build/bm_softmax
./build/bm_flash_attention
```

## Reduce

![alt text](assets/image.png)  

包含 v0-v7 和 CUB reference，依次分析非合并访存、线程束分化、shared-memory bank conflict、`float4` 向量化、循环展开和 warp shuffle。测试规模为 `N = 67,108,864`；向量化版本记录的显存吞吐为 320 GB/s。

实现与分析见 [READ_REDUCE](reduce/READ_REDUCE.md)。

## SGEMM

![alt text](assets/image-1.png)  

包含 v0-v5 和 cuBLAS reference，覆盖 shared-memory tiling、单线程多数据映射、register tiling、向量化访存、矩阵转置实验和 double buffering。`4096 x 4096 x 4096` 测试中，v5 记录为 6310.79 GFLOPS，对应同配置 cuBLAS FP32 性能的 95.9%。

实现与分析见 [READ_SGEMM](sgemm/READ_SGEMM.md)。

## Softmax

![alt text](assets/image-2.png)  

包含 v0-v5，覆盖 shared-memory reduction、warp shuffle、`float4` 向量化、warp-level reduction、warp-per-row 和 online softmax。`M = 4096, N = 1024` 测试中，v2 与 v3 记录为 0.1099 ms，对应约 305 GB/s 的有效带宽。

实现与分析见 [READ_SOFTMAX](softmax/READ_SOFTMAX.md)。

## Flash Attention

项目实现 FP32 Flash Attention forward kernel，输入布局为 `[batch, head, sequence, head_dim]`。Kernel 使用 tiled Q/K/V 和 online softmax，不生成完整的 attention score 矩阵，支持 causal、non-causal 和不规则 sequence length。

当前实现固定支持 `head_dim = 64`。算法、内存布局和状态合并方式见 [READ_FLASH_ATTENTION](flash_attention/READ_FLASH_ATTENTION.md)。

## PyTorch C++/CUDA Extension

`flash_attention/pytorch/` 将 Flash Attention kernel 注册为 PyTorch 自定义算子。接口接收三个连续的 FP32 CUDA Tensor，并使用 PyTorch 当前 CUDA stream：

```python
import torch
from cuda_operator import flash_attention

q = torch.randn((1, 8, 257, 64), device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q, k, v, causal=True)
```

扩展包含正确性测试以及与 PyTorch SDPA 的 benchmark，构建与使用方式见 [PyTorch Extension README](flash_attention/pytorch/README.md)。

## Rust Interface

`flash_attention/rust/` 通过 C ABI 调用同一个 Flash Attention kernel。Rust 层提供 `Tensor`、`CudaStream`、参数检查、CUDA 错误处理和显存生命周期管理：

```rust
use cuda_operator::{Tensor, flash_attention};

let q = Tensor::from_slice(&q_host, [1, 8, 257, 64])?;
let k = Tensor::from_slice(&k_host, [1, 8, 257, 64])?;
let v = Tensor::from_slice(&v_host, [1, 8, 257, 64])?;
let output = flash_attention(&q, &k, &v, true)?;
```

构建、测试和非默认 CUDA stream 接口见 [Rust Interface README](flash_attention/rust/README.md)。

## 验证

- 所有 CMake target 均通过 Release 构建。
- PyTorch Extension 的 9 项测试通过。
- Rust 接口的 4 项测试通过。
- Flash Attention 测试覆盖 causal、non-causal、规则与不规则 sequence length、非默认 CUDA stream 和非法输入。
- Flash Attention 分别与 CPU reference、PyTorch 显式实现和 PyTorch SDPA 进行结果对比。

验证环境：

- Ubuntu on WSL2
- Python 3.10
- PyTorch `2.6.0+cu124`
- CUDA Toolkit 12.4
- RTX 3060 Laptop GPU（`sm_86`）

性能数据对应上述硬件、软件版本及各算子文档中记录的输入规模。

## 当前限制

- Flash Attention 仅实现 FP32 forward，固定 `head_dim = 64`。
- PyTorch Extension 不支持 autograd、dropout、FP16/BF16 和打包的变长序列。
- Reduce、SGEMM 和 Softmax 的部分版本使用编译期固定的输入规模。
- 当前接口未包含跨 GPU 通信和分布式执行。
