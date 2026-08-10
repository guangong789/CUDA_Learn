# CUDA Operator

这是一个用于学习 CUDA 算子开发与性能优化的项目，包含 Reduce、SGEMM、Softmax 和 Flash Attention 的多种实现。项目记录了从基础版本到分块、向量化、warp-level primitive、shared memory 等优化方法的实践过程，并通过 CPU、cuBLAS 或 PyTorch 实现进行正确性验证。

本项目以理解算子实现、性能分析和工程接入为主要目的，不作为生产级算子库使用。

## 构建

当前工程使用 C++23、CUDA C++20，并将目标 GPU 架构设为 `sm_86`。在其他架构上构建时，需要修改根目录 `CMakeLists.txt` 中的 `CMAKE_CUDA_ARCHITECTURES`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

构建完成后，可执行文件位于 `build/`，其中包括各版本的独立测试程序以及以下 benchmark：

```bash
./build/bm_sgemm
./build/bm_softmax
./build/bm_flash_attention
```

## Reduce

![alt text](assets/image.png)  
Read me in [READ_REDUCE](https://github.com/guangong789/CUDA_Learn/blob/main/reduce/READ_REDUCE.md)  

## Sgemm

![alt text](assets/image-1.png)  
Read me in [READ_SGEMM](https://github.com/guangong789/CUDA_Learn/blob/main/sgemm/READ_SGEMM.md)  

## Softmax  

![alt text](assets/image-2.png)  
Read me in [READ_SOFTMAX](https://github.com/guangong789/CUDA_Learn/blob/main/softmax/READ_SOFTMAX.md)

## Flash Attention

项目实现了一个 FP32 Flash Attention forward kernel。该实现使用 tiled Q/K/V 计算和 online softmax，不生成完整的 attention score 矩阵，支持 causal、non-causal 和不规则 sequence length。

当前实现固定支持 `head_dim = 64`，read me in [READ_FLASH_ATTENTION](flash_attention/READ_FLASH_ATTENTION.md)。

## PyTorch C++/CUDA Extension

`flash_attention/pytorch/` 将 Flash Attention kernel 注册为 PyTorch 自定义算子，可以从 Python 直接调用：

```python
import torch
from cuda_operator import flash_attention

q = torch.randn((1, 8, 257, 64), device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)

output = flash_attention(q, k, v, causal=True)
```

扩展提供正确性测试和与 PyTorch SDPA 的 benchmark。Read me in [PyTorch Extension README](flash_attention/pytorch/README.md)。

## 验证

当前版本已完成以下验证：

- 所有 CMake target 均可完成 Release 构建。
- PyTorch Extension 的 9 项测试全部通过。
- 测试覆盖规则与不规则 sequence length、causal 与 non-causal attention、非默认 CUDA stream 和非法输入检查。
- Flash Attention 的结果分别与显式 PyTorch 实现和 PyTorch SDPA 对比。

验证环境为 Python 3.10、PyTorch `2.6.0+cu124`、CUDA Toolkit 12.4 和 RTX 3060 Laptop GPU（`sm_86`）。不同硬件和软件版本下的结果可能存在差异。

## 当前限制

- Flash Attention 仅实现 FP32 forward，固定 `head_dim = 64`。
- PyTorch Extension 暂不支持 autograd、dropout、FP16/BF16 和打包的变长序列。
- 各算子的实现主要用于学习和实验，接口与性能未按生产环境要求设计。
