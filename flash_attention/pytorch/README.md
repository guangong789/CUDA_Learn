# PyTorch C++/CUDA 扩展

本模块将 `flash_attention_v0` 注册为 PyTorch 自定义算子，使其可以直接在 Python 中调用。整体调用链如下：

```text
cuda_operator.flash_attention(...)
    -> torch.ops.cuda_operator.flash_attention(...)
    -> C++ validation and output allocation
    -> current PyTorch CUDA stream
    -> launch_flash_attention_v0(...)
    -> CUDA kernel
```

## 支持范围

- 输入为 `[batch, head, sequence, head_dim]` 布局的 CUDA Tensor
- 数据类型为 `torch.float32`
- Q、K、V 的形状相同且内存连续
- batch、head 和 sequence 维度可以动态变化
- `head_dim = 64`
- 支持 causal 和 non-causal forward

这是一个仅实现前向计算（forward）的教学算子，不支持自动求导（autograd）、dropout、FP16/BF16 和打包的变长序列。

## 构建

首先通过 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)安装支持 CUDA 的 PyTorch，并确认 PyTorch 能够识别 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

本项目已在 Python 3.10、PyTorch `2.6.0+cu124`、CUDA Toolkit 12.4 和 RTX 3060 Laptop GPU（`sm_86`）环境下完成验证。可以在仓库根目录创建对应的隔离环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

随后在源码目录中编译并以可编辑（editable）模式安装扩展：

```bash
cd flash_attention/pytorch
python -m pip install -e . --no-build-isolation
```

PyTorch 通常会自动检测当前可见 GPU 的计算架构。如果构建环境无法访问 GPU，也可以显式指定架构。例如 RTX 30 系列 GPU 使用：

```bash
TORCH_CUDA_ARCH_LIST=8.6 python -m pip install -e . --no-build-isolation
```

## 使用

```python
import torch
from cuda_operator import flash_attention

q = torch.randn((1, 8, 257, 64), device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v, causal=True)
```

## 测试

```bash
python test_flash_attention.py
```

测试覆盖 causal 和 non-causal attention、规则与不规则序列长度、非默认 CUDA stream、非法输入诊断和 `torch.library.opcheck`。计算结果会同时与显式 PyTorch 实现及 `torch.nn.functional.scaled_dot_product_attention` 对拍。

## 性能测试

```bash
python benchmark.py
python benchmark.py --causal
python benchmark.py --batch-size 1 --heads 8 --seq-len 257
```

性能测试使用 CUDA Event 统计本算子和 PyTorch SDPA 的耗时与有效 TFLOPS。PyTorch 可能会选择高度优化的融合后端，因此该测试旨在如实展示性能差距，而不是保证本算子能够获得加速。

在用于验证的 RTX 3060 Laptop GPU 上，使用 `B=4, H=8, N=1024, D=64` 进行短时间测试，结果如下：

| 实现 | non-causal | causal |
|---|---:|---:|
| `cuda_operator` | 7.41 ms | 4.53 ms |
| PyTorch SDPA | 2.61 ms | 1.43 ms |

这些数据只是当前开发环境下的测试快照，不代表跨设备的通用性能结论。请在目标 GPU 上运行 benchmark 脚本复现结果。
