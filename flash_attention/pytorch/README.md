# PyTorch C++/CUDA Extension

This module exposes `flash_attention_v0` to Python as a PyTorch custom operator. The implementation follows this call path:

```text
cuda_operator.flash_attention(...)
    -> torch.ops.cuda_operator.flash_attention(...)
    -> C++ validation and output allocation
    -> current PyTorch CUDA stream
    -> launch_flash_attention_v0(...)
    -> CUDA kernel
```

## Supported inputs

- CUDA tensors with `[batch, head, sequence, head_dim]` layout
- `torch.float32`
- contiguous Q, K and V with identical shapes
- dynamic batch, head and sequence dimensions
- `head_dim = 64`
- causal and non-causal forward

This is a forward-only educational operator. It does not implement autograd, dropout, FP16/BF16 or packed variable-length sequences.

## Build

Install a CUDA-enabled PyTorch build first using the [official PyTorch selector](https://pytorch.org/get-started/locally/). Verify that PyTorch can see the CUDA toolkit:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

This repository was verified with Python 3.10, PyTorch `2.6.0+cu124`, CUDA toolkit 12.4 and an RTX 3060 Laptop GPU (`sm_86`). A matching isolated environment can be created from the repository root with:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

Then build the extension in place:

```bash
cd flash_attention/pytorch
python -m pip install -e . --no-build-isolation
```

PyTorch normally detects the architecture of the visible GPU. It can also be specified explicitly, for example for an RTX 30-series GPU:

```bash
TORCH_CUDA_ARCH_LIST=8.6 python -m pip install -e . --no-build-isolation
```

## Use

```python
import torch
from cuda_operator import flash_attention

q = torch.randn((1, 8, 257, 64), device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)
output = flash_attention(q, k, v, causal=True)
```

## Test

```bash
python test_flash_attention.py
```

The test suite covers causal and non-causal attention, regular and irregular sequence lengths, execution on a non-default CUDA stream, invalid input diagnostics, and `torch.library.opcheck`. Results are compared with both an explicit PyTorch implementation and `torch.nn.functional.scaled_dot_product_attention`.

## Benchmark

```bash
python benchmark.py
python benchmark.py --causal
python benchmark.py --batch-size 1 --heads 8 --seq-len 257
```

The benchmark reports CUDA event time and effective TFLOPS for this operator and PyTorch SDPA. PyTorch may select a highly optimized fused backend, so the comparison is intended to expose the real performance gap rather than guarantee a speedup.

On the verified RTX 3060 Laptop GPU, a short `B=4, H=8, N=1024, D=64` run produced:

| implementation | non-causal | causal |
|---|---:|---:|
| `cuda_operator` | 7.41 ms | 4.53 ms |
| PyTorch SDPA | 2.61 ms | 1.43 ms |

These numbers are a development snapshot rather than a cross-machine performance claim. Use the benchmark script to reproduce them on the target GPU.
