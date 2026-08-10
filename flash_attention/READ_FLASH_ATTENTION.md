# Flash Attention

`flash_attention_v0` 是一个 FP32 forward 实现，输入布局为 `[batch, head, sequence, head_dim]`，当前固定支持 `head_dim = 64`，同时支持 causal 和 non-causal attention。

Kernel 使用 `32 x 32` 的 Q/KV tile。每个 block 包含 8 个 warp，处理 32 行 Q；Q 在整个 KV 循环期间常驻 shared memory，K 以转置且带 padding 的布局写入 shared memory，避免计算 `QK^T` 时产生 shared-memory bank conflict。

每个 KV tile 内先计算局部：

```text
m_tile = max(S_tile)
l_tile = sum(exp(S_tile - m_tile))
O_tile = exp(S_tile - m_tile) @ V_tile
```

随后使用 online softmax 合并历史状态：

```text
m_new = max(m_old, m_tile)
alpha = exp(m_old - m_new)
beta  = exp(m_tile - m_new)
l_new = alpha * l_old + beta * l_tile
O_new = alpha * O_old + beta * O_tile
```

因此整个过程中不会生成或写回 `[sequence, sequence]` 的 attention score 矩阵，也不需要额外的 global-memory workspace；Q/K/V/O 的 global storage 为 O(ND)，而不是 O(N^2)。

`flash_attention_v0` 会分别对拍 non-causal 和 causal CPU reference；`bm_flash_attention` 默认测试 `B=4, H=8, N=1024, D=64`。

## PyTorch Extension

`pytorch/` 将同一个 CUDA kernel 注册为 `torch.ops.cuda_operator.flash_attention`，并提供 Python 包装、正确性测试和与 PyTorch SDPA 的 benchmark。构建与使用方式见 [pytorch/README.md](pytorch/README.md)。
