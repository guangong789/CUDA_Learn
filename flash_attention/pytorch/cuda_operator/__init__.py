import torch

from . import _C  # noqa: F401

def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                    causal: bool = False,) -> torch.Tensor:
    return torch.ops.cuda_operator.flash_attention(q, k, v, causal)

@torch.library.register_fake("cuda_operator::flash_attention")  # fake kernel
def _flash_attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          causal: bool = False,) -> torch.Tensor:
    torch._check(q.shape == k.shape and q.shape == v.shape)
    torch._check(q.ndim == 4)
    torch._check(q.shape[-1] == 64)
    torch._check(q.dtype == torch.float32)
    torch._check(k.dtype == torch.float32)
    torch._check(v.dtype == torch.float32)
    return torch.empty_like(q)

__all__ = ["flash_attention"]  # only