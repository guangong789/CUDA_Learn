import math
import unittest

import torch
import torch.nn.functional as F

from cuda_operator import flash_attention


def attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
) -> torch.Tensor:
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if causal:
        seq_len = q.size(-2)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, -torch.inf)
    return torch.matmul(torch.softmax(scores, dim=-1), v)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FlashAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(2026)
        torch.cuda.manual_seed_all(2026)

    def assert_attention_close(self, shape: tuple[int, int, int, int], causal: bool) -> None:
        q = torch.randn(shape, device="cuda", dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        output = flash_attention(q, k, v, causal)
        reference = attention_reference(q, k, v, causal)
        sdpa = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=causal
        )

        torch.testing.assert_close(output, reference, atol=3e-4, rtol=3e-3)
        torch.testing.assert_close(output, sdpa, atol=3e-4, rtol=3e-3)

    def test_regular_and_irregular_shapes(self) -> None:
        shapes = [
            (1, 1, 32, 64),
            (1, 2, 127, 64),
            (2, 4, 256, 64),
            (1, 3, 257, 64),
            (1, 2, 1000, 64),
        ]
        for shape in shapes:
            for causal in (False, True):
                with self.subTest(shape=shape, causal=causal):
                    self.assert_attention_close(shape, causal)

    def test_non_default_stream(self) -> None:
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            q = torch.randn((1, 2, 127, 64), device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            output = flash_attention(q, k, v, causal=True)
        stream.synchronize()

        reference = attention_reference(q, k, v, causal=True)
        torch.testing.assert_close(output, reference, atol=3e-4, rtol=3e-3)

    def test_pytorch_operator_registration(self) -> None:
        q = torch.randn((1, 2, 127, 64), device="cuda")
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        torch.library.opcheck(
            torch.ops.cuda_operator.flash_attention.default,
            (q, k, v, False),
        )

    def test_rejects_wrong_dtype(self) -> None:
        q = torch.randn((1, 1, 32, 64), device="cuda", dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "float32"):
            flash_attention(q, q, q)

    def test_rejects_cpu_input(self) -> None:
        q = torch.randn((1, 1, 32, 64), device="cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA-only"):
            flash_attention(q, q, q)

    def test_rejects_non_contiguous_input(self) -> None:
        q = torch.randn((2, 3, 32, 64), device="cuda").transpose(0, 1)
        self.assertFalse(q.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            flash_attention(q, q, q)

    def test_rejects_wrong_head_dimension(self) -> None:
        q = torch.randn((1, 1, 32, 32), device="cuda")
        with self.assertRaisesRegex(RuntimeError, "head_dim=64"):
            flash_attention(q, q, q)

    def test_rejects_shape_mismatch(self) -> None:
        q = torch.randn((1, 1, 32, 64), device="cuda")
        k = torch.randn((1, 1, 33, 64), device="cuda")
        v = torch.randn_like(q)
        with self.assertRaisesRegex(RuntimeError, "same shape"):
            flash_attention(q, k, v)

    def test_rejects_autograd_input(self) -> None:
        q = torch.randn((1, 1, 32, 64), device="cuda", requires_grad=True)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        with self.assertRaisesRegex(RuntimeError, "forward-only"):
            flash_attention(q, k, v)


if __name__ == "__main__":
    unittest.main(verbosity=2)
