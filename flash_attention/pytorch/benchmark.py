import argparse

import torch
import torch.nn.functional as F

from cuda_operator import flash_attention


def benchmark_ms(function, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    stop.record()
    stop.synchronize()
    return start.elapsed_time(stop) / iterations


def attention_tflops(
    batch_size: int,
    head_num: int,
    seq_len: int,
    head_dim: int,
    causal: bool,
    milliseconds: float,
) -> float:
    score_count = seq_len * (seq_len + 1) / 2 if causal else seq_len * seq_len
    flop = 4.0 * batch_size * head_num * score_count * head_dim
    return flop / (milliseconds * 1e9)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the Flash Attention extension")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    shape = (args.batch_size, args.heads, args.seq_len, 64)
    torch.manual_seed(2026)
    q = torch.randn(shape, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.inference_mode():
        custom_ms = benchmark_ms(
            lambda: flash_attention(q, k, v, args.causal),
            args.warmup,
            args.iterations,
        )
        sdpa_ms = benchmark_ms(
            lambda: F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=args.causal
            ),
            args.warmup,
            args.iterations,
        )

    custom_tflops = attention_tflops(*shape, args.causal, custom_ms)
    sdpa_tflops = attention_tflops(*shape, args.causal, sdpa_ms)

    print("===== PyTorch Flash Attention Benchmark =====")
    print(f"GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"shape={shape}  causal={args.causal}\n")
    print(f"{'cuda_operator':24s} time={custom_ms:8.4f} ms  TFLOPS={custom_tflops:8.2f}")
    print(f"{'PyTorch SDPA':24s} time={sdpa_ms:8.4f} ms  TFLOPS={sdpa_tflops:8.2f}")
    print(f"relative speed: {sdpa_ms / custom_ms:.2f}x")


if __name__ == "__main__":
    main()
