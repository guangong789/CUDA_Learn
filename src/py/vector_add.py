import torch
import triton
import triton.language as tl

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("cuda version:", torch.version.cuda)

x = torch.randn(2048, 2048, device="cuda")
y = x @ x

print("result shape:", y.shape)
print("mean:", y.mean().item())

@triton.jit
def square_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    output = x * x
    tl.store(output_ptr + offsets, output, mask=mask)

def square(x: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    square_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output

try:
    data = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
    result = square(data)
    print(f"Triton 运行成功! 结果: {result}")
except Exception as e:
    print(f"运行失败，错误原因:\n{e}")