from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


PYTORCH_DIR = Path(__file__).resolve().parent
FLASH_ATTENTION_DIR = PYTORCH_DIR.parent


setup(
    name="cuda-operator",
    version="0.1.0",
    description="Educational FP32 Flash Attention PyTorch CUDA extension",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="cuda_operator._C",
            sources=[
                str(PYTORCH_DIR / "csrc" / "flash_attention_binding.cpp"),
                str(FLASH_ATTENTION_DIR / "flash_attention_v0.cu"),
            ],
            include_dirs=[str(FLASH_ATTENTION_DIR)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)
