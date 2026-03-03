import os
import sys

# 【核心修改】强制夺权，从内部指定使用 GCC 11 作为宿主编译器
os.environ["CC"] = "gcc-11"
os.environ["CXX"] = "g++-11"

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.path.dirname(os.path.abspath(__file__))

setup(
    name="xray_gaussian_rasterization_voxelization",
    packages=["xray_gaussian_rasterization_voxelization"],
    ext_modules=[
        CUDAExtension(
            name="xray_gaussian_rasterization_voxelization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "cuda_voxelizer/voxelizer_impl.cu",
                "cuda_voxelizer/forward.cu",
                "cuda_voxelizer/backward.cu",
                "voxelize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "-I"
                    + os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"
                    )
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)