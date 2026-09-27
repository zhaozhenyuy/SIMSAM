import glob
import os

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME


def groundingdino_extension():
    source_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "groundingdino",
        "models",
        "GroundingDINO",
        "csrc",
    )
    if CUDA_HOME is None:
        raise RuntimeError("CUDA toolkit was not found; it is required to build GroundingDINO ops.")

    sources = [os.path.join(source_dir, "vision.cpp")]
    sources += glob.glob(os.path.join(source_dir, "MsDeformAttn", "*.cpp"))
    sources += glob.glob(os.path.join(source_dir, "MsDeformAttn", "*.cu"))
    sources += glob.glob(os.path.join(source_dir, "*.cu"))
    return CUDAExtension(
        "groundingdino._C",
        sources=sources,
        include_dirs=[source_dir],
        define_macros=[("WITH_CUDA", None)],
        extra_compile_args={
            "cxx": [],
            "nvcc": [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
        },
    )


setup(
    name="simsam",
    version="1.0.0",
    description="Automatic-prompt SAM for sparsely supervised echocardiography videos",
    packages=find_packages(
        exclude=(
            "models.segment_anything",
            "models.segment_anything.*",
            "models.segment_anything_memsam.utils",
            "models.segment_anything_memsam.utils.*",
        )
    ),
    ext_modules=[groundingdino_extension()],
    cmdclass={"build_ext": BuildExtension},
)
