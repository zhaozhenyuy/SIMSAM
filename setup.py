import glob
import os

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
    build_cuda = os.environ.get("SIMSAM_BUILD_CUDA", "auto").lower()
    if build_cuda not in ("auto", "0", "1"):
        raise ValueError("SIMSAM_BUILD_CUDA must be auto, 0, or 1")
    if build_cuda == "0" or (build_cuda == "auto" and CUDA_HOME is None):
        print("GroundingDINO will use its PyTorch deformable-attention fallback.")
        return []
    if CUDA_HOME is None:
        raise RuntimeError("SIMSAM_BUILD_CUDA=1 requires a CUDA toolkit and compiler.")

    sources = [os.path.join(source_dir, "vision.cpp")]
    sources += glob.glob(os.path.join(source_dir, "MsDeformAttn", "*.cpp"))
    sources += glob.glob(os.path.join(source_dir, "MsDeformAttn", "*.cu"))
    sources += glob.glob(os.path.join(source_dir, "*.cu"))
    return [CUDAExtension(
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
    )]


setup(
    name="simsam",
    version="1.0.1",
    python_requires=">=3.10",
    description="Automatic-prompt SAM for sparsely supervised echocardiography videos",
    packages=find_packages(),
    py_modules=[
        "config", "lvef_metrics", "plot_lvef_agreement", "evaluate_lvef",
        "trainmemsam", "testmemsam", "trainsimsam", "testsimsam", "train",
        "export_camus_dino",
    ],
    ext_modules=groundingdino_extension(),
    cmdclass={"build_ext": BuildExtension},
)
