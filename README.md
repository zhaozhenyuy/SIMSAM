# SIMSAM

SIMSAM is an automatic left ventricular segmentation model for echocardiographic
videos, using a shared visual encoder, text-driven automatic prompting and
phase-aware memory.

## Installation

```bash
conda create -n simsam python=3.10 -y
conda activate simsam
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

On Windows without a CUDA compiler, set `$env:SIMSAM_BUILD_CUDA = "0"` in
PowerShell before the last installation command.

## Usage

### Prepare Dataset

Download the datasets from their official websites:

- [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/)
- [EchoNet-Dynamic](https://echonet.github.io/dynamic/)

Preprocessing scripts are provided in `utils/preprocess_camus.py` and
`utils/preprocess_echonet.py`. Use the processed video dataset layout:

```text
DATA_ROOT/
  class.json
  videos/{train,val,test}/*.npy
  annotations/{train,val,test}/*.npz
```

### Download Pretrained Weights

- [SAM ViT-B: sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
- [GroundingDINO Swin-T: groundingdino_swint_ogc.pth](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth)

Place both files in `weights/`. Place the dataset-specific GroundingDINO LoRA
checkpoint at `weights/best_model.pth` and the trained SIMSAM checkpoint at
`weights/simsam_best.pth`. Task-specific checkpoints must be trained or obtained
separately; they are not included in this repository.

### Train And Test

Run from the repository root. Replace `DATA_ROOT` with your processed dataset path.

Train:

```bash
python trainsimsam.py --data_path DATA_ROOT
```

Test:

```bash
python testsimsam.py --data_path DATA_ROOT --load_path weights/simsam_best.pth
```

The default dataset is CAMUS. For EchoNet-Dynamic, add `--task EchoNet_Video`
and use its corresponding dataset and checkpoints. CAMUS testing evaluates all
10 frames; EchoNet-Dynamic testing evaluates ED and ES. Optional settings are
available through `--help`.

## Acknowledgement

This work builds on [SAM](https://github.com/facebookresearch/segment-anything),
[GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and MemSAM.
We thank the authors for their open-source contributions.
