# SIMSAM

SIMSAM performs left-ventricle segmentation in echocardiography videos with a
shared GroundingDINO visual encoder, automatic first-frame prompting, feature
memory, and normalized ED-to-ES phase memory.

The release configuration enables APFE and phase memory by default. The old
MemSAM foreground-reinforcement convolution and its Mamba replacement have
been removed.

## Installation

Python 3.8+ and a CUDA-enabled PyTorch installation are required.

```bash
pip install -r requirements.txt
pip install -e .
```

Place pretrained weights in `weights/`:

```text
weights/
├── sam_vit_b_01ec64.pth
├── groundingdino_swint_ogc.pth
├── dino_lora_camus.pth
└── dino_lora_echonet.pth
```

## Data

Both datasets use the same processed layout:

```text
DATA_ROOT/
├── class.json
├── videos/{train,val,test}/*.npy
└── annotations/{train,val,test}/*.npz
```

Preprocessing helpers for CAMUS and EchoNet-Dynamic are available in `utils/`.

To fine-tune the automatic prompt localizer on CAMUS instead of using the
provided LoRA weights:

```bash
python export_camus_dino.py
python train.py
```

The paths and GroundingDINO training settings are defined in
`configs/train_config.yaml`.

## Training

The defaults reproduce the main setup: 10 frames, endpoint supervision,
automatic prompting, feature memory, APFE, phase memory, AdamW, learning rate
`1e-4`, and weight decay `0.02`.

CAMUS:

```bash
python trainmemsam.py --data_path /path/to/CAMUS_public
```

EchoNet-Dynamic:

```bash
python trainmemsam.py --task EchoNet_Video --data_path /path/to/echocycle
```

Use `--seed`, `--epochs`, or `--output_dir` only when overriding the defaults.
For ablations, use `--disable_phase_memory` or `--disable_apfe`.

## Evaluation

CAMUS evaluates all 10 sampled frames by default:

```bash
python testmemsam.py \
  --load_path checkpoints/camus/simsam_best.pth \
  --data_path /path/to/CAMUS_public
```

EchoNet-Dynamic automatically evaluates its ED and ES endpoints:

```bash
python testmemsam.py \
  --task EchoNet_Video \
  --load_path checkpoints/echonet/simsam_best.pth \
  --data_path /path/to/echocycle
```

Add `--visual` to save per-frame visualizations under the configured `results/`
directory.
