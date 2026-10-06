# SIMSAM: 94.04 Reference Without Mamba Reinforcement

This release uses the original `94.04` model, loss, video loader, training
schedule and evaluation implementation. It does not use the separately
rewritten/trimmed SIMSAM training implementation.

## Scope

- The foreground Mamba branch `memory.memory_reinforce` is disabled and cannot
  be enabled through `--reinforce`. Installing `mamba-ssm` is not required.
- The basic value-encoder `HiddenReinforcer` and decoder `HiddenUpdater` remain
  active exactly as in `94.04 --disable_reinforce`. They are not the Mamba branch.
- Automatic first-frame DINO prompting, the shared visual encoder, feature
  adapter, temporal memory, APFE and the original phase MLP are retained.
- APFE, phase memory, DINO LoRA, warmup and optional ES shape losses retain
  the original explicit switches. They are not silently turned on.
- Original loss arithmetic, data augmentation and frame sampling are retained.
  The legacy GroundingDINO LoRA training/merge implementation is also retained.
- CAMUS/EchoNet testing, visualization, CAMUS patient-level LVEF export and
  correlation/Bland-Altman plotting are included.
- Portable paths, Windows worker serialization, Unicode image output and
  stricter full-model checkpoint validation are added. Empty CAMUS masks use
  the original evaluator's existing surface-metric fallback helper.

`trainsimsam.py` and `testsimsam.py` are aliases of `trainmemsam.py` and
`testmemsam.py`. Specify `--modelname SharedGroundedMemSAM` when training SIMSAM.
The training parser's original default model name remains `MemSAM`.

## Installation

Use a separate Python 3.10 environment:

```bash
conda create -n simsam python=3.10 -y
conda activate simsam
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
```

Native GroundingDINO CUDA ops require a compatible CUDA toolkit and compiler.
For Windows without a configured compiler, use the PyTorch attention fallback:

```powershell
$env:SIMSAM_BUILD_CUDA = "0"
python -m pip install --no-build-isolation -e .
```

For CPU functional checks install the matching CPU wheels and add `--device cpu`.
CPU checks and the fallback ops are not evidence of published GPU FPS.
Use the same ops implementation, hardware and settings when comparing speed.
The first model build also needs cached/downloadable `bert-base-uncased` and
ImageNet ResNet-18 weights. Only load checkpoints from trusted sources.

## Weights

| File | Official Download | Official Source |
| --- | --- | --- |
| `groundingdino_swint_ogc.pth` | [GroundingDINO Swin-T](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) | [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) |
| `sam_vit_b_01ec64.pth` | [SAM ViT-B](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) | [Segment Anything](https://github.com/facebookresearch/segment-anything) |

Put these files in `weights/`. Use `--sam_ckpt` and `--dino_weights` to select
another location. Provide your task-specific GroundingDINO LoRA checkpoint
using `--dino_use_lora --dino_lora_weights PATH`.

Official SAM/DINO pretraining weights are not SIMSAM segmentation weights.
No verified public URL for task-specific LoRA or SIMSAM checkpoints is supplied;
these must be supplied by the authors or trained locally. They are not included
in the source archive. Do not replace a fine-tuned locator with the base DINO
weights when trying to reproduce an experiment that used LoRA.

## Datasets

- [CAMUS official project/download entry](https://www.creatis.insa-lyon.fr/Challenge/camus/)
  and [Human Heart Project database](https://humanheart-project.creatis.insa-lyon.fr/database/).
- [EchoNet-Dynamic official dataset](https://echonet.github.io/dynamic/)
  and [Stanford AIMI access page](https://aimi.stanford.edu/datasets/echonet-dynamic-cardiac-ultrasound).

Follow the providers' registration and use conditions. Keep the exact processed
data, patient splits and ED/ES conventions used to train the selected checkpoint.
The original video preprocessing helpers are retained in `utils/`; do not rerun
them with a different split/order for an existing checkpoint.

```text
DATA_ROOT/
  class.json
  videos/{train,val,test}/*.npy
  annotations/{train,val,test}/*.npz
```

Videos use `(3, T, H, W)` arrays. Annotations contain `fnum_mask` (a dictionary
of original frame indices and masks), `ef`, `edv`, `esv`, and `spacing`.
`class.json` must define the binary task, for example `{"camus": 2}` or
`{"EchoNet": 2}`. The loader retains the original class-key selection policy.
This is not an `images/test/*.png` endpoint dataset interface.

## Training

Run from the repository root; supply your existing processed dataset and LoRA
weights. This example explicitly selects the phase/APFE configuration:

```bash
python trainmemsam.py --modelname SharedGroundedMemSAM --task CAMUS_Video_Full --data_path DATA_ROOT --sam_ckpt weights/sam_vit_b_01ec64.pth --dino_weights weights/groundingdino_swint_ogc.pth --dino_use_lora --dino_lora_weights weights/best_model.pth --enable_memory --semi --disable_point_prompt --enable_apfe --apfe_kernel_size 7 --enable_phase_memory --frame_length 10 --batch_size 1 --epochs 150 --base_lr 0.0001 --warmup --warmup_period 250 --keep_log --output_dir runs/camus
```

For EchoNet use `--task EchoNet_Video --data_path ECHOCYCLE_ROOT` and the
corresponding locator weights. Keep `--semi`: replicated intermediate masks
are not extra ground-truth annotations. Match epochs and all other settings
to the intended experiment; the task defaults remain those of `94.04`.

ES weighting is `--es_loss_weight` (original default 1.0). Additional ES shape
losses require `--enable_es_shape_loss`; their original defaults are boundary
0.2 and area 0.1. Select the exact values used in your experiment. The original
warmup/polynomial learning-rate schedule is used only with `--warmup`.
`--workers`, `--seed` and `--output_dir` are portable overrides.

## Testing And LVEF

```bash
python testmemsam.py --modelname SharedGroundedMemSAM --task CAMUS_Video_Full --data_path DATA_ROOT --load_path SIMSAM_CHECKPOINT --dino_use_lora --dino_lora_weights weights/best_model.pth --enable_apfe --apfe_kernel_size 7 --enable_phase_memory --frame_length 10 --batch_size 1 --full_eval --visual --compute_ef --clinical_output_dir runs/camus_lvef --output_dir runs/camus_test
```

Remove `--enable_phase_memory` if the checkpoint was trained without that
module. APFE, memory, prompt and phase switches must match training. Missing,
extra or shape-mismatched active tensors stop evaluation; only tensors belonging
to the removed Mamba branch are ignored. Evaluating an Mamba-trained checkpoint
with Mamba disabled is a changed configuration, not its original trained model.

For EchoNet-Dynamic use `--task EchoNet_Video`, the matching checkpoint and
locator, and omit `--full_eval --compute_ef --clinical_output_dir ...`.
The video still contains 10 sampled frames; only ED/ES are evaluated because
intermediate GT is unavailable. CAMUS paired-view LVEF cannot be applied to
EchoNet's single-view annotations.

Visualization is saved under `<output_dir>/results/vis/<modelname>/`.
CAMUS LVEF produces `clinical_per_patient.csv`, `clinical_summary.json` and
`clinical_invalid.csv`. A patient requires both 2CH and 4CH endpoints.
Corr (%), MAE, signed Bias and 95% LoA use the original evaluation definitions.
Inspect excluded patients before comparing methods. PSD is not a reported metric.

```bash
python plot_lvef_agreement.py --csv runs/camus_lvef/clinical_per_patient.csv --output-dir runs/camus_lvef/figures --dpi 600
```

The original 10-frame CAMUS FPS timing is retained: wall-clock model-forward
time, no explicit CUDA synchronization and no warmup. It must not be described
as a synchronized inference benchmark. Use 10 frames and batch size 1 for that
legacy comparison; no new paper performance or FPS values are claimed here.

## GroundingDINO Box Export And Fine-Tuning

Use the loader-based exporter to preserve the segmentation dataset convention:

```bash
python tools/export_dino_from_loader.py --dataset_path DATA_ROOT --out_root data/camus_dino --export_mode ED_ES --resize_to_img_size --img_size 256 --num_workers 0
python train.py --config configs/train_config.yaml
```

Use `data/echodynamic_dino` and `configs/echodynamic_train_config.yaml` for
EchoNet. Boxes are taken from the endpoint LV masks; no intermediate GT is
needed. Edit YAML paths to your exported CSVs. The older specialized exporter
scripts are retained for reference; prefer the loader-based command above.
This release does not silently replace the original LoRA attention/merge or
checkpoint-selection algorithm with the previous cleanup's rewritten version.

## Checks

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

Development verification compared this model with `94.04` configured with
Mamba off: active state keys/shapes, a full synthetic 10-frame prediction,
endpoint/shape losses and gradients, and loader outputs matched. Checks use
CPU and synthetic inputs, not the complete clinical test set. Reproduce paper
accuracy and GPU throughput separately with the correct trained weights/data.
