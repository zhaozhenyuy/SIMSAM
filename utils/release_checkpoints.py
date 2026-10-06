from pathlib import Path

import torch


def load_release_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    for wrapper in ('model', 'state_dict', 'model_state_dict'):
        if wrapper in checkpoint and isinstance(checkpoint[wrapper], dict):
            checkpoint = checkpoint[wrapper]
            break
    source = {key.removeprefix('module.'): value for key, value in checkpoint.items()}
    target = model.state_dict()
    removed = sorted(key for key in source if key.startswith('memory.memory_reinforce.'))
    source = {key: value for key, value in source.items() if key not in removed}
    missing = sorted(target.keys() - source.keys())
    unexpected = sorted(source.keys() - target.keys())
    mismatched = sorted(key for key in source.keys() & target.keys()
                        if source[key].shape != target[key].shape)
    if missing or unexpected or mismatched:
        raise RuntimeError(
            'Checkpoint does not fully match 94.04 without Mamba reinforcement. '
            'Check APFE, phase, memory and model switches; required tensors must not be random.\n'
            f'Missing: {missing[:12]}\nUnexpected: {unexpected[:12]}\n'
            f'Shape mismatches: {mismatched[:12]}'
        )
    model.load_state_dict(source, strict=True)
    if removed:
        print(f'[CHECKPOINT] Ignored {len(removed)} removed Mamba tensors only.')
    return [], []
