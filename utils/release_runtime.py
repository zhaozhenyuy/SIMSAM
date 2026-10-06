from pathlib import Path

import torch

from .runtime import project_path, validate_dataset_splits


def configure_release(args, opt, training):
    if args.reinforce:
        raise ValueError('Mamba reinforcement is disabled in this release; remove --reinforce.')
    opt.device = args.device
    opt.modelname = args.modelname
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use --device cpu for functional checks only.')
    if args.n_gpu != 1 or (not training and args.batch_size != 1):
        raise ValueError('Use --n_gpu 1 and evaluation --batch_size 1 with the original evaluator.')
    if args.workers is not None:
        opt.workers = args.workers
    if opt.workers < 0:
        raise ValueError('workers must be nonnegative')
    if training and args.epochs is not None:
        if args.epochs < 1:
            raise ValueError('epochs must be positive')
        opt.epochs = args.epochs
    if not args.data_path:
        raise ValueError('Specify --data_path with the existing processed dataset root.')
    args.data_path = project_path(args.data_path)
    opt.data_path = args.data_path
    if training and not args.semi and args.task in ('EchoNet_Video', 'EchoDynamic'):
        raise ValueError('EchoNet endpoint-only masks require --semi; intermediate masks are not GT.')
    validate_dataset_splits(args.data_path,
                            (opt.train_split, opt.val_split) if training else (opt.test_split,))
    for name in ('sam_ckpt', 'dino_config', 'dino_weights', 'dino_lora_weights', 'load_path'):
        value = getattr(args, name, '')
        if value:
            setattr(args, name, project_path(value))
    required = ['sam_ckpt']
    if args.modelname == 'SharedGroundedMemSAM' or args.enable_box_prompt:
        required.extend(('dino_config', 'dino_weights'))
        if args.dino_use_lora:
            required.append('dino_lora_weights')
    if args.load_path:
        required.append('load_path')
    for name in required:
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f'--{name}: {getattr(args, name)}')
    for name in ('save_path', 'result_path', 'tensorboard_path'):
        setattr(opt, name, project_path(getattr(opt, name)) + '/')
    if args.output_dir:
        output = Path(project_path(args.output_dir))
        opt.save_path = str(output / 'checkpoints') + '/'
        opt.result_path = str(output / 'results') + '/'
        opt.tensorboard_path = str(output / 'tensorboard') + '/'
    for name in ('log_dir', 'test_log_dir', 'clinical_output_dir'):
        if getattr(args, name, ''):
            setattr(args, name, project_path(getattr(args, name)))
