from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """Resolve repository-relative paths independently of the working directory."""
    if value is None:
        return None
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else PROJECT_ROOT / path).resolve())


def resolve_argument_paths(args, *names):
    for name in names:
        setattr(args, name, project_path(getattr(args, name, None)))


def validate_dataset_splits(dataset_path, splits):
    root = Path(dataset_path)
    for split in splits:
        videos = root / "videos" / split
        annotations = root / "annotations" / split
        if not videos.is_dir() or not annotations.is_dir():
            raise FileNotFoundError(f"Expected dataset directories: {videos} and {annotations}")
        files = sorted(videos.glob("*.npy"))
        if not files:
            raise ValueError(f"No .npy videos found in {videos}")
        missing = [annotations / f"{video.stem}.npz" for video in files
                   if not (annotations / f"{video.stem}.npz").is_file()]
        if missing:
            examples = "\n  ".join(str(path) for path in missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} video annotations:\n  {examples}")
