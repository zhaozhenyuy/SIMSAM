from dataclasses import dataclass


@dataclass
class VideoConfig:
    data_path: str
    save_path: str
    result_path: str
    tensorboard_path: str
    data_subpath: str
    eval_mode: str
    epochs: int
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    workers: int = 4
    batch_size: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.02
    classes: int = 2
    img_size: int = 256
    crop: object = None
    eval_freq: int = 1
    save_freq: int = 25
    device: str = "cuda"
    mode: str = "train"
    visual: bool = False
    semi: bool = True
    pre_trained: bool = False
    load_path: str = ""
    save_path_code: str = "_"
    modelname: str = "SharedGroundedMemSAM"


def get_config(task="CAMUS_Video_Full"):
    if task == "CAMUS_Video_Full":
        return VideoConfig(
            data_path="data/CAMUS_public",
            save_path="checkpoints/camus",
            result_path="results/camus",
            tensorboard_path="runs/camus",
            data_subpath="camus",
            eval_mode="camus",
            epochs=150,
        )
    if task == "EchoNet_Video":
        return VideoConfig(
            data_path="data/EchoNet/echocycle",
            save_path="checkpoints/echonet",
            result_path="results/echonet",
            tensorboard_path="runs/echonet",
            data_subpath="camus",
            eval_mode="echonet",
            epochs=30,
        )
    raise ValueError(
        f"Unsupported task {task!r}. Choose 'CAMUS_Video_Full' or 'EchoNet_Video'."
    )
