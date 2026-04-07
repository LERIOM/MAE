from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

DATASET_HANDLE = "ronithrr/jasper-japanese-x-sri-lankan-textile-image-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_dataset(dataset_handle: str = DATASET_HANDLE) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is required to download the dataset automatically. "
            "Install it with `pip install kagglehub` or pass `dataset_dir` "
            "to `create_dataloaders`."
        ) from exc

    return Path(kagglehub.dataset_download(dataset_handle))


def resolve_dataset_root(dataset_dir: str | Path | None = None) -> Path:
    dataset_root = Path(dataset_dir).expanduser() if dataset_dir else download_dataset()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    return dataset_root


def find_image_files(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    image_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No image files found under {root}")
    return image_paths


def build_transform(image_size: int | tuple[int, int] = 256) -> transforms.Compose:
    size = (image_size, image_size) if isinstance(image_size, int) else image_size
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
        ]
    )


class TextileImageDataset(Dataset):
    def __init__(
        self,
        dataset_root: str | Path,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.image_paths = find_image_files(self.dataset_root)
        self.transform = transform or build_transform()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image)


def create_dataloaders(
    dataset_dir: str | Path | None = None,
    batch_size: int = 32,
    image_size: int | tuple[int, int] = 256,
    val_split: float = 0.2,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> tuple[DataLoader, DataLoader]:
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1.")

    dataset_root = resolve_dataset_root(dataset_dir)
    dataset = TextileImageDataset(
        dataset_root=dataset_root,
        transform=build_transform(image_size),
    )

    if len(dataset) < 2:
        raise ValueError("At least two images are required to create train/val splits.")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    if train_size == 0:
        raise ValueError("Validation split is too large for the dataset size.")

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=drop_last,
        **common_loader_args,
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        drop_last=False,
        **common_loader_args,
    )

    return train_loader, val_loader


get_dataloaders = create_dataloaders

