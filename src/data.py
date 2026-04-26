from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

try:
    import kagglehub
except ImportError:  # pragma: no cover - optional dependency
    kagglehub = None


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_FLOWERS_DIR = DEFAULT_RAW_DIR / "flowers"
DEFAULT_ANIMALS10_DIR = DEFAULT_RAW_DIR / "animals10" / "raw-img"
DEFAULT_TEXTILE_DIRS = (
    DEFAULT_RAW_DIR / "japanese_textiles",
    DEFAULT_RAW_DIR / "sri_lankan_textiles",
)
FLOWERS_VAL_DIR_NAMES = ("valid", "val", "validation")


def default_transform(image_size=256):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def default_resnet_transform(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def list_image_files(root_dir=DEFAULT_RAW_DIR):
    root_dir = Path(root_dir)

    if not root_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root_dir}")

    image_paths = sorted(
        path for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise ValueError(f"No image files found in: {root_dir}")

    return image_paths


def normalize_dataset_name(dataset_name):
    normalized_name = str(dataset_name).strip().lower()
    aliases = {
        "animals10": "animals10",
        "animals": "animals10",
        "textile": "textile",
        "textiles": "textile",
        "flower": "flowers",
        "flowers": "flowers",
    }

    if normalized_name not in aliases:
        supported = ", ".join(sorted(aliases))
        raise ValueError(
            f"Unsupported dataset_name={dataset_name!r}. Supported values: {supported}"
        )

    return aliases[normalized_name]


def resolve_default_root_dir(dataset_name):
    if dataset_name == "animals10":
        if DEFAULT_ANIMALS10_DIR.exists():
            return DEFAULT_ANIMALS10_DIR

        if kagglehub is None:
            raise ImportError(
                "kagglehub is required to auto-download Animals-10. "
                "Install it with `pip install kagglehub` or pass root_dir explicitly."
            )

        try:
            downloaded_root = Path(kagglehub.dataset_download("alessiocorrado99/animals10"))
        except Exception:  # pragma: no cover - network/auth dependent
            raise RuntimeError(
                "Failed to download Animals-10 from Kaggle. "
                "Make sure Kaggle access is configured, or pass root_dir explicitly."
            ) from None
        candidate_dirs = (
            downloaded_root / "raw-img",
            downloaded_root / "animals10" / "raw-img",
            downloaded_root / "data" / "raw-img",
        )

        for candidate in candidate_dirs:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            "Animals-10 downloaded but no `raw-img` directory was found under "
            f"{downloaded_root}"
        )

    if dataset_name == "flowers":
        return DEFAULT_FLOWERS_DIR
    if dataset_name == "textile":
        return DEFAULT_RAW_DIR
    raise ValueError(f"Unsupported dataset_name={dataset_name!r}")


def find_existing_roots(root_dirs):
    return [Path(root_dir) for root_dir in root_dirs if Path(root_dir).exists()]


def list_image_files_from_roots(root_dirs):
    image_paths = []

    for root_dir in root_dirs:
        image_paths.extend(list_image_files(root_dir))

    if not image_paths:
        roots = ", ".join(str(Path(root_dir)) for root_dir in root_dirs)
        raise ValueError(f"No image files found in: {roots}")

    return sorted(image_paths)


def resolve_textile_image_paths(root_dir=None):
    if root_dir is not None:
        return list_image_files(root_dir)

    textile_roots = find_existing_roots(DEFAULT_TEXTILE_DIRS)
    if not textile_roots:
        raise FileNotFoundError(
            "Default textile directories not found. Pass root_dir explicitly or "
            f"create one of: {', '.join(str(path) for path in DEFAULT_TEXTILE_DIRS)}"
        )

    return list_image_files_from_roots(textile_roots)


def resolve_split_dir(root_dir, candidates):
    root_dir = Path(root_dir)

    for candidate in candidates:
        split_dir = root_dir / candidate
        if split_dir.is_dir():
            return split_dir

    return None


def normalize_max_classes(max_classes):
    if max_classes is None:
        return None
    if not isinstance(max_classes, int):
        raise TypeError(
            f"max_classes must be an integer or None, got {type(max_classes).__name__}"
        )
    if max_classes <= 0:
        raise ValueError(f"max_classes must be >= 1, got {max_classes}")
    return max_classes


def class_name_sort_key(class_name):
    class_name = str(class_name)
    if class_name.isdigit():
        return (0, int(class_name), "")
    return (1, 0, class_name)


def list_class_dirs(root_dir):
    root_dir = Path(root_dir)
    class_dirs = [path for path in root_dir.iterdir() if path.is_dir()]

    if not class_dirs:
        raise ValueError(f"No class directories found in: {root_dir}")

    return sorted(class_dirs, key=lambda path: class_name_sort_key(path.name))


def select_class_names(class_names, max_classes=None):
    max_classes = normalize_max_classes(max_classes)
    class_names = list(class_names)

    if max_classes is None:
        return class_names
    if max_classes > len(class_names):
        raise ValueError(
            f"max_classes={max_classes} exceeds the number of available classes "
            f"({len(class_names)})"
        )

    return class_names[:max_classes]


def list_image_files_for_class_names(root_dir, class_names):
    root_dir = Path(root_dir)
    image_paths = []

    for class_name in class_names:
        class_dir = root_dir / str(class_name)
        if class_dir.is_dir():
            image_paths.extend(list_image_files(class_dir))

    if not image_paths:
        raise ValueError(
            f"No image files found for classes {list(class_names)} in: {root_dir}"
        )

    return sorted(image_paths)


def resolve_flowers_class_names(root_dir, max_classes=None):
    train_dir = Path(root_dir) / "train"
    class_names = [class_dir.name for class_dir in list_class_dirs(train_dir)]
    return select_class_names(class_names, max_classes=max_classes)


def resolve_flowers_split_paths(root_dir, max_classes=None):
    root_dir = Path(root_dir)
    train_dir = root_dir / "train"
    val_dir = resolve_split_dir(root_dir, FLOWERS_VAL_DIR_NAMES)

    if not train_dir.is_dir():
        raise FileNotFoundError(f"Flowers train split not found: {train_dir}")
    if val_dir is None:
        candidates = ", ".join(FLOWERS_VAL_DIR_NAMES)
        raise FileNotFoundError(
            f"Flowers validation split not found in {root_dir}. "
            f"Expected one of: {candidates}"
        )

    class_names = resolve_flowers_class_names(root_dir, max_classes=max_classes)
    train_paths = list_image_files_for_class_names(train_dir, class_names)
    val_paths = list_image_files_for_class_names(val_dir, class_names)
    return train_paths, val_paths


class TextileDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = list(image_paths)
        self.transform = transform

        if not self.image_paths:
            raise ValueError("Dataset requires at least one image")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


def split_image_paths(image_paths, val_split=0.2, seed=42):
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    num_images = len(image_paths)
    num_val = int(num_images * val_split)

    if num_val == 0 or num_val == num_images:
        raise ValueError(
            f"val_split={val_split} produces an empty split for {num_images} images"
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_images, generator=generator).tolist()

    val_indices = indices[:num_val]
    train_indices = indices[num_val:]

    train_paths = [image_paths[idx] for idx in train_indices]
    val_paths = [image_paths[idx] for idx in val_indices]

    return train_paths, val_paths


def create_datasets(
    root_dir=None,
    image_size=256,
    val_split=0.2,
    seed=42,
    train_transform=None,
    val_transform=None,
    dataset_name="textile",
    max_classes=None,
):
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "flowers":
        if root_dir is None:
            root_dir = resolve_default_root_dir(dataset_name)
        train_paths, val_paths = resolve_flowers_split_paths(
            root_dir,
            max_classes=max_classes,
        )
    else:
        if root_dir is None and dataset_name == "animals10":
            root_dir = resolve_default_root_dir(dataset_name)
        image_paths = resolve_textile_image_paths(root_dir=root_dir)
        train_paths, val_paths = split_image_paths(
            image_paths,
            val_split=val_split,
            seed=seed,
        )

    if train_transform is None:
        train_transform = default_transform(image_size=image_size)
    if val_transform is None:
        val_transform = default_transform(image_size=image_size)

    train_dataset = TextileDataset(train_paths, transform=train_transform)
    val_dataset = TextileDataset(val_paths, transform=val_transform)

    return train_dataset, val_dataset


def create_dataloaders(
    root_dir=None,
    batch_size=32,
    image_size=256,
    val_split=0.2,
    seed=42,
    num_workers=0,
    pin_memory=None,
    drop_last=False,
    train_transform=None,
    val_transform=None,
    dataset_name="textile",
    max_classes=None,
):
    train_dataset, val_dataset = create_datasets(
        root_dir=root_dir,
        dataset_name=dataset_name,
        image_size=image_size,
        val_split=val_split,
        seed=seed,
        train_transform=train_transform,
        val_transform=val_transform,
        max_classes=max_classes,
    )

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader


def create_resnet_dataloaders(
    root_dir=None,
    dataset_name="animals10",
    batch_size=32,
    image_size=224,
    val_split=0.2,
    seed=42,
    num_workers=0,
    pin_memory=None,
):
    if root_dir is None:
        root_dir = resolve_default_root_dir(normalize_dataset_name(dataset_name))

    transform = default_resnet_transform(image_size=image_size)
    dataset = datasets.ImageFolder(root=str(root_dir), transform=transform)

    if len(dataset) < 2:
        raise ValueError("ResNet pipeline requires at least 2 images to build train/val splits")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError(
            f"val_split={val_split} leaves no training samples for dataset size {len(dataset)}"
        )

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, dataset.classes


def dataset_summary(root_dir=None, dataset_name="textile", max_classes=None):
    dataset_name = normalize_dataset_name(dataset_name)

    if dataset_name == "flowers":
        if root_dir is None:
            root_dir = resolve_default_root_dir(dataset_name)
        root_dir = Path(root_dir)
        class_names = resolve_flowers_class_names(root_dir, max_classes=max_classes)

        summary = {}
        split_dirs = {
            "train": root_dir / "train",
            "valid": resolve_split_dir(root_dir, FLOWERS_VAL_DIR_NAMES),
            "test": root_dir / "test",
        }

        for split_name, split_dir in split_dirs.items():
            if split_dir is not None and split_dir.is_dir():
                summary[split_name] = len(
                    list_image_files_for_class_names(split_dir, class_names)
                )

        return summary

    if root_dir is None:
        textile_roots = find_existing_roots(DEFAULT_TEXTILE_DIRS)
        return {
            textile_root.name: len(list_image_files(textile_root))
            for textile_root in textile_roots
        }

    root_dir = Path(root_dir)
    image_paths = list_image_files(root_dir)

    summary = {}
    for image_path in image_paths:
        split_name = image_path.parent.name
        summary[split_name] = summary.get(split_name, 0) + 1

    return summary


if __name__ == "__main__":
    train_loader, val_loader = create_dataloaders()
    print(f"Dataset summary: {dataset_summary()}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
