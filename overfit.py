import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from src.data import create_datasets
from src.model import build_mae
from src.visualisation import save_epoch_visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit test for the MAE model on a tiny fixed subset."
    )
    parser.add_argument("--subset-size", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs/overfit")
    parser.add_argument("--visualization-every", type=int, default=5)
    parser.add_argument("--num-visualization-images", type=int, default=5)
    parser.add_argument("--visualization-seed", type=int, default=42)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument(
        "--norm-pix-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fixed-mask-seed", type=int, default=123)
    parser.add_argument(
        "--random-mask-training",
        action="store_true",
        help="Disable the fixed training mask and sample a new random mask at every step.",
    )
    return parser.parse_args()


def select_subset(dataset, subset_size, seed):
    if subset_size <= 0:
        raise ValueError(f"subset_size must be >= 1, got {subset_size}")
    if subset_size > len(dataset):
        raise ValueError(
            f"subset_size ({subset_size}) cannot exceed dataset size ({len(dataset)})"
        )

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator).tolist()[:subset_size]
    return Subset(dataset, indices), indices


def build_loader(dataset, batch_size, shuffle, num_workers):
    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=persistent_workers,
    )


def get_fixed_visualization_images(dataset, num_images, device):
    num_images = min(num_images, len(dataset))
    images = torch.stack([dataset[idx] for idx in range(num_images)])
    return images.to(device)


def set_mask_seed(seed, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def train_one_epoch_overfit(
    model,
    train_loader,
    optimizer,
    device,
    fixed_mask_seed=None,
):
    model.train()
    total_loss = 0.0

    for images in train_loader:
        images = images.to(device, non_blocking=True)

        if fixed_mask_seed is not None:
            set_mask_seed(fixed_mask_seed, device)

        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(images, return_loss=True)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate_one_epoch_overfit(
    model,
    val_loader,
    device,
    fixed_mask_seed=None,
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images in val_loader:
            images = images.to(device, non_blocking=True)

            if fixed_mask_seed is not None:
                set_mask_seed(fixed_mask_seed, device)

            loss, _ = model(images, return_loss=True)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def save_losses_plot(train_losses, val_losses, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Overfit Test Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_dataset, _ = create_datasets(
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
    )

    subset, subset_indices = select_subset(
        train_dataset,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    default_batch_size = 1 if args.subset_size == 1 else min(4, max(1, args.subset_size // 2))
    batch_size = args.batch_size or default_batch_size
    batch_size = min(batch_size, len(subset))
    fixed_mask_seed = None if args.random_mask_training else args.fixed_mask_seed

    train_loader = build_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = build_loader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir = output_dir / "visualisations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "subset_indices.txt").write_text(
        "\n".join(str(idx) for idx in subset_indices) + "\n",
        encoding="utf-8",
    )

    print(f"Subset size: {len(subset)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(train_loader)}")
    print(f"Total optimizer steps: {len(train_loader) * args.epochs}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"norm_pix_loss: {args.norm_pix_loss}")
    print(
        "Training mask: "
        + (
            f"fixed (seed={fixed_mask_seed})"
            if fixed_mask_seed is not None
            else "random at every step"
        )
    )
    print(f"Saved subset indices to: {output_dir / 'subset_indices.txt'}")

    model = build_mae(
        img_size=args.image_size,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    train_losses = []
    val_losses = []

    visualization_images = get_fixed_visualization_images(
        subset,
        num_images=args.num_visualization_images,
        device=device,
    )
    save_epoch_visualization(
        model,
        visualization_images,
        epoch=0,
        output_dir=visualization_dir,
        seed=args.visualization_seed,
    )

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_loss = train_one_epoch_overfit(
            model,
            train_loader,
            optimizer,
            device,
            fixed_mask_seed=fixed_mask_seed,
        )
        val_loss = validate_one_epoch_overfit(
            model,
            val_loader,
            device,
            fixed_mask_seed=fixed_mask_seed,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"  train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if (epoch + 1) % args.visualization_every == 0:
            save_epoch_visualization(
                model,
                visualization_images,
                epoch=epoch + 1,
                output_dir=visualization_dir,
                seed=args.visualization_seed,
            )

    save_losses_plot(
        train_losses,
        val_losses,
        output_dir / "loss_curve.png",
    )


if __name__ == "__main__":
    main()
