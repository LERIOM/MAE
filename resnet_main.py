import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data import create_classification_dataloaders, default_resnet_transform


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ResNet18 classifier for comparison with the MAE encoder."
    )
    parser.add_argument("--dataset-name", default="flowers")
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet-pretrained ResNet18 weights.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-prefix", default="resnet_classifier")
    parser.add_argument("--loss-curve-path", default=None)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, freeze_backbone, backbone_lr, classifier_lr, weight_decay):
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if not name.startswith("fc."):
                parameter.requires_grad = False
        return torch.optim.AdamW(
            model.fc.parameters(),
            lr=classifier_lr,
            weight_decay=weight_decay,
        )

    backbone_params = []
    classifier_params = []
    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            classifier_params.append(parameter)
        else:
            backbone_params.append(parameter)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": classifier_params, "lr": classifier_lr},
        ],
        weight_decay=weight_decay,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, freeze_backbone):
    model.train()
    if freeze_backbone:
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()
        model.layer4.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100.0 * correct / max(total, 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), 100.0 * correct / max(total, 1)


def save_curves(train_losses, val_losses, train_accs, val_accs, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, train_losses, label="Train Loss")
    axes[0].plot(epochs, val_losses, label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("ResNet Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, train_accs, label="Train Accuracy")
    axes[1].plot(epochs, val_accs, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("ResNet Accuracy")
    axes[1].legend()
    axes[1].grid()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    transform = default_resnet_transform(image_size=args.image_size)
    train_loader, val_loader, class_names = create_classification_dataloaders(
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        image_size=args.image_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        train_transform=transform,
        val_transform=transform,
        max_classes=args.max_classes,
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")

    weights = ResNet18_Weights.DEFAULT if args.pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        freeze_backbone=args.freeze_backbone,
        backbone_lr=args.backbone_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{args.checkpoint_prefix}_best.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            freeze_backbone=args.freeze_backbone,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "pretrained": args.pretrained,
                    "freeze_backbone": args.freeze_backbone,
                },
                best_checkpoint_path,
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%"
        )

    loss_curve_path = args.loss_curve_path
    if loss_curve_path is None:
        loss_curve_path = DEFAULT_OUTPUT_DIR / f"{args.checkpoint_prefix}_curves.png"
    save_curves(train_losses, val_losses, train_accs, val_accs, loss_curve_path)
    print(f"Best val_acc={best_val_acc:.2f}%")
    print(f"Saved best checkpoint: {best_checkpoint_path}")
    print(f"Saved curves: {loss_curve_path}")


if __name__ == "__main__":
    main()
