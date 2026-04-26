import argparse
from contextlib import nullcontext
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

from src.data import create_classification_dataloaders
from src.model import build_mae


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CHECKPOINT_DIR = DEFAULT_OUTPUT_DIR / "checkpoints"
CHECKPOINT_STATE_DICT_KEYS = ("model_state_dict", "state_dict", "model")


class MAEClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim, num_classes, dropout=0.0):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )

    def forward(self, images):
        latent, _, _, _ = self.encoder(images)
        features = latent.mean(dim=1)
        return self.classifier(features)


class TorchvisionViTMAEEncoder(nn.Module):
    def __init__(self, img_size, patch_size=16, in_chans=3, mask_ratio=0.0):
        super().__init__()
        if patch_size != 16:
            raise ValueError(
                f"torchvision ViT-B/16 checkpoints require patch_size=16, got {patch_size}"
            )
        if in_chans != 3:
            raise ValueError(
                f"torchvision ViT-B/16 checkpoints require in_chans=3, got {in_chans}"
            )
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.vit = vit_b_16(weights=None, image_size=img_size)
        self.encoder_dim = self.vit.hidden_dim

    def random_masking(self, x):
        batch_size, num_patches, dim = x.shape
        num_keep = int(num_patches * (1 - self.mask_ratio))
        if num_keep <= 0:
            raise ValueError(
                f"mask_ratio={self.mask_ratio} keeps no visible patches for N={num_patches}"
            )
        if num_keep == num_patches:
            ids = torch.arange(num_patches, device=x.device).unsqueeze(0).expand(batch_size, -1)
            mask = torch.zeros(batch_size, num_patches, device=x.device)
            return x, mask, ids, ids

        noise = torch.rand(batch_size, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]

        x_vis = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, dim),
        )

        mask = torch.ones(batch_size, num_patches, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_vis, mask, ids_restore, ids_keep

    def forward(self, images):
        x = self.vit._process_input(images)
        patch_pos_embed = self.vit.encoder.pos_embedding[:, 1:, :]
        x = x + patch_pos_embed

        x_vis, mask, ids_restore, ids_keep = self.random_masking(x)

        cls_token = self.vit.class_token + self.vit.encoder.pos_embedding[:, :1, :]
        cls_token = cls_token.expand(images.shape[0], -1, -1)

        x_vis = torch.cat([cls_token, x_vis], dim=1)
        x_vis = self.vit.encoder.dropout(x_vis)
        x_vis = self.vit.encoder.layers(x_vis)
        x_vis = self.vit.encoder.ln(x_vis)
        return x_vis[:, 1:, :], mask, ids_restore, ids_keep


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a classifier on top of a MAE encoder from src.model."
    )
    parser.add_argument(
        "--checkpoint-path",
        required=True,
        help=(
            "Path to a trained MAE checkpoint. Supports src.model.build_mae() "
            "checkpoints and the older encoder.vit.* checkpoints."
        ),
    )
    parser.add_argument("--dataset-name", default="flowers")
    parser.add_argument("--root-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Defaults to the image size inferred from the MAE checkpoint.",
    )
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.0,
        help="Use 0.0 for classification unless you intentionally want masked inputs.",
    )
    parser.add_argument(
        "--norm-pix-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--encoder-heads",
        type=int,
        default=8,
        help="Must match the MAE checkpoint architecture.",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=8,
        help="Must match the MAE checkpoint architecture.",
    )
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--classifier-dropout", type=float, default=0.0)
    parser.add_argument(
        "--freeze-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze the MAE encoder for a linear-probe style evaluation.",
    )
    parser.add_argument("--encoder-lr", type=float, default=1e-5)
    parser.add_argument("--classifier-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--checkpoint-prefix", default="mae_classifier")
    parser.add_argument("--loss-curve-path", default=None)
    return parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in CHECKPOINT_STATE_DICT_KEYS:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value

        if any(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint

    raise TypeError(
        "checkpoint must be a state_dict or contain one of: "
        + ", ".join(CHECKPOINT_STATE_DICT_KEYS)
    )


def strip_state_dict_prefix(state_dict, prefix):
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix) :]: value for key, value in state_dict.items()}
    return state_dict


def is_torchvision_vit_mae_state_dict(state_dict):
    return any(key.startswith("encoder.vit.") for key in state_dict)


def validate_local_mae_state_dict(state_dict, checkpoint_path):
    if is_torchvision_vit_mae_state_dict(state_dict):
        required_keys = (
            "encoder.vit.conv_proj.weight",
            "encoder.vit.encoder.pos_embedding",
            "encoder.vit.encoder.ln.weight",
        )
        missing_keys = [key for key in required_keys if key not in state_dict]
        if missing_keys:
            raise ValueError(
                f"{checkpoint_path} looks like a torchvision ViT MAE checkpoint, "
                f"but is missing keys: {missing_keys}"
            )
        return

    required_keys = (
        "encoder.patch_embed.proj.weight",
        "encoder.pos_embed",
        "decoder.pos_embed",
        "decoder.mask_token",
        "decoder.decoder_pred.weight",
    )
    missing_keys = [key for key in required_keys if key not in state_dict]
    if missing_keys:
        raise ValueError(
            f"{checkpoint_path} does not look like a src.model.MAE checkpoint. "
            f"Missing keys: {missing_keys}"
        )


def load_mae_state_dict(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(checkpoint)
    state_dict = strip_state_dict_prefix(state_dict, "module.")
    validate_local_mae_state_dict(state_dict, checkpoint_path)
    return state_dict


def filter_prefixed_state_dict(state_dict, prefix):
    return {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def count_module_list_depth(state_dict, prefix):
    indices = set()
    for key in state_dict:
        if key.startswith(prefix):
            suffix = key[len(prefix) :]
            index_text = suffix.split(".", 1)[0]
            if index_text.isdigit():
                indices.add(int(index_text))

    if not indices:
        raise ValueError(f"could not infer module depth from keys starting with {prefix!r}")

    return max(indices) + 1


def infer_mlp_ratio(state_dict, prefix, dim):
    key = f"{prefix}.0.mlp.net.0.weight"
    if key not in state_dict:
        return 4.0
    return state_dict[key].shape[0] / dim


def infer_mae_config_from_state_dict(
    state_dict,
    mask_ratio,
    norm_pix_loss,
    encoder_heads,
    decoder_heads,
    dropout,
):
    patch_weight = state_dict["encoder.patch_embed.proj.weight"]
    encoder_dim, in_chans, patch_h, patch_w = patch_weight.shape
    if patch_h != patch_w:
        raise ValueError(f"expected square patches, got {patch_h}x{patch_w}")

    num_patches = state_dict["encoder.pos_embed"].shape[1]
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches ({num_patches}) must form a square grid")

    decoder_dim = state_dict["decoder.pos_embed"].shape[2]

    return {
        "img_size": grid_size * patch_h,
        "patch_size": patch_h,
        "in_chans": in_chans,
        "encoder_dim": encoder_dim,
        "encoder_depth": count_module_list_depth(state_dict, "encoder.blocks."),
        "encoder_heads": encoder_heads,
        "decoder_dim": decoder_dim,
        "decoder_depth": count_module_list_depth(state_dict, "decoder.blocks."),
        "decoder_heads": decoder_heads,
        "mlp_ratio": infer_mlp_ratio(state_dict, "encoder.blocks", encoder_dim),
        "mask_ratio": mask_ratio,
        "dropout": dropout,
        "norm_pix_loss": norm_pix_loss,
    }


def infer_torchvision_vit_config_from_state_dict(state_dict, mask_ratio):
    patch_weight = state_dict["encoder.vit.conv_proj.weight"]
    encoder_dim, in_chans, patch_h, patch_w = patch_weight.shape
    if patch_h != patch_w:
        raise ValueError(f"expected square patches, got {patch_h}x{patch_w}")

    pos_embed = state_dict["encoder.vit.encoder.pos_embedding"]
    num_patches = pos_embed.shape[1] - 1
    grid_size = int(num_patches ** 0.5)
    if grid_size * grid_size != num_patches:
        raise ValueError(f"num_patches ({num_patches}) must form a square grid")

    return {
        "checkpoint_format": "torchvision_vit_mae",
        "img_size": grid_size * patch_h,
        "patch_size": patch_h,
        "in_chans": in_chans,
        "encoder_dim": encoder_dim,
        "num_patches": num_patches,
        "mask_ratio": mask_ratio,
    }


def build_torchvision_vit_encoder(state_dict, checkpoint_path, args):
    encoder_config = infer_torchvision_vit_config_from_state_dict(
        state_dict,
        mask_ratio=args.mask_ratio,
    )
    encoder = TorchvisionViTMAEEncoder(
        img_size=encoder_config["img_size"],
        patch_size=encoder_config["patch_size"],
        in_chans=encoder_config["in_chans"],
        mask_ratio=args.mask_ratio,
    )
    encoder_state_dict = filter_prefixed_state_dict(state_dict, "encoder.")
    encoder.load_state_dict(encoder_state_dict, strict=args.strict_checkpoint)

    print(f"Loaded torchvision ViT MAE checkpoint: {checkpoint_path}")
    print(
        "Encoder config: "
        + ", ".join(f"{key}={value}" for key, value in encoder_config.items())
    )
    return encoder, encoder_config


def build_local_mae_encoder(state_dict, checkpoint_path, args):
    model_config = infer_mae_config_from_state_dict(
        state_dict,
        mask_ratio=args.mask_ratio,
        norm_pix_loss=args.norm_pix_loss,
        encoder_heads=args.encoder_heads,
        decoder_heads=args.decoder_heads,
        dropout=args.dropout,
    )
    mae = build_mae(**model_config)
    mae.load_state_dict(state_dict, strict=args.strict_checkpoint)

    model_config["checkpoint_format"] = "src_model_mae"
    print(f"Loaded src.model MAE checkpoint: {checkpoint_path}")
    print(
        "MAE config: "
        + ", ".join(f"{key}={value}" for key, value in model_config.items())
    )
    return mae.encoder, model_config


def build_trained_encoder(checkpoint_path, args):
    state_dict = load_mae_state_dict(checkpoint_path)
    if is_torchvision_vit_mae_state_dict(state_dict):
        return build_torchvision_vit_encoder(state_dict, checkpoint_path, args)
    return build_local_mae_encoder(state_dict, checkpoint_path, args)


def build_optimizer(model, freeze_encoder, encoder_lr, classifier_lr, weight_decay):
    if freeze_encoder:
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        return torch.optim.AdamW(
            model.classifier.parameters(),
            lr=classifier_lr,
            weight_decay=weight_decay,
        )

    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.classifier.parameters(), "lr": classifier_lr},
        ],
        weight_decay=weight_decay,
    )


def autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def run_train_epoch(model, dataloader, criterion, optimizer, scaler, device, use_amp, freeze_encoder):
    model.train()
    if freeze_encoder:
        model.encoder.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(dataloader), 100.0 * correct / max(total, 1)


def evaluate(model, dataloader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast_context(device, use_amp):
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
    axes[0].set_title("MAE Classifier Loss")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(epochs, train_accs, label="Train Accuracy")
    axes[1].plot(epochs, val_accs, label="Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("MAE Classifier Accuracy")
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

    encoder, encoder_config = build_trained_encoder(args.checkpoint_path, args)
    image_size = args.image_size or encoder_config["img_size"]

    train_loader, val_loader, class_names = create_classification_dataloaders(
        root_dir=args.root_dir,
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        image_size=image_size,
        val_split=args.val_split,
        seed=args.seed,
        num_workers=args.num_workers,
        max_classes=args.max_classes,
    )
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train size: {len(train_loader.dataset)}")
    print(f"Val size: {len(val_loader.dataset)}")
    print("Encoder: frozen" if args.freeze_encoder else "Encoder: fine-tuned")

    model = MAEClassifier(
        encoder=encoder,
        encoder_dim=encoder_config["encoder_dim"],
        num_classes=len(class_names),
        dropout=args.classifier_dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        model,
        freeze_encoder=args.freeze_encoder,
        encoder_lr=args.encoder_lr,
        classifier_lr=args.classifier_lr,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0.0

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / f"{args.checkpoint_prefix}_best.pth"

    for epoch in range(args.epochs):
        train_loss, train_acc = run_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            freeze_encoder=args.freeze_encoder,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, use_amp)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_config": encoder_config,
                    "class_names": class_names,
                    "epoch": epoch + 1,
                    "val_acc": val_acc,
                    "freeze_encoder": args.freeze_encoder,
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
    if args.epochs > 0:
        print(f"Best val_acc={best_val_acc:.2f}%")
        print(f"Saved best checkpoint: {best_checkpoint_path}")
    else:
        print("No epochs were run; no classifier checkpoint was saved.")
    print(f"Saved curves: {loss_curve_path}")


if __name__ == "__main__":
    main()
