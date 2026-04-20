from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models import ViT_B_16_Weights, vit_b_16
from torchvision.models.vision_transformer import interpolate_embeddings

from src.data import create_dataloaders
from src.model import MAE, MAEDecoder
from src.train import train_one_epoch, validate_one_epoch
from src.visualisation import (
    get_visualization_batch,
    save_epoch_visualization,
    save_mask_ratio_sweep_visualization,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_TORCH_CACHE_DIR = PROJECT_ROOT / ".torch_cache"


class PretrainedViTMAEEncoder(nn.Module):
    def __init__(
        self,
        img_size=256,
        patch_size=16,
        in_chans=3,
        mask_ratio=0.5,
        pretrained=True,
        weights=ViT_B_16_Weights.DEFAULT,
        cache_dir=DEFAULT_TORCH_CACHE_DIR,
    ):
        super().__init__()

        if patch_size != 16:
            raise ValueError(
                f"This pretrained encoder only supports patch_size=16, got {patch_size}"
            )
        if in_chans != 3:
            raise ValueError(
                f"This pretrained encoder only supports in_chans=3, got {in_chans}"
            )
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        vit = vit_b_16(weights=None, image_size=img_size)
        if pretrained:
            self._load_pretrained_weights(
                vit=vit,
                weights=weights,
                cache_dir=cache_dir,
                image_size=img_size,
                patch_size=patch_size,
            )

        self.vit = vit
        self.encoder_dim = vit.hidden_dim

    def _load_pretrained_weights(
        self,
        vit,
        weights,
        cache_dir,
        image_size,
        patch_size,
    ):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_dir))

        state_dict = weights.get_state_dict(progress=True)
        state_dict = interpolate_embeddings(
            image_size=image_size,
            patch_size=patch_size,
            model_state=state_dict,
            reset_heads=True,
        )
        vit.load_state_dict(state_dict, strict=False)

    def random_masking(self, x):
        """
        x: [B, N, D]
        returns:
            x_vis: [B, N_keep, D]
            mask: [B, N] with 0=visible, 1=masked
            ids_restore: [B, N]
            ids_keep: [B, N_keep]
        """
        B, N, D = x.shape
        n_keep = int(N * (1 - self.mask_ratio))
        if n_keep <= 0:
            raise ValueError(
                f"mask_ratio={self.mask_ratio} keeps no visible patches for N={N}"
            )

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]

        x_vis = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).expand(-1, -1, D),
        )

        mask = torch.ones(B, N, device=x.device)
        mask[:, :n_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_vis, mask, ids_restore, ids_keep

    def forward(self, imgs):
        x = self.vit._process_input(imgs)
        patch_pos_embed = self.vit.encoder.pos_embedding[:, 1:, :]
        x = x + patch_pos_embed

        x_vis, mask, ids_restore, ids_keep = self.random_masking(x)

        cls_token = self.vit.class_token + self.vit.encoder.pos_embedding[:, :1, :]
        cls_token = cls_token.expand(imgs.shape[0], -1, -1)

        x_vis = torch.cat([cls_token, x_vis], dim=1)
        x_vis = self.vit.encoder.dropout(x_vis)
        x_vis = self.vit.encoder.layers(x_vis)
        x_vis = self.vit.encoder.ln(x_vis)

        latent = x_vis[:, 1:, :]
        return latent, mask, ids_restore, ids_keep


def build_pretrained_mae(
    img_size=256,
    patch_size=16,
    in_chans=3,
    mask_ratio=0.5,
    pretrained=True,
    norm_pix_loss=True,
    decoder_dim=384,
    decoder_depth=4,
    decoder_heads=6,
    mlp_ratio=4.0,
    dropout=0.0,
    cache_dir=DEFAULT_TORCH_CACHE_DIR,
):
    encoder = PretrainedViTMAEEncoder(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        mask_ratio=mask_ratio,
        pretrained=pretrained,
        cache_dir=cache_dir,
    )

    decoder = MAEDecoder(
        num_patches=encoder.num_patches,
        encoder_dim=encoder.encoder_dim,
        decoder_dim=decoder_dim,
        patch_size=patch_size,
        in_chans=in_chans,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    )

    return MAE(encoder, decoder, norm_pix_loss=norm_pix_loss)


def build_pretrained_optimizer(
    model,
    encoder_lr=1e-5,
    decoder_lr=1e-4,
    weight_decay=0.05,
):
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.decoder.parameters(), "lr": decoder_lr},
        ],
        betas=(0.9, 0.95),
        weight_decay=weight_decay,
    )


def build_pretrained_scheduler(
    optimizer,
    steps_per_epoch,
    num_epochs,
    warmup_epochs=10,
    warmup_start_factor=0.1,
):
    if steps_per_epoch <= 0:
        raise ValueError(f"steps_per_epoch must be >= 1, got {steps_per_epoch}")
    if num_epochs <= 0:
        raise ValueError(f"num_epochs must be >= 1, got {num_epochs}")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError(
            f"warmup_start_factor must be in (0, 1], got {warmup_start_factor}"
        )

    total_steps = steps_per_epoch * num_epochs
    if total_steps <= 1:
        return None, 0, total_steps

    warmup_steps = min(warmup_epochs * steps_per_epoch, total_steps - 1)
    cosine_steps = total_steps - warmup_steps

    if warmup_steps == 0:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=0.0,
        )
        return scheduler, warmup_steps, total_steps

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=0.0,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )
    return scheduler, warmup_steps, total_steps


def format_learning_rates(optimizer):
    group_names = ("encoder_lr", "decoder_lr")
    lr_parts = []

    for index, param_group in enumerate(optimizer.param_groups):
        group_name = group_names[index] if index < len(group_names) else f"group_{index}_lr"
        lr_parts.append(f"{group_name}={param_group['lr']:.2e}")

    return " | ".join(lr_parts)


def build_mask_ratio_sweep_values(start=0.5, end=0.95, step=0.05):
    if not 0.0 <= start < 1.0:
        raise ValueError(f"mask sweep start must be in [0, 1), got {start}")
    if not 0.0 <= end < 1.0:
        raise ValueError(f"mask sweep end must be in [0, 1), got {end}")
    if end < start:
        raise ValueError(f"mask sweep end ({end}) must be >= start ({start})")
    if step <= 0.0:
        raise ValueError(f"mask sweep step must be > 0, got {step}")

    mask_ratios = []
    current = start
    while current <= end + 1.0e-9:
        mask_ratios.append(round(current, 4))
        current += step

    return tuple(mask_ratios)


def train_pretrained_mae(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    encoder_lr=1e-5,
    decoder_lr=1e-4,
    weight_decay=0.05,
    save_visualizations=True,
    visualization_dir="outputs/pretrained_visualisations",
    num_visualization_images=5,
    visualization_seed=42,
    visualization_every=5,
    save_mask_ratio_sweep=False,
    mask_ratio_sweep_start=0.5,
    mask_ratio_sweep_end=0.95,
    mask_ratio_sweep_step=0.05,
    mask_ratio_sweep_image_index=0,
    use_scheduler=True,
    scheduler_warmup_epochs=10,
    scheduler_warmup_start_factor=0.3,
):
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer = build_pretrained_optimizer(
        model,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=weight_decay,
    )
    scheduler = None
    warmup_steps = 0
    total_steps = len(train_loader) * num_epochs
    mask_ratio_sweep_values = None

    if use_scheduler:
        scheduler, warmup_steps, total_steps = build_pretrained_scheduler(
            optimizer,
            steps_per_epoch=len(train_loader),
            num_epochs=num_epochs,
            warmup_epochs=scheduler_warmup_epochs,
            warmup_start_factor=scheduler_warmup_start_factor,
        )

    train_losses = []
    val_losses = []
    visualization_images = None
    needs_visualization_batch = save_visualizations or save_mask_ratio_sweep

    if save_mask_ratio_sweep:
        mask_ratio_sweep_values = build_mask_ratio_sweep_values(
            start=mask_ratio_sweep_start,
            end=mask_ratio_sweep_end,
            step=mask_ratio_sweep_step,
        )

    if needs_visualization_batch:
        if visualization_every <= 0:
            raise ValueError(
                f"visualization_every must be >= 1, got {visualization_every}"
            )
        visualization_images = get_visualization_batch(
            val_loader,
            device=device,
            num_images=num_visualization_images,
        )
        if save_visualizations:
            save_epoch_visualization(
                model,
                visualization_images,
                epoch=0,
                output_dir=visualization_dir,
                seed=visualization_seed,
            )
        if save_mask_ratio_sweep:
            save_mask_ratio_sweep_visualization(
                model,
                visualization_images,
                epoch=0,
                output_dir=visualization_dir,
                seed=visualization_seed,
                mask_ratios=mask_ratio_sweep_values,
                image_index=mask_ratio_sweep_image_index,
            )

    if scheduler is None:
        print("Scheduler: disabled")
    else:
        print(
            "Scheduler: warmup + cosine decay "
            f"(total_steps={total_steps}, warmup_steps={warmup_steps})"
        )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  {format_learning_rates(optimizer)}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            use_amp,
            lr_scheduler=scheduler,
        )
        val_loss = validate_one_epoch(
            model,
            val_loader,
            device,
            use_amp,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if visualization_images is not None and (epoch + 1) % visualization_every == 0:
            if save_visualizations:
                save_epoch_visualization(
                    model,
                    visualization_images,
                    epoch=epoch + 1,
                    output_dir=visualization_dir,
                    seed=visualization_seed,
                )
            if save_mask_ratio_sweep:
                save_mask_ratio_sweep_visualization(
                    model,
                    visualization_images,
                    epoch=epoch + 1,
                    output_dir=visualization_dir,
                    seed=visualization_seed,
                    mask_ratios=mask_ratio_sweep_values,
                    image_index=mask_ratio_sweep_image_index,
                )
            torch.save(model.state_dict(), Path("outputs/checkpoints") / f"pretrained_mae_epoch_{epoch + 1}.pth")

    return train_losses, val_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(dataset_name="flowers",
                                                   batch_size=64)

    model = build_pretrained_mae(
        img_size=256,
        patch_size=16,
        mask_ratio=0.75,
        pretrained=True,
        norm_pix_loss=True,
        decoder_dim=384,
        decoder_depth=4,
        decoder_heads=6,
    )

    num_epochs = 200
    encoder_lr = 1e-5
    decoder_lr = 1e-4
    scheduler_warmup_epochs = 10
    scheduler_warmup_start_factor = 0.1

    train_losses, val_losses = train_pretrained_mae(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        encoder_lr=encoder_lr,
        decoder_lr=decoder_lr,
        weight_decay=0.05,
        save_visualizations=True,
        visualization_dir="outputs/pretrained_visualisations/flowers",
        num_visualization_images=5,
        visualization_seed=42,
        visualization_every=5,
        save_mask_ratio_sweep=True,
        mask_ratio_sweep_start=0.5,
        mask_ratio_sweep_end=0.95,
        mask_ratio_sweep_step=0.05,
        mask_ratio_sweep_image_index=0,
        use_scheduler=True,
        scheduler_warmup_epochs=scheduler_warmup_epochs,
        scheduler_warmup_start_factor=scheduler_warmup_start_factor,
    )

    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pretrained ViT MAE Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_dir / "pretrained_loss_curve.png")
    plt.close()


if __name__ == "__main__":
    main()
