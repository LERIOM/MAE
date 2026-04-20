from contextlib import contextmanager
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _extract_images(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def get_visualization_batch(dataloader, device, num_images=5):
    images = _extract_images(next(iter(dataloader)))
    images = images[:num_images]

    if images.shape[0] == 0:
        raise ValueError("visualization batch is empty")

    return images.to(device)


def build_masked_images(model, images, mask):
    patches = model.patchify(images).clone()
    patches = patches * (1.0 - mask.unsqueeze(-1).to(dtype=patches.dtype))
    return model.unpatchify(patches)


def denormalize_predicted_patches(model, images, pred_patches):
    if not model.norm_pix_loss:
        return pred_patches

    target_patches = model.patchify(images)
    mean = target_patches.mean(dim=-1, keepdim=True)
    var = target_patches.var(dim=-1, keepdim=True)
    pred_patches = pred_patches * (var + 1.0e-6).sqrt() + mean
    return pred_patches


def build_reconstructed_images(model, images, pred_patches, mask):
    pred_patches = denormalize_predicted_patches(model, images, pred_patches)
    original_patches = model.patchify(images)
    mask = mask.unsqueeze(-1).to(dtype=pred_patches.dtype)
    blended_patches = original_patches * (1.0 - mask) + pred_patches * mask
    return model.unpatchify(blended_patches)


def _to_display_image(image_tensor):
    image_tensor = image_tensor.detach().cpu().clamp(0.0, 1.0)
    return image_tensor.permute(1, 2, 0).numpy()


@contextmanager
def _visualization_context(model):
    was_training = model.training
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        model.eval()
        yield
    finally:
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        if was_training:
            model.train()


def _run_visualization_forward(model, images, seed):
    model_device = next(model.parameters()).device
    images = images.to(model_device)

    torch.manual_seed(seed)
    if model_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    loss, _, pred_patches, mask, _, _ = model(
        images,
        return_aux=True,
        return_loss=True,
    )
    masked_images = build_masked_images(model, images, mask)
    reconstructed_images = build_reconstructed_images(
        model,
        images,
        pred_patches,
        mask,
    )
    return images, loss, masked_images, reconstructed_images


def save_epoch_visualization(
    model,
    images,
    epoch,
    output_dir="outputs/visualisations",
    seed=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with _visualization_context(model):
        with torch.no_grad():
            images, loss, masked_images, reconstructed_images = _run_visualization_forward(
                model,
                images,
                seed,
            )

    num_images = images.shape[0]
    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=3,
        figsize=(9, 3 * num_images),
        squeeze=False,
    )

    column_titles = ["Original", "Masquee", "Reconstruite"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    for row in range(num_images):
        row_images = (
            images[row],
            masked_images[row],
            reconstructed_images[row],
        )
        for col, image in enumerate(row_images):
            axes[row, col].imshow(_to_display_image(image))
            axes[row, col].axis("off")

    fig.suptitle(f"Epoch {epoch:03d} | loss={loss.item():.4f}")
    fig.tight_layout()

    output_path = output_dir / f"epoch_{epoch:03d}.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_mask_ratio_sweep_visualization(
    model,
    images,
    epoch,
    output_dir="outputs/visualisations",
    seed=42,
    mask_ratios=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
    image_index=0,
):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "mask_ratio"):
        raise ValueError("model.encoder.mask_ratio is required for mask ratio sweep visualization")

    if images.shape[0] == 0:
        raise ValueError("visualization batch is empty")
    if not 0 <= image_index < images.shape[0]:
        raise ValueError(
            f"image_index must be in [0, {images.shape[0] - 1}], got {image_index}"
        )

    mask_ratios = tuple(float(mask_ratio) for mask_ratio in mask_ratios)
    if len(mask_ratios) == 0:
        raise ValueError("mask_ratios must contain at least one value")
    for mask_ratio in mask_ratios:
        if not 0.0 <= mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_mask_ratio = model.encoder.mask_ratio
    sweep_image = images[image_index : image_index + 1]
    sweep_results = []

    try:
        with _visualization_context(model):
            with torch.no_grad():
                for mask_ratio in mask_ratios:
                    model.encoder.mask_ratio = mask_ratio
                    (
                        sweep_image,
                        loss,
                        masked_images,
                        reconstructed_images,
                    ) = _run_visualization_forward(
                        model,
                        sweep_image,
                        seed,
                    )
                    sweep_results.append(
                        (
                            mask_ratio,
                            loss.item(),
                            masked_images[0],
                            reconstructed_images[0],
                        )
                    )

        num_ratios = len(sweep_results)
        fig = plt.figure(figsize=(2.2 * (num_ratios + 1), 5.0))
        grid = fig.add_gridspec(2, num_ratios + 1, width_ratios=[1.15] + [1.0] * num_ratios)

        original_axis = fig.add_subplot(grid[:, 0])
        original_axis.imshow(_to_display_image(sweep_image[0]))
        original_axis.set_title("Original")
        original_axis.axis("off")

        for index, (mask_ratio, loss_value, masked_image, reconstructed_image) in enumerate(
            sweep_results,
            start=1,
        ):
            masked_axis = fig.add_subplot(grid[0, index])
            masked_axis.imshow(_to_display_image(masked_image))
            masked_axis.set_title(f"mask={mask_ratio:.2f}")
            masked_axis.axis("off")

            reconstructed_axis = fig.add_subplot(grid[1, index])
            reconstructed_axis.imshow(_to_display_image(reconstructed_image))
            reconstructed_axis.set_xlabel(f"loss={loss_value:.4f}")
            reconstructed_axis.axis("off")

            if index == 1:
                masked_axis.set_ylabel("Masquee")
                reconstructed_axis.set_ylabel("Reconstruite")

        fig.suptitle(
            f"Epoch {epoch:03d} | image={image_index} | sweep mask ratio",
        )
        fig.tight_layout()

        output_path = output_dir / f"epoch_{epoch:03d}_mask_ratio_sweep_img_{image_index:02d}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    finally:
        model.encoder.mask_ratio = original_mask_ratio

    return output_path
