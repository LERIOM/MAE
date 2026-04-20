from contextlib import nullcontext

import torch
import PIL.Image as Image
from tqdm import tqdm
from torchvision import transforms

from src.visualisation import get_visualization_batch, save_epoch_visualization


def _autocast_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def apply_mask(images, mask_ratio, seed=None):
    patch_size = 16
    batch_size, channels, height, width = images.shape
    num_patches = (height // patch_size) * (width // patch_size)
    num_masked = int(mask_ratio * num_patches)
    masked_images = images.clone()
    for i in range(batch_size):
        if seed is not None:
            torch.manual_seed(seed)
        random_permutation = torch.randperm(num_patches)

        mask_indices = random_permutation[:num_masked]
        visible_indices = random_permutation[num_masked:]
        visible_indices_sorted = torch.sort(visible_indices)[0]
        for idx in mask_indices:
            row = (idx // (width // patch_size)) * patch_size
            col = (idx % (width // patch_size)) * patch_size
            masked_images[i, :, row:row + patch_size, col:col + patch_size] = 0

    return masked_images, visible_indices_sorted


def test_masking():

    image = Image.open("data/raw/japanese_textiles/image_0005.png").convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0)
    masked_image, visible_indices = apply_mask(image_tensor, mask_ratio=0.95, seed=42)
    masked_image_pil = transforms.ToPILImage()(masked_image.squeeze(0))
    masked_image_pil.save("masked_image.jpg")
    print(f"Visible indices: {visible_indices}")

def train_one_epoch(
    model,
    train_loader,
    optimizer,
    device,
    scaler,
    use_amp,
    lr_scheduler=None,
):
    model.train()

    total_loss = 0.0

    progress_bar = tqdm.tqdm(train_loader, desc="Training", unit="batch")

    for idx, images in enumerate(progress_bar):
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast_context(device, use_amp):
            loss, _ = model(images, return_loss=True)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if lr_scheduler is not None:
            lr_scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": total_loss / (idx + 1)})

    return total_loss / len(train_loader)

def validate_one_epoch(model, val_loader, device, use_amp):
    model.eval()

    total_loss = 0.0

    progress_bar = tqdm.tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for idx, images in enumerate(progress_bar):
            images = images.to(device, non_blocking=True)

            with _autocast_context(device, use_amp):
                loss, _ = model(images, return_loss=True)
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": total_loss / (idx + 1)})

    return total_loss / len(val_loader)

def train(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs,
    save_visualizations=True,
    visualization_dir="outputs/visualisations",
    num_visualization_images=5,
    visualization_seed=42,
    visualization_every=5,
):
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train_losses = []
    val_losses = []

    visualization_images = None
    if save_visualizations:
        if visualization_every <= 0:
            raise ValueError(
                f"visualization_every must be >= 1, got {visualization_every}"
            )
        visualization_images = get_visualization_batch(
            val_loader,
            device=device,
            num_images=num_visualization_images,
        )
        save_epoch_visualization(
            model,
            visualization_images,
            epoch=0,
            output_dir=visualization_dir,
            seed=visualization_seed,
        )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler, use_amp)
        val_loss = validate_one_epoch(model, val_loader, device, use_amp)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if visualization_images is not None and (epoch + 1) % visualization_every == 0:
            save_epoch_visualization(
                model,
                visualization_images,
                epoch=epoch + 1,
                output_dir=visualization_dir,
                seed=visualization_seed,
            )

    return train_losses, val_losses

if __name__ == "__main__":
    test_masking()    
