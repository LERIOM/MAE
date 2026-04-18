import torch
import torch.nn as nn
import PIL.Image as Image
from tqdm import tqdm
from torchvision import transforms

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

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    model.to(device)

    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc="Training", unit="batch")

    for idx, images in enumerate(progress_bar):
        masked_images, visible_indices = apply_mask(images, mask_ratio=0.75)
        masked_images = masked_images.to(device)
        images = images.to(device)

        outputs = model((masked_images, visible_indices))
        loss = criterion(outputs, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": total_loss / (idx + 1)})

    return total_loss / len(train_loader)

def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    model.to(device)

    total_loss = 0.0

    progress_bar = tqdm(val_loader, desc="Validation", unit="batch")

    with torch.no_grad():
        for idx, images in enumerate(progress_bar):
            masked_images, visible_indices = apply_mask(images, mask_ratio=0.75)
            masked_images = masked_images.to(device)
            images = images.to(device)

            outputs = model((masked_images, visible_indices))
            loss = criterion(outputs, images)
            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": total_loss / (idx + 1)})

    return total_loss / len(val_loader)

def train(model, train_loader, val_loader, device, num_epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return train_losses, val_losses

if __name__ == "__main__":
    test_masking()