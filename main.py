import torch
import os
from src.data import create_dataloaders
from src.model import AutoEncoder
from src.train import train

def main():
    # Config
    IMG_SIZE = 256
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CP_PATH = "checkpoints/autoencoder.pth"

    print(f"Device: {DEVICE}")

    # Dataset
    train_loader, val_loader = create_dataloaders(
        batch_size = BATCH_SIZE,
        image_size = IMG_SIZE,
        val_split = 0.2,
        pin_memory = DEVICE.type == "cuda",
    )

    print(f"Train: {len(train_loader.dataset)} images | Val: {len(val_loader.dataset)} images")

    # Model
    model = AutoEncoder(img_size=IMG_SIZE)
    model.to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training parameters : {total_params:,}")

    # Training
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
    )

    # Save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict" : model.state_dict(),
        "train_losses" : train_losses,
        "val_losses" : val_losses,
    }, CP_PATH)
    print(f"Model saved at {CP_PATH}")


if __name__ == "__main__":
    main()