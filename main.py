from src.data import create_dataloaders
from src.model import build_mae
from src.train import train
import torch
import matplotlib.pyplot as plt


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(dataset_name="textile")

    model = build_mae()

    num_epochs = 200
    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        device,
        num_epochs=num_epochs,
        save_visualizations=True,
        visualization_dir="outputs/visualisations",
        num_visualization_images=5,
        visualization_seed=42,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/loss_curve.png")
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
