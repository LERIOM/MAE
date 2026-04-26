import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from src.data import create_resnet_dataloaders


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    train_loader, val_loader, class_names = create_resnet_dataloaders(
        dataset_name="animals10",
        batch_size=32,
        image_size=224,
        val_split=0.2,
        seed=42,
        num_workers=0,
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    num_epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                running_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = running_val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        val_acc = 100.0 * correct / max(total, 1)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train_loss={epoch_train_loss:.4f} | val_loss={epoch_val_loss:.4f} | val_acc={val_acc:.2f}%"
        )

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig("outputs/resnet_loss_curve.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
