import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models.resnet import ResNet
from data_loader import get_dataloader
from utils.metrics import accuracy
from config import Config


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, total_acc = 0, 0
    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += accuracy(outputs, labels)
    return total_loss / len(dataloader), total_acc / len(dataloader)


if __name__ == "__main__":
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader, num_classes = get_dataloader(cfg.data_path, cfg.batch_size, image_size=cfg.image_size)
    model = ResNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    print(f"Training ResNet on dataset at: {cfg.data_path} with {num_classes} classes")

    for epoch in range(cfg.num_epochs):
        loss, acc = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] | Loss: {loss:.4f} | Accuracy: {acc:.4f}")
