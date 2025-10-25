import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(data_path: str, batch_size: int = 32, shuffle: bool = True, image_size: int = 224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    dataset = datasets.ImageFolder(os.path.expanduser(data_path), transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, len(dataset.classes)
