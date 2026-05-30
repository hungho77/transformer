"""Vision datasets for ViT training."""
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def build_vision_dataset(name: str, train: bool = True, data_dir: str = "data"):
    """Return a torchvision dataset. ``name`` in {cifar10, fake}.

    'fake' yields random 32x32 images (no download) for smoke tests / offline CI.
    """
    name = name.lower()
    if name == "cifar10":
        tfm = transforms.Compose([
            transforms.RandomCrop(32, padding=4) if train else transforms.Lambda(lambda im: im),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda im: im),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        return datasets.CIFAR10(data_dir, train=train, download=True, transform=tfm)
    if name == "fake":
        return datasets.FakeData(
            size=512, image_size=(3, 32, 32), num_classes=10, transform=transforms.ToTensor()
        )
    raise ValueError(f"Unknown vision dataset '{name}'.")
