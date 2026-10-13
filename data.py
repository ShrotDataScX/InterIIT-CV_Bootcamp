"""Dataset + DataLoaders for the CIFAR-10N bootcamp task."""
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def get_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CIFAR10N(Dataset):
    """Returns (image, noisy_label, clean_label, synthetic_label).

    If return_index=True, the original CIFAR-10 training index is appended
    as a 5th element (needed for Task 3 per-image scores).

    NOTE: clean_label is returned only so you can *measure* things (confusion
    matrix, Task 2 curves). Never use it in the loss, filtering, or model
    selection for Tasks 2-5.
    """

    def __init__(self, images_path, labels_path, idx_path, train=True,
                 return_index=False):
        images = np.load(images_path)                      # (50000, 32, 32, 3) uint8
        df = pd.read_csv(labels_path).sort_values("index").reset_index(drop=True)
        assert (df["index"].values == np.arange(len(df))).all(), \
            "labels.csv indices must be 0..N-1 so row i matches image i"
        assert len(images) == len(df)

        self.indices = np.loadtxt(idx_path, dtype=np.int64)  # subset to use
        self.images = images[self.indices]
        self.noisy = df["noisy_label"].values[self.indices].astype(np.int64)
        self.clean = df["clean_label"].values[self.indices].astype(np.int64)
        self.synth = df["synthetic_label"].values[self.indices].astype(np.int64)

        self.transform = get_transforms(train)
        self.return_index = return_index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img = self.transform(Image.fromarray(self.images[i]))
        out = (img, int(self.noisy[i]), int(self.clean[i]), int(self.synth[i]))
        if self.return_index:
            out += (int(self.indices[i]),)
        return out


def get_loaders(data_dir=".", batch_size=128, num_workers=4, return_index=False):
    p = lambda f: f"{data_dir}/{f}"
    train_ds = CIFAR10N(p("train_images.npy"), p("labels.csv"), p("train_idx.txt"),
                        train=True, return_index=return_index)
    val_ds = CIFAR10N(p("train_images.npy"), p("labels.csv"), p("val_idx.txt"),
                      train=False, return_index=return_index)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_ds, val_ds, train_loader, val_loader


if __name__ == "__main__":
    train_ds, val_ds, train_loader, val_loader = get_loaders(".")
    print(len(train_ds), len(val_ds))                       # 45000 5000
    x, noisy, clean, synth = next(iter(train_loader))
    print(x.shape, noisy.shape, clean.shape, synth.shape)   # [128,3,32,32], [128] x3
    print("noise rate (train):", (train_ds.noisy != train_ds.clean).mean())
