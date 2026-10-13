"""Standard cross-entropy training of ResNet-18 for 60 epochs, with Task 2 logging.

Usage:
    python train_baseline.py                    # noisy labels (Task 1 noisy baseline + Task 2 dynamics)
    python train_baseline.py --label clean      # Task 1 clean reference model ONLY
    python train_baseline.py --label synthetic  # Task 5 baseline

Outputs (in the current directory):
    t2_dynamics.csv                        only for --label noisy
    epoch_preds_<label>/preds_epoch_XX.csv test predictions after every epoch (id,pred)
    t1_<label>_final.csv                   test predictions after the last epoch
    resnet18_<label>_final.pt              final weights

The training target and the "main" validation accuracy use the same label type.
Clean labels are used only to *measure* (subset split, clean val accuracy), never in the loss.
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from data import get_loaders, get_transforms
from model import ResNet18

EPOCHS = 60
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

# Dataset returns (img, noisy, clean, synthetic)
LABEL_POS = {"noisy": 1, "clean": 2, "synthetic": 3}
DYNAMICS_HEADER = "epoch,train_acc_correct,train_acc_mislabeled,val_acc_noisy,val_acc_clean"


def evaluate(model, loader, device):
    """One pass over the validation set; returns accuracy (%) against each label type."""
    model.eval()
    correct = {k: 0 for k in LABEL_POS}
    total = 0
    with torch.no_grad():
        for batch in loader:
            pred = model(batch[0].to(device, non_blocking=True)).argmax(1)
            for k, pos in LABEL_POS.items():
                correct[k] += (pred == batch[pos].to(device, non_blocking=True)).sum().item()
            total += pred.size(0)
    return {k: 100.0 * v / total for k, v in correct.items()}


def load_test_tensor(path):
    """test_images.npy (10000,32,32,3) uint8 -> normalized float tensor (10000,3,32,32).
    Uses the validation transforms (ToTensor + Normalize, no augmentation). Loaded once."""
    imgs = np.load(path)
    tf = get_transforms(train=False)
    return torch.stack([tf(Image.fromarray(im)) for im in imgs])


def predict_test(model, test_x, device, batch_size=500):
    """Argmax predictions for every test image; returns a numpy int array of length 10000."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_x), batch_size):
            preds.append(model(test_x[i:i + batch_size].to(device)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def save_preds(preds, path):
    pd.DataFrame({"id": np.arange(len(preds)), "pred": preds}).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=LABEL_POS, default="noisy")
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    pos = LABEL_POS[args.label]
    track = args.label == "noisy"          # Task 2 logging only for the noisy run

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, train_loader, val_loader = get_loaders(args.data_dir, BATCH_SIZE, args.num_workers)
    test_x = load_test_tensor(os.path.join(args.data_dir, "test_images.npy"))

    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    epoch_dir = f"epoch_preds_{args.label}"
    os.makedirs(epoch_dir, exist_ok=True)
    if track:
        with open("t2_dynamics.csv", "w") as f:       # fresh file + header; rows appended per epoch
            f.write(DYNAMICS_HEADER + "\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = hit = total = 0
        c_hit = c_n = w_hit = w_n = 0                  # correctly-labeled / mislabeled subsets
        for batch in train_loader:
            x = batch[0].to(device, non_blocking=True)
            y = batch[pos].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            ok = out.argmax(1) == y                    # agreement with the *training target*
            loss_sum += loss.item() * y.size(0)
            hit += ok.sum().item()
            total += y.size(0)
            if track:
                wrong = (batch[1] != batch[2]).to(device, non_blocking=True)   # measurement only
                c_hit += ok[~wrong].sum().item()
                c_n += (~wrong).sum().item()
                w_hit += ok[wrong].sum().item()
                w_n += wrong.sum().item()
        scheduler.step()

        val = evaluate(model, val_loader, device)
        msg = (f"Epoch {epoch:02d}/{EPOCHS} | lr {optimizer.param_groups[0]['lr']:.4f} | "
               f"train loss {loss_sum / total:.4f} | train acc {100 * hit / total:.2f}% | "
               f"val acc ({args.label} labels) {val[args.label]:.2f}%")
        if track:
            tr_c, tr_w = 100 * c_hit / c_n, 100 * w_hit / max(w_n, 1)
            with open("t2_dynamics.csv", "a") as f:
                f.write(f"{epoch},{tr_c:.4f},{tr_w:.4f},{val['noisy']:.4f},{val['clean']:.4f}\n")
            msg += (f" | train correct {tr_c:.2f}% | train mislabeled {tr_w:.2f}% "
                    f"| val clean {val['clean']:.2f}%")
        print(msg, flush=True)

        preds = predict_test(model, test_x, device)
        save_preds(preds, os.path.join(epoch_dir, f"preds_epoch_{epoch:02d}.csv"))

    save_preds(preds, f"t1_{args.label}_final.csv")    # predictions from the final epoch
    torch.save(model.state_dict(), f"resnet18_{args.label}_final.pt")


if __name__ == "__main__":
    main()
