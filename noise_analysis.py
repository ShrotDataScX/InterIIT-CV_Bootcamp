"""Task 1: noise rate and 10x10 noisy-vs-clean confusion matrix on the 45k training split."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DATA_DIR = "."

# ---- load -------------------------------------------------------------------
df = pd.read_csv(f"{DATA_DIR}/labels.csv").set_index("index")
train_idx = np.loadtxt(f"{DATA_DIR}/train_idx.txt", dtype=np.int64)
df = df.loc[train_idx]                       # the 45,000 training images only

classes_path = f"{DATA_DIR}/classes.txt"
if os.path.exists(classes_path):
    classes = [l.strip() for l in open(classes_path) if l.strip()]
else:
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

clean = df["clean_label"].values
noisy = df["noisy_label"].values

# ---- noise rate -------------------------------------------------------------
n_wrong = int((clean != noisy).sum())
print(f"Mislabeled: {n_wrong} / {len(df)}  ->  noise rate = {n_wrong / len(df):.4f}")

# ---- confusion matrix (rows = clean/true class, cols = noisy/annotated class)
cm = confusion_matrix(clean, noisy, labels=range(10))
off = cm.copy()
np.fill_diagonal(off, 0)
# % of each clean class that was flipped to each other class
flip_pct = 100 * off / cm.sum(axis=1, keepdims=True)

# ---- most confused pairs (directed: clean -> noisy) -------------------------
order = np.dstack(np.unravel_index(np.argsort(-off, axis=None), off.shape))[0]
print("\nTop 10 flips (clean -> noisy):")
print(f"{'clean':>11} -> {'noisy':<11} {'count':>6} {'% of class':>11} {'% of all noise':>15}")
for i, j in order[:10]:
    print(f"{classes[i]:>11} -> {classes[j]:<11} {off[i, j]:>6} "
          f"{flip_pct[i, j]:>10.1f}% {100 * off[i, j] / n_wrong:>14.1f}%")

# ---- plot -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
            xticklabels=classes, yticklabels=classes)
axes[0].set_title("Counts (all samples)")

sns.heatmap(flip_pct, annot=True, fmt=".1f", cmap="Reds", ax=axes[1],
            mask=np.eye(10, dtype=bool),
            xticklabels=classes, yticklabels=classes,
            cbar_kws={"label": "% of clean class"})
axes[1].set_title("Label flips only (diagonal masked), % of each clean class")

for ax in axes:
    ax.set_xlabel("Noisy label (annotator)")
    ax.set_ylabel("Clean label (true)")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("noise_confusion.png", dpi=150)
plt.show()
