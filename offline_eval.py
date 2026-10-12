"""Offline evaluation using only the images you already captured (no webcam, no partner).

Models compared (all with out-of-fold scores, so every image is scored by a model
that never trained on it):
  LDA raw pixels | LR raw pixels | SVM raw pixels | SVM HOG

Comparing "SVM raw" with "SVM HOG" isolates the effect of the features.
Comparing "LR raw" with "SVM raw" isolates the effect of the classifier.

Outputs: offline_roc.png, offline_oof_scores.png, offline_learning_curve.png,
         offline_hog.png, offline_hard_cases.png (weakest client and strongest imposter
         per model) and a printed summary table with a scale-free margin column.
"""
import glob
import re
import time
from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve
from sklearn.metrics import roc_curve, roc_auc_score

CLIENT_GLOB = "faces_client/*.png"
IMPOSTER_GLOB = "faces_imposter/*.png"
SIZE = (100, 100)
N_FOLDS = 5

# Use the values you chose from your validation grid searches
LR_C = 1e-8
SVM_C = 0.1


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p)]


def load(pattern):
    imgs, names = [], []
    for f in sorted(glob.glob(pattern), key=natural_key):   # student2 before student10
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if im is not None:
            imgs.append(cv2.resize(im, SIZE))
            names.append(f)
    return np.array(imgs), names


def hog_feat(im, visualize=False):
    return hog(im, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", visualize=visualize)


client, client_files = load(CLIENT_GLOB)
imposter, imposter_files = load(IMPOSTER_GLOB)
print(f"Loaded {len(client)} client, {len(imposter)} imposter images")
imgs = np.concatenate((client, imposter))
files = np.array(client_files + imposter_files)
y = np.concatenate((np.ones(len(client), int), np.zeros(len(imposter), int)))

X_raw = imgs.reshape(len(imgs), -1).astype(float)
X_hog = np.array([hog_feat(im) for im in imgs])

models = {
    "LDA (raw)": (LinearDiscriminantAnalysis(), X_raw, lambda im: im.reshape(1, -1).astype(float)),
    "LR (raw)": (LogisticRegression(C=LR_C, max_iter=1000), X_raw,
                 lambda im: im.reshape(1, -1).astype(float)),
    "SVM (raw)": (SVC(kernel="linear", C=SVM_C), X_raw, lambda im: im.reshape(1, -1).astype(float)),
    "SVM (HOG)": (SVC(kernel="linear", C=SVM_C), X_hog, lambda im: hog_feat(im).reshape(1, -1)),
}
colors = {"LDA (raw)": "tab:blue", "LR (raw)": "tab:purple",
          "SVM (raw)": "tab:gray", "SVM (HOG)": "tab:red"}

# Folds are contiguous blocks within each class (StratifiedKFold without shuffling, images in
# capture order). Consecutive webcam frames are near-duplicates, so random folds would leak
# almost identical images into train and test; blocks keep a whole stretch of the capture
# session (e.g. one lighting condition) held out together.
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=False)

results = {}
for name, (est, X, make_x) in models.items():
    oof = cross_val_predict(est, X, y, cv=cv, method="decision_function")
    fpr, tpr, _ = roc_curve(y, oof)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    pred = oof > 0
    # hardest cases: the lowest-scoring client and the highest-scoring imposter
    cl_idx, im_idx = np.where(y == 1)[0], np.where(y == 0)[0]
    worst_cl = cl_idx[np.argmin(oof[cl_idx])]
    worst_im = im_idx[np.argmax(oof[im_idx])]
    # scale-free margin: gap between the closest client and imposter, in units of the
    # spread of all this model's scores. Positive = every client outranks every imposter.
    margin = (oof[worst_cl] - oof[worst_im]) / np.std(oof)
    t0 = time.perf_counter()
    est.fit(X, y)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        est.decision_function(make_x(imgs[0]))
    pred_ms = (time.perf_counter() - t0) / 20 * 1000
    results[name] = dict(
        oof=oof, fpr=fpr, tpr=tpr, auc=roc_auc_score(y, oof), eer=(fpr[i] + fnr[i]) / 2,
        acc=(pred == y).mean(), far=pred[y == 0].mean(), frr=(~pred[y == 1]).mean(),
        dim=X.shape[1], fit_s=fit_s, pred_ms=pred_ms,
        margin=margin, worst_cl=worst_cl, worst_im=worst_im)

# ------------------------------------------------------------------ summary table
print(f"\nOut-of-fold results ({N_FOLDS} contiguous folds, default decision threshold 0)")
print(f"{'model':<11}{'features':>9}{'AUC':>8}{'EER':>7}{'acc':>7}{'FAR':>7}{'FRR':>7}"
      f"{'margin':>8}{'fit (s)':>9}{'pred (ms)':>11}")
for n, r in results.items():
    print(f"{n:<11}{r['dim']:>9}{r['auc']:>8.3f}{r['eer']:>7.2f}{r['acc']:>7.2f}"
          f"{r['far']:>7.2f}{r['frr']:>7.2f}{r['margin']:>8.2f}{r['fit_s']:>9.3f}"
          f"{r['pred_ms']:>11.2f}")
print("pred (ms) is per image and includes feature extraction.")
print("margin = (lowest client score - highest imposter score) / std of all scores; "
      "positive means every client outranks every imposter, larger is better.")

print("\nHardest cases per model (out-of-fold decision scores)")
for n, r in results.items():
    print(f"{n:<11} weakest client    : {files[r['worst_cl']]}  ({r['oof'][r['worst_cl']]:+.2f})")
    print(f"{'':<11} strongest imposter: {files[r['worst_im']]}  ({r['oof'][r['worst_im']]:+.2f})")
for label, key in [("imposter", "worst_im"), ("client", "worst_cl")]:
    counts = Counter(files[r[key]] for r in results.values())
    name, k = counts.most_common(1)[0]
    if k > 1:
        print(f"{name} is the hardest {label} image for {k} of {len(results)} models")

# ------------------------------------------------------------------ ROC
fig1, ax = plt.subplots(figsize=(5.5, 5))
for n, r in results.items():
    ax.plot(r["fpr"], r["tpr"], color=colors[n], label=f"{n} (AUC {r['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k:", linewidth=0.8)
ax.set_xlabel("FAR (imposters accepted)")
ax.set_ylabel("client accept rate (1 - FRR)")
ax.set_title("ROC from out-of-fold scores")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig("offline_roc.png", dpi=150)

# ------------------------------------------------------------------ score distributions
fig2, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (n, r) in zip(axes, results.items()):
    ax.hist(r["oof"][y == 0], bins=15, alpha=0.7, color="tab:red", label="imposter")
    ax.hist(r["oof"][y == 1], bins=15, alpha=0.7, color="tab:green", label="client")
    ax.axvline(0, color="k", linestyle="--")
    ax.set_title(n)
    ax.set_xlabel("out-of-fold decision score")
    ax.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig("offline_oof_scores.png", dpi=150)

# ------------------------------------------------------------------ learning curves
fig3, ax = plt.subplots(figsize=(6, 4))
for name, (est, X, _) in models.items():
    sizes, _, val = learning_curve(est, X, y, cv=cv, scoring="roc_auc",
                                   train_sizes=np.linspace(0.2, 1.0, 5),
                                   shuffle=True, random_state=0, error_score=np.nan)
    m, s = np.nanmean(val, axis=1), np.nanstd(val, axis=1)
    ax.plot(sizes, m, "o-", color=colors[name], label=name)
    ax.fill_between(sizes, m - s, m + s, color=colors[name], alpha=0.15)
ax.set_xlabel("number of training images")
ax.set_ylabel("held-out ROC AUC")
ax.set_title("Learning curves")
ax.set_ylim(0, 1.05)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig("offline_learning_curve.png", dpi=150)

# ------------------------------------------------------------------ what HOG sees
fig4, axes = plt.subplots(2, 2, figsize=(6, 6))
for row, (label, im) in enumerate([("client", client[0]), ("imposter", imposter[0])]):
    _, hog_img = hog_feat(im, visualize=True)
    axes[row, 0].imshow(im, cmap="gray")
    axes[row, 0].set_title(f"{label} (input)")
    axes[row, 1].imshow(hog_img, cmap="gray")
    axes[row, 1].set_title(f"{label} (HOG)")
for a in axes.ravel():
    a.axis("off")
fig4.tight_layout()
fig4.savefig("offline_hog.png", dpi=150)

# ------------------------------------------------------------------ hardest cases
fig5, axes = plt.subplots(2, len(results), figsize=(3 * len(results), 6.6), squeeze=False)
for col, (n, r) in enumerate(results.items()):
    for row, (key, label) in enumerate([("worst_cl", "weakest client"),
                                        ("worst_im", "strongest imposter")]):
        i = r[key]
        fname = files[i].replace("\\", "/").split("/")[-1]
        axes[row, col].imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_title(f"{n}\n{label}\n{fname}  score {r['oof'][i]:+.2f}", fontsize=8)
        axes[row, col].axis("off")
fig5.tight_layout()
fig5.savefig("offline_hard_cases.png", dpi=150)

plt.show()
