import glob
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

imposter_path = "faces_imposter/*.png"
client_path = "faces_client/*.png"
SIZE = (100, 100)
THRESHOLD = 0.3
NAMES = {0: "imposter", 1: "client"}


def extract_hog(img):
    img = cv2.resize(img, SIZE)
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")


def load_data(pattern):
    feats, imgs, names = [], [], []
    for f in sorted(glob.glob(pattern)):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {f}")
            continue
        imgs.append(cv2.resize(img, SIZE))
        feats.append(extract_hog(img))
        names.append(f)
    return np.array(feats), np.array(imgs), np.array(names)


client_X, client_img, client_files = load_data(client_path)
imposter_X, imposter_img, imposter_files = load_data(imposter_path)
print(f"Loaded {len(client_X)} client, {len(imposter_X)} imposter. "
      f"Feature length: {client_X.shape[1]}")

client_y = np.ones(len(client_X), dtype=np.int32)
imposter_y = np.zeros(len(imposter_X), dtype=np.int32)

(X1_train, X1_val, img1_train, img1_val, f1_train, f1_val,y1_train, y1_val) = train_test_split(imposter_X, imposter_img, imposter_files, imposter_y,train_size=25, test_size=5, random_state=42)
(X2_train, X2_val, img2_train, img2_val, f2_train, f2_val,y2_train, y2_val) = train_test_split(client_X, client_img, client_files, client_y,train_size=25, test_size=5, random_state=42)

X_train = np.concatenate((X1_train, X2_train))
X_val = np.concatenate((X1_val, X2_val))
img_train = np.concatenate((img1_train, img2_train))
img_val = np.concatenate((img1_val, img2_val))
files_train = np.concatenate((f1_train, f2_train))
files_val = np.concatenate((f1_val, f2_val))
y_train = np.concatenate((y1_train, y2_train))
y_val = np.concatenate((y1_val, y2_val))

# Search over C------------------------------------------------
c_vals = np.logspace(-5, 3, 17)
train_accuracy, val_accuracy = [], []

for c in c_vals:
    m = SVC(kernel="linear", C=c, random_state=42)   
    m.fit(X_train, y_train)
    train_accuracy.append(m.score(X_train, y_train))
    val_accuracy.append(m.score(X_val, y_val))
    print(f"C={c:10.4g}  train={train_accuracy[-1]:.3f}  val={val_accuracy[-1]:.3f}")

best_acc = max(val_accuracy)
cands = [i for i, a in enumerate(val_accuracy) if a == best_acc]
best_c = c_vals[cands[len(cands) // 2]]   
print(f"\nBest validation accuracy {best_acc:.3f} for {len(cands)} value(s) of C; "
      f"choosing C = {best_c:g} (lambda = {1 / best_c:g})")

fig_acc = plt.figure(figsize=(6, 4))
plt.semilogx(c_vals, train_accuracy, "o-", label="train")
plt.semilogx(c_vals, val_accuracy, "s-", label="validation")
plt.axvline(best_c, color="gray", linestyle="--", label=f"chosen C = {best_c:g}")
plt.xlabel("C (= 1/lambda)")
plt.ylabel("accuracy")
plt.title("Linear SVM: accuracy vs C")
plt.legend()
plt.grid(True)
plt.tight_layout()
fig_acc.savefig("svm_accuracy_vs_C.png", dpi=150)

#final model------------------------------------------------------------------ :)
svm = SVC(kernel="linear", C=best_c, random_state=42)
svm.fit(X_train, y_train)
joblib.dump(svm, "svm.joblib")


def predict_with_threshold(model, X):
    return (model.decision_function(X) > THRESHOLD).astype(int)


def report(name, X, y):
    pred = predict_with_threshold(svm, X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\n=== {name} ({len(y)} samples, threshold = {THRESHOLD:g}) ===")
    print(f"Accuracy : {accuracy_score(y, pred):.3f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y, pred, zero_division=0):.3f}")
    print(f"F1       : {f1_score(y, pred, zero_division=0):.3f}")
    print(f"FAR (imposters accepted): {fp / max(fp + tn, 1):.3f}")
    print(f"FRR (client rejected)   : {fn / max(fn + tp, 1):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y, pred, labels=[0, 1]))
    print(classification_report(y, pred, target_names=["imposter", "client"],
                                zero_division=0))


report("TRAIN", X_train, y_train)
report("VALIDATION", X_val, y_val)

np.set_printoptions(precision=3, suppress=True)
print("\nValidation decision scores:")
for fn_, t, s in zip(files_val, y_val, svm.decision_function(X_val)):
    print(f"  {fn_:35s} true={NAMES[t]:9s} score={s:+.3f}")


def show_results(imgs, X, y, files, title, cols=5):

    pred = predict_with_threshold(svm, X)
    score = svm.decision_function(X)
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.9 * rows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t, p, s, f in zip(axes.ravel(), imgs, y, pred, score, files):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{f.split('/')[-1].split(chr(92))[-1]}\n"f"true: {NAMES[t]}\npred: {NAMES[p]} ({s:+.2f})",fontsize=8, color="green" if t == p else "red")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


fig_val = show_results(img_val, X_val, y_val, files_val, "Validation images")
fig_val.savefig("validation_results.png", dpi=150)

wrong = np.where(predict_with_threshold(svm, X_train) != y_train)[0]
if len(wrong):
    fig_wrong = show_results(img_train[wrong], X_train[wrong], y_train[wrong],files_train[wrong], "Misclassified training images",cols=max(len(wrong), 2))
    fig_wrong.savefig("train_misclassified.png", dpi=150)
else:
    print("\nNo misclassified training images.")

plt.show()  
