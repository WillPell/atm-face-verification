import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
THRESHOLD = 0.65


def show_grid(images, title, cols=10):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im in zip(axes.ravel(), images):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
    fig.suptitle(title)
    fig.tight_layout()


imposter_path = "faces_imposter/*png"
client_path = "faces_client/*png"
client_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob(client_path)]
imposter_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in glob.glob(imposter_path)]
print(f"Loaded {len(client_images)} client images.")
print(f"Loaded {len(imposter_images)} imposter images.")


labels_imposter = np.zeros(len(imposter_images), dtype=np.int32)
labels_client = np.ones(len(client_images), dtype=np.int32)



X1_train, X1_test, y1_train, y1_test = train_test_split(imposter_images, labels_imposter, test_size=5, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(client_images, labels_client, test_size=5, random_state=42)
X_train = np.concatenate((X1_train, X2_train), axis=0)
X_test = np.concatenate((X1_test, X2_test), axis=0)

show_grid(X2_train, f"Client: training ({len(X2_train)} images)")
show_grid(X2_test, f"Client: validation ({len(X2_test)} images)", cols=5)
show_grid(X1_train, f"Imposter: training ({len(X1_train)} images)")
show_grid(X1_test, f"Imposter: validation ({len(X1_test)} images)", cols=5)
plt.show()


y_train = np.concatenate((y1_train, y2_train), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_flat, y_train)


pred = lda.predict(X_test_flat)
prob = lda.predict_proba(X_test_flat)
accuracy = lda.score(X_test_flat, y_test)
print(f"LDA Accuracy: {accuracy}")


#Find best C values
c_vals= np.logspace(-8,2, 11)
train_accuracy = []
val_accuracy = []

for c in c_vals:
    model = LogisticRegression(C=c, solver='lbfgs', max_iter=1000)
    model.fit(X_train_flat, y_train)
    train_accuracy.append(model.score(X_train_flat, y_train))
    val_accuracy.append(model.score(X_test_flat, y_test))

i = int(np.argmax(val_accuracy))
c = c_vals[i]
print(f"Best C value is {c} (lambda = {1/c:g}), val acc = {val_accuracy[i]:.3f}")
lr = LogisticRegression(C=c, solver="lbfgs", max_iter=1000)
lr.fit(X_train_flat, y_train)

pred = lr.predict(X_test_flat)
prob = lr.predict_proba(X_test_flat)
accuracy = lr.score(X_test_flat, y_test)

joblib.dump({"lda": lda, "lr": lr}, "faceModel.model")

print(f"LR Accuracy: {accuracy}")
from sklearn.metrics import confusion_matrix

for name, model in [("LDA", lda), ("LR", lr)]:
    print(name, "train acc:", model.score(X_train_flat, y_train))
    print(name, "test acc: ", model.score(X_test_flat, y_test))
    print(confusion_matrix(y_test, model.predict(X_test_flat)))
    print(np.round(model.predict_proba(X_test_flat)[:, 1], 3))


def far_frr(y, p_client, t):
    pred = (p_client > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return fp / max(fp + tn, 1), fn / max(fn + tp, 1)


for name, model in [("LDA", lda), ("LR", lr)]:
    for split, X, y in [("train", X_train_flat, y_train),
                        ("validation", X_test_flat, y_test)]:
        far, frr = far_frr(y, model.predict_proba(X)[:, 1], THRESHOLD)
        print(f"{name} {split:10s} threshold={THRESHOLD}: FAR={far:.2f}  FRR={frr:.2f}")



