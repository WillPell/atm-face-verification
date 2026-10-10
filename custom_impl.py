import glob
import numpy as np
import joblib
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.svm import SVC
imposter_path = "faces_imposter/*.png"
client_path = "faces_client/*.png"
SIZE = (100, 100)

def extract_hog(img):
    img = cv2.resize(img, SIZE)
    return hog(img, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm="L2-Hys")

def load_features(pattern):
    feats = []
    for f in glob.glob(pattern):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not read {f}")
            continue
        feats.append(extract_hog(img))
    return np.array(feats)

client_X = load_features(client_path)
imposter_X = load_features(imposter_path)
print(f"Loaded {len(client_X)} client, {len(imposter_X)} imposter. Feature length: {client_X.shape[1]}")

client_y = np.ones(len(client_X), dtype=np.int32)
imposter_y = np.zeros(len(imposter_X), dtype=np.int32)

X1_train, X1_test, y1_train, y1_test = train_test_split(
    imposter_X, imposter_y, train_size=25, test_size=5, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(
    client_X, client_y, train_size=25, test_size=5, random_state=42)

X_train = np.concatenate((X1_train, X2_train))
X_test = np.concatenate((X1_test, X2_test))
y_train = np.concatenate((y1_train, y2_train))
y_test = np.concatenate((y1_test, y2_test))


svm = SVC(kernel="linear", C=1.0, probability=True, random_state=42)
svm.fit(X_train, y_train)
joblib.dump(svm, "svm.joblib")

def report(name, X, y):
    pred = svm.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    print(f"\n=== {name} ({len(y)} samples) ===")
    print(f"Accuracy : {accuracy_score(y, pred):.3f}")
    print(f"Precision: {precision_score(y, pred, zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y, pred, zero_division=0):.3f}")
    print(f"F1       : {f1_score(y, pred, zero_division=0):.3f}")
    print(f"FAR (imposters accepted): {fp / max(fp + tn, 1):.3f}")
    print(f"FRR (client rejected)   : {fn / max(fn + tp, 1):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y, pred, labels=[0, 1]))
    print(classification_report(y, pred, target_names=["imposter", "client"], zero_division=0))

report("TRAIN", X_train, y_train)
report("TEST", X_test, y_test)
print("Test decision scores:", np.round(svm.decision_function(X_test), 2))
