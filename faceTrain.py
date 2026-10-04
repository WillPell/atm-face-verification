import cv2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

imposter_path = "faces_imposter/*png"
client_path = "faces_client/*png"
client_images = [cv2.imread(file) for file in glob.glob(client_path)]
imposter_images = [cv2.imread(file) for file in glob.glob(imposter_path)]
print(f"Loaded {len(client_images)} client images.")
print(f"Loaded {len(imposter_images)} imposter images.")


labels_imposter = np.zeros(imposter_images.shape[0], dtype=np.int32)
labels_client = np.ones(client_images.shape[0], dtype=np.int32)

X1_train, X1_test, y1_train, y1_test = train_test_split(imposter_images, labels_imposter, test_size=0.2, random_state=42)
X2_train, X2_test, y2_train, y2_test = train_test_split(client_images, labels_client, test_size=0.2, random_state=42)

X_train = np.concatenate((X1_train, X2_train), axis=0)
X_test = np.concatenate((X1_test, X2_test), axis=0)

y_train = np.concatenate((y1_train, y2_train), axis=0)
y_test = np.concatenate((y1_test, y2_test), axis=0)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_flat, y_train)
joblib.dump(lda, "lda.joblib")

pred = lda.predict(X_test_flat)
prob = lda.predict_proba(X_test_flat)
accuracy = lda.score(X_test_flat, y_test)
print(f"Accuracy: {accuracy}")

lr = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
lr.fit(X_train_flat, y_train)
joblib.dump(lr, "lr.joblib")
pred = lr.predict(X_test_flat)
prob = lr.predict_proba(X_test_flat)
accuracy = lr.score(X_test_flat, y_test)
print(f"Accuracy: {accuracy}")
