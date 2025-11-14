import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "Dataset"
SAMPLE_RATE = 22050
DURATION = 2  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# -----------------------------
# FEATURE EXTRACTION
# -----------------------------
def extract_features(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad/truncate to fixed width
    if mel_spec_db.shape[1] < max_len:
        pad_width = max_len - mel_spec_db.shape[1]
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_spec_db = mel_spec_db[:, :max_len]

    return mel_spec_db

# -----------------------------
# LOAD DATA
# -----------------------------
X, y = [], []

for label, folder in enumerate(["Gunshot", "Normal_sound"]):
    path = os.path.join(DATASET_PATH, folder)
    print(f"Loading {folder}...")
    for file in os.listdir(path):
        if file.endswith(".mp3") or file.endswith(".wav"):
            file_path = os.path.join(path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# -----------------------------
# PREPARE DATA
# -----------------------------
X = X[..., np.newaxis]  # Add channel dimension
y = to_categorical(y, 2)  # One-hot encode labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# CNN MODEL
# -----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n✅ Starting training...\n")

# -----------------------------
# TRAIN MODEL
# -----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=16,
    verbose=1
)

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("gunshot_classifier.h5")
print("\n✅ Model trained and saved as gunshot_classifier.h5")

# -----------------------------
# EVALUATION
# -----------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")
print(f"📉 Test Loss: {test_loss:.4f}")

# -----------------------------
# TRAINING CURVES
# -----------------------------
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# -----------------------------
# CONFUSION MATRIX & REPORT
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Gunshot", "Normal_sound"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=["Gunshot", "Normal_sound"]))
