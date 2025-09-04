import os
import librosa
import numpy as np
from scipy.signal import resample
import zipfile
import os

zip_path = "/content/synthetic_voice_dataset (1).zip"  # Path to the ZIP file
extract_path = "/content/dataset"  # Directory where the dataset will be extracted

# Create extraction folder
os.makedirs(extract_path, exist_ok=True)

# Extract ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"✅ Dataset extracted to: {extract_path}")
from google.colab import drive
drive.mount('/content/drive')
# Step 1: Import necessary libraries
import zipfile
import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import numpy as np

def extract_features(file_path, max_pad_len=100):
    audio, sr = librosa.load(file_path, sr=22050)  # Load audio file
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)  # Extract MFCC features

    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]

    return mfccs[..., np.newaxis]  # Ensures output shape is (40, 100, 1)
dataset_path = "/content/dataset"
speakers = sorted(os.listdir(dataset_path))  # Get speaker folder names
X, y = [], []

for label, speaker in enumerate(speakers):
    speaker_path = os.path.join(dataset_path, speaker)

    if os.path.isdir(speaker_path):
        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                features = extract_features(file_path)  # Extract MFCC features

                X.append(features)
                y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = to_categorical(y, num_classes=len(speakers))  # Convert labels to one-hot encoding
X = np.expand_dims(X, -1)  # Add a channel dimension for CNN input
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape)  # Should match (samples, 40, 100, 100, 1)
X_train = np.squeeze(X_train, axis=-1)  # Removes last redundant dimension
X_test = np.squeeze(X_test, axis=-1)

print(X_train.shape, X_test.shape)
# Expected Output: (800, 40, 100, 1) (200, 40, 100, 1)

print("X_train shape:", X_train.shape)  # Should be (num_samples, 40, 100, 1, 1)
print("y_train shape:", y_train.shape)  # Should be (num_samples,)
import numpy as np

y_train = np.argmax(y_train, axis=1)  # Convert (800, 100) → (800,)
y_test = np.argmax(y_test, axis=1)    # Convert (200, 100) → (200,)
# Reshape X to (num_samples, time_steps, height, width, channels)
# Assuming max_pad_len is 100 and each MFCC has 40 coefficients
X_train = X_train.reshape(X_train.shape[0], 40, 100, 1, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 100, 1, 1)
from tensorflow.keras.optimizers import Adam
print(Adam)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, LSTM, Dense, Dropout

model = Sequential([
    TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding="same"), input_shape=(40, 100, 100, 1)),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding="same")),
    TimeDistributed(MaxPooling2D((2, 2))),
    TimeDistributed(GlobalAveragePooling2D()),  # ✅ Reduce feature size
    LSTM(64, return_sequences=True),  # ✅ Reduce LSTM size
    LSTM(32),
    Dense(32, activation='relu'),  # ✅ Reduce Dense layer size
    Dropout(0.3),
    Dense(100, activation='softmax')  # Output for 100 classes
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
from tensorflow.keras.utils import to_categorical

# If already one-hot encoded, convert back to integers
if y_train.ndim == 3:  # Shape (800, 100, 100) is wrong
    y_train = np.argmax(y_train, axis=-1)  # Convert (800, 100, 100) → (800, 100)
    y_test = np.argmax(y_test, axis=-1)    # Convert (200, 100, 100) → (200, 100)

# Check shape before re-applying one-hot encoding
print("y_train shape before fixing:", y_train.shape)  # Should be (800, 100)
print("y_test shape before fixing:", y_test.shape)    # Should be (200, 100)
y_train = np.expand_dims(y_train, axis=1)  # Add a time step dimension
y_train = np.repeat(y_train, 40, axis=1)   # Repeat across 40 time steps

y_test = np.expand_dims(y_test, axis=1)
y_test = np.repeat(y_test, 40, axis=1)

print(y_train.shape)  # Expected: (800, 40, 100)
print(y_test.shape)   # Expected: (200, 40, 100)
import numpy as np

print("Unique class labels in y_train:", np.unique(np.argmax(y_train, axis=1)))
model.compile(
    loss='categorical_crossentropy',  # Correct loss for one-hot labels
    optimizer='adam',
    metrics=['accuracy']
)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model with a smaller batch size
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose = 1)
# Save the entire model (architecture + weights)
model.save("speaker_recognition_model.h5")
print("Model saved successfully! ✅")
from tensorflow.keras.models import load_model

# Load the model
model = load_model("speaker_recognition_model.h5")
print("Model loaded successfully! ✅")
# Evaluate on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f} ✅")
