# Speech-Based-Biometric
This project identifies speakers from audio using MFCC features and a hybrid CNN + LSTM model. It includes preprocessing, training, and evaluation, with applications in voice authentication, smart assistants, and call center automation.
# 🎤 Speaker Recognition using CNN + LSTM

This project implements a **Deep Learning-based Speaker Recognition
System** that identifies speakers from audio samples using **MFCC
feature extraction** and a hybrid **CNN + LSTM model**.

------------------------------------------------------------------------

## 📂 Project Structure

-   dataset/ → Extracted dataset (folders per speaker)
-   speaker_recognition_model.h5 → Saved trained model
-   main.py → Training + evaluation script
-   README.md → Project documentation

------------------------------------------------------------------------

## 🚀 Features

-   Preprocessing audio with **MFCC feature extraction**
-   Handling variable audio lengths with padding/truncation
-   CNN layers for **spatial feature extraction**
-   LSTM layers for **temporal modeling**
-   Model training, validation, and evaluation
-   Model saving and loading for reuse

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository and install dependencies:

    git clone https://github.com/yourusername/speaker-recognition.git
    cd speaker-recognition
    pip install -r requirements.txt

### Requirements

-   Python 3.8+
-   TensorFlow 2.x
-   Librosa
-   NumPy
-   Scikit-learn
-   SciPy

------------------------------------------------------------------------

## 📊 Training the Model

Run the training script:

    python main.py

This will: 1. Extract features from the dataset 2. Train the CNN+LSTM
model 3. Save the trained model as `speaker_recognition_model.h5`

------------------------------------------------------------------------

## 🧪 Evaluation

    from tensorflow.keras.models import load_model

    model = load_model("speaker_recognition_model.h5")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

------------------------------------------------------------------------

## 🎤 Demo Usage (Predict Speaker)

    import librosa
    import numpy as np
    from tensorflow.keras.models import load_model

    model = load_model("speaker_recognition_model.h5")

    def extract_features(file_path, max_pad_len=100):
        audio, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs[..., np.newaxis]

    def predict_speaker(audio_path, speakers):
        features = extract_features(audio_path)
        features = features.reshape(1, 40, 100, 1, 1)
        prediction = model.predict(features)
        speaker_id = np.argmax(prediction)
        return speakers[speaker_id]

    speakers = ["speaker_1", "speaker_2", "speaker_3"]
    print("Predicted Speaker:", predict_speaker("test_audio.wav", speakers))

------------------------------------------------------------------------

## 🎯 Applications

-   Voice authentication systems
-   Speaker verification in smart assistants
-   Security systems requiring speaker ID

------------------------------------------------------------------------

## 📌 Future Improvements

-   Add **data augmentation**
-   Implement **attention mechanism**
-   Extend to **real-time speaker recognition**

------------------------------------------------------------------------

## 📝 License

This project is licensed under the MIT License.
