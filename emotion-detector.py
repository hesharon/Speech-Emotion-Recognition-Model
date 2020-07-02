import librosa
import soundfile as sf
import os
import glob  # finds all pathnames matching a specified pattern, results returned in arbitrary order
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


def extract_feature(file_name, mfcc, chroma, mel):
    with sf.SoundFile(file_name) as sound_file:
        x = sound_file.read(dtype="float32")
        # Number of samples of audio carried per sec (frequency of samples used in a digital recording)
        sample_rate = sound_file.samplerate
        if chroma:  # if chroma exists
            # find the short time fourier transform of np.ndarray input signal, returns an np.ndarray of stft coefficients
            stft = np.abs(librosa.stft(x))
        result = np.array([])

        if mfcc:  # if mfcc exists
            mfccs = np.mean(librosa.feature.mfcc(
                y=x, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(
                S=stft, sr=sample_rate,).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(
                y=x, sr=sample_rate,).T, axis=0)
            result = np.hstack((result, mel))

        return result


# Dictionary to hold numbers and emotions available
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Load data and extract features for each sound file


def load_data(test_size=0.2):
    x, y = [], []
    file_list = glob.glob(
        "/Users/sharonhe/Downloads/speech-emotion-recognition-ravdess-data/Actor_*/*.wav")
    for file in file_list:
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train, x_test, y_train, y_test = load_data(test_size=0.25)

# print((x_train.shape[0], x_test.shape[0]))

# print(f'Features extracted: {x_train.shape[1]}')

# Initializing the MLPClassifier
lr = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                   hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)

lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print(f'Accuracy: {accuracy*100}%')

# Saving model using joblib
joblib.dump(lr, "emotion-detector-model.pkl")

lr = joblib.load("emotion-detector-model.pkl")
