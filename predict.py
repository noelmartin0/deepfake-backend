import os
import cv2
import face_recognition
import matplotlib.pyplot as plt
import numpy as np
import librosa
import moviepy.editor as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# Function to preprocess video for deepfake detection


def preprocess_video(video_path, output_folder):
    video_capture = cv2.VideoCapture(video_path)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_counter = 0

    while frame_counter < 5:
        ret, frame = video_capture.read()

        if not ret:
            break

        face_locations = face_recognition.face_locations(frame)

        for face_location in face_locations:
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            face_image = cv2.resize(face_image, (224, 224))

            face_filename = os.path.join(
                output_folder, f"{os.path.basename(video_path)}_frame_{frame_counter}_face.jpg")
            cv2.imwrite(face_filename, face_image)

        frame_counter += 1

    video_capture.release()
    print("Face extraction completed.")


# Function to convert video to audio
def video_to_audio(video_path, audio_path):
    clip = mp.VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
    print("Audio extracted successfully.")


# Function to extract MFCC features from audio file
def extract_mfcc(audio_path, duration=2.0, num_mfcc=25, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(audio_path, sr=None)

    target_length = int(duration * sr)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), 'constant')
    else:
        audio = audio[:target_length]

    mfccs = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)

    if mfccs.shape[1] < 173:
        mfccs = np.pad(
            mfccs, ((0, 0), (0, 173 - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :173]

    return mfccs


# Function to preprocess audio file
def preprocess_audio_file(input_audio_path, output_dir, duration=2.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mfccs = extract_mfcc(input_audio_path, duration=duration)

    output_file_path = os.path.join(output_dir, os.path.basename(
        input_audio_path).replace('.wav', '.npy'))
    np.save(output_file_path, mfccs)


# Function to load preprocessed audio
def load_preprocessed_audio(audio_file_path):
    preprocess_audio_file(audio_file_path, "Preprocessed_audio", duration=2.0)
    audio_data = np.load("Preprocessed_audio/output_audio.npy")
    return audio_data


# Function to predict audio deepfake
def predict_audio_deepfake(audio_data):
    loaded_model = load_model("Models/audio_model.h5")
    audio_data = audio_data.reshape(-1, 25, 173, 1)
    prediction = loaded_model.predict(audio_data)
    predicted_label = np.round(prediction).astype(int)[0]
    class_labels = ["real", "fake"]
    predicted_label_text = class_labels[predicted_label[0]]
    return predicted_label_text


# Function to predict video deepfake
def predict_video_deepfake():
    model = tf.keras.models.load_model('Models/video_model.h5')

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'Finalv3/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

    test_generator = test_datagen.flow_from_directory(
        'Preprocess',
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary')

    class_labels = list(train_generator.class_indices.keys())

    test_generator.reset()
    predictions = model.predict(test_generator)

    plt.figure(figsize=(10, 10))
    num_images = 1
    for i in range(num_images):
        image, label = test_generator.next()
        prediction = predictions[i][0]
        predicted_label = class_labels[int(round(prediction))]
    files = os.listdir("Preprocess/real")

    for file in files:
        file_path = os.path.join("Preprocess/real", file)
        os.remove(file_path)
        
    result = predicted_label
    return predicted_label



def multimodal_deepfake_detection(video_path, audio_path):
    preprocess_video(video_path, "Preprocess/real")

    # Check if the video has an audio track
    clip = mp.VideoFileClip(video_path)
    has_audio = clip.audio is not None
    clip.close()

    if has_audio:
        video_to_audio(video_path, audio_path)
        video_label = predict_video_deepfake()
        audio_data = load_preprocessed_audio(audio_path)
        audio_label = predict_audio_deepfake(audio_data)
        return video_label, audio_label
    else:
        video_label = predict_video_deepfake()
        return video_label, None



def prediction(video_path):
    audio_path = "output_audio.wav"
    video_label, audio_label = multimodal_deepfake_detection(video_path, audio_path)
    if audio_label != None:
        if video_label == "real" and audio_label == "real":
            return "Real" # Pass it to frontend via props
        else:
            return "Fake"
    else:
        if video_label == "real":
            return "Real"
        else:
            return "Fake"
        
