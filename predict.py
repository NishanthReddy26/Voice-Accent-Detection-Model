import easygui
import math
import librosa
import numpy as np
import tensorflow.keras as keras
import json
import os
SAMPLE_RATE = 22050
MODEL_PATH = "model.h5"
DATA_PATH = "data.json"


def predict(file_path, num_mfcc=13, n_fft=2048, hop_length=512, segment_duration=3):
    model = keras.models.load_model(MODEL_PATH)
    probability = [0] * 999
    samples_per_segment = segment_duration * SAMPLE_RATE
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

    track_duration = librosa.get_duration(y=signal, sr=sample_rate)
    num_segments = math.floor(track_duration / segment_duration)

    for d in range(num_segments):

        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        # use only mfcc feature with expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            x = mfcc[..., np.newaxis]
            x = x[np.newaxis, ...]
            prediction = model.predict(x)
            index = 0
            for p in prediction[0]:
                probability[index] += p
                index += 1

    index = 0
    curr_max = -1
    max_index = -1
    for p in probability:
        if curr_max < p:
            curr_max = p
            max_index = index
        index += 1

    with open(DATA_PATH, "r") as fp:
        data = json.load(fp)

    classes = np.array(data["mapping"])

    return classes[max_index]


if __name__ == "__main__":
    # for i in range(0,800):
    path = './test/'
    print(path)

    #print(os.path.isfile("'c:\\Users\\E NISHANTH REDDY\\PycharmProjects\\pythonProject3\\test\\test_1.mp3"))

    # for files in os.listdir('c:\\Users\\E NISHANTH REDDY\\PycharmProjects\\pythonProject3\\test'):
    #     if files.endswith('.mp3'):
    #         os.rename(files, files.replace('.mp3', '.wav'))
                # predict(file, segment_duration=3)
    m = {}
    # l1=[]
    # l2=[]
    for files in os.listdir('./test'):
        place = path + files
        name = files[:len(files)-4]

        now = predict(place, segment_duration=3)
        # l1.append(name)
        # l2.append(now)
        # print(name,now)
        m[name] = now
    # for i in range(len(l1)):
    #     # print(l1[i],l2[i])
    print(m)

    bro = json.dumps(m, indent = 4)
    with open ("Merlions.json","w") as hi:
        json.dump(m,hi)
    # filename = easygui.fileopenbox()
    # if filename.endswith('.wav'):
    #     predict(filename, segment_duration=3)
