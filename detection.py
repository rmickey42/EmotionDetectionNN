import tensorflow
from tensorflow import keras
from tensorflow.keras import layers, models, utils
import wave
import os.path
import librosa
import numpy as np

# file label indexes. meaning at https://www.kaggle.com/uwrfkaggler/ravdess-emotional-speech-audio/data#
MODALITY = 0
VOCAL_CHANNEL = 1
EMOTION = 2
EMOTIONAL_INTENSITY = 3
STATEMENT = 4
REPETITION = 5
ACTOR = 6

emotions = {1:"neutral", 2:"calm", 3:"happy", 4:"sad", 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}
emoIntensity = {1:"", 2:"very"}

EPOCHS = 10

# assigns first half of actors to training, second half to testing
def prepData():
	actorFolders = os.listdir("data")
	numActors = len(actorFolders)
	X = []
	y = []
	maxLen = 116247
	numFiles = 0
	for i in range(numActors):
		folder = "data/"+actorFolders[i]
		files = os.listdir(folder)
		print(folder)
		for filepath in files:
			samples, samplerate = librosa.load(folder +"/"+filepath)
			samples = list(samples) + [0 for i in range(maxLen-len(samples))]
			labels = [int(lb) for lb in filepath.replace(".wav", "").split('-')]
			# we only care about the emotion and the emotional intensity in this case
			y.append(labels[EMOTION:EMOTIONAL_INTENSITY+1])
			if len(samples) > maxLen:
				maxLen = len(samples)
			X.append(samples)
			numFiles = numFiles+1

	X = np.reshape(np.array(X), (numFiles, maxLen, 1))
	
	X_train = X[0:int(len(X)/2)]
	X_test = X[int(len(X)/2):int(len(X))]
	y_train = np.array(y[0:int(len(y)/2)])
	y_test = (y[int(len(y)/2):int(len(y))])
	return (X_train, y_train), (X_test, y_test)

def trainModel(X, y):
	model = keras.Sequential()
	model.add(layers.LSTM(16, input_shape=(X.shape[1], X.shape[2])))
	model.add(layers.Dense(y.shape[1], activation="softmax"))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
	model.fit(X, y, epochs=EPOCHS)
	model.save('latest-model.h5')

def test(samples, classification):
	model = models.load_model('latest-model.h5')
	X = samples.reshape(1, len(samples), 1)
	pred = model.predict(X)
	print(pred)
	print(classification)

(X_train, y_train), (X_test, y_test) = prepData()
trainModel(X_train, y_train)



