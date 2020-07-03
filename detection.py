import tensorflow as tf
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

# emotion is class 1, emotion intensity is class 2
emotions = {0:"neutral", 1:"calm", 2:"happy", 3:"sad", 4:"angry", 5:"fearful", 6:"disgust", 7:"surprised"}
emoIntensity = {0:"", 1:"very"}

EPOCHS = 10

# assigns first half of actors to training, second half to testing
def prepData():
	actorFolders = os.listdir("data")
	numActors = int(len(actorFolders)/6)
	X = []
	y1 = []
	y2 = []
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
			y1.append(labels[EMOTION]-1)
			y2.append(labels[EMOTIONAL_INTENSITY]-1)
			if len(samples) > maxLen:
				maxLen = len(samples)
			X.append(samples)
			numFiles = numFiles+1

	X = np.reshape(np.array(X), (numFiles, maxLen, 1))
	
	X_train = X[0:int(len(X)/2)]
	X_test = X[int(len(X)/2):int(len(X))]
	y_train_1 = np.array(y1[0:int(len(y1)/2)])
	y_test_1 = np.array(y1[int(len(y1)/2):int(len(y1))])
	y_train_2 = np.array(y2[0:int(len(y2)/2)])
	y_test_2 = np.array(y2[int(len(y2)/2):int(len(y2))])

	return (X_train, y_train_1, y_train_2), (X_test, y_test_1, y_test_2)

def trainModel(X, y1, y2):
	inputs = keras.Input(shape=(X.shape[1], X.shape[2]))
	lstm = layers.LSTM(64)(inputs)
	dense = layers.Dense(32)(lstm)
	emoOutput = layers.Dense(len(emotions), activation=tf.nn.softmax, name='emotion')(dense)
	emoIntOutput = layers.Dense(len(emoIntensity), activation=tf.nn.softmax, name='emotionIntensity')(dense)
	model = keras.Model(inputs=inputs, outputs=[emoOutput, emoIntOutput])

	model.compile(loss={'emotion':'sparse_categorical_crossentropy', 'emotionIntensity':'sparse_categorical_crossentropy'}, metrics={'emotion':'accuracy', 'emotionIntensity':'accuracy'}, optimizer='adam')
	model.summary()
	model.fit(X, [y1, y2], epochs=EPOCHS, verbose=2)
	model.save('latest-model.h5')

def test(samples, class1, class2):
	model = models.load_model('latest-model.h5')
	X = samples.reshape(1, len(samples), 1)
	pred = model.predict(X)
	print(pred)
	print("{}, {}".format(class1, class2))

(X_train, y_train_1, y_train_2), (X_test, y_test_1, y_test_2) = prepData()
trainModel(X_train, y_train_1, y_train_2)



