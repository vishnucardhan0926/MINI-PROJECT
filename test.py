import os
import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

# Non-Binary Image Classification using Convolution Neural Networks

names = ['Palm','I','Fist','Fist Moved','Thumb','Index','OK','Palm Moved','C','Down']

'''
path = 'Dataset'

labels = []
X_train = []
Y_train = []


for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        print(name+" "+root+"/"+directory[j])
        if 'Thumbs.db' not in directory[j]:
            img = cv2.imread(root+"/"+directory[j], cv2.IMREAD_COLOR)
            img = cv2.flip(img, 1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (41, 41), 0)  #tuple indicates blur value
            ret, thresh = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thresh = cv2.resize(thresh, (64, 64))
            thresh = np.array(thresh)
            X_train.append(thresh)
            Y_train.append(int(name))
        
X_train = np.asarray(X_train)

Y_train = np.asarray(Y_train)
print(Y_train)
print(X_train.shape)

X_train = X_train.astype('float32')
X_train = X_train/255
    
test = X_train[3]
cv2.imshow("aa",test)
cv2.waitKey(0)
indices = np.arange(X_train.shape[0])
np.random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]
Y_train = to_categorical(Y_train)
np.save('model/X.txt',X_train)
np.save('model/Y.txt',Y_train)
'''

X_train = np.load('model/X.txt.npy')
Y_train = np.load('model/Y.txt.npy')
X_train = np.stack((X_train,)*3, axis=-1)
print(Y_train)
print(X_train.shape)
if os.path.exists('model/model.json'):
    with open('model/model.json', "r") as json_file:
        loaded_model_json = json_file.read()
        classifier = model_from_json(loaded_model_json)
    classifier.load_weights("model/model_weights.h5")
    classifier._make_predict_function()   
    print(classifier.summary())
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[19] * 100
    print("Training Model Accuracy = "+str(accuracy))
else:
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = 10, activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X_train, Y_train, batch_size=64, epochs=20, shuffle=True, verbose=2)
    classifier.save_weights('model/model_weights.h5')            
    model_json = classifier.to_json()
    with open("model/model.json", "w") as json_file:
        json_file.write(model_json)
    f = open('model/history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()
    acc = data['accuracy']
    accuracy = acc[19] * 100
    print("Training Model Accuracy = "+str(accuracy))
