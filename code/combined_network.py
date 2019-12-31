# import the libraries

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from imutils import paths
import cv2
import os
import numpy as np
import pickle


# DrowsyNetwork class for neural network model

class DrowsyNetwork:
    # initialize the model
    
    def build_model_binary_classifier():
        model = Sequential()

        # 1st layer of convolution and pooling
        model.add(Convolution2D(filters = 32, kernel_size = (3, 3),input_shape = (64, 64, 3), activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # 2nd layer of convolution and pooling
        model.add(Convolution2D(32, 3, 3, activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # flattening layer
        model.add(Flatten())
    
        # fully connected layer
        # hidden layer
        model.add(Dense(units = 128, activation = "relu"))
        # output layer
        model.add(Dense(units = 1, activation = "sigmoid"))

        # compile the cnn
        model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])
        
        return model
    
    def build_model_multi_label_classifier(numCategories):
        model = Sequential()

        # 1st layer of convolution and pooling
        model.add(Convolution2D(filters = 32, kernel_size = (3, 3),input_shape = (64, 64, 3), activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # 2nd layer of convolution and pooling
        model.add(Convolution2D(32, 3, 3, activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        # 3rd layer of convolution and pooling
        model.add(Convolution2D(32, 3, 3, activation = "relu"))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        # flattening layer
        model.add(Flatten())
    
        # fully connected layer
        # hidden layer
        model.add(Dense(units = 128, activation = "relu"))
        # output layer
        model.add(Dense(units = numCategories, activation = "softmax"))

        # compile the cnn
        model.compile(optimizer = "adam",loss = "categorical_crossentropy",metrics = ["accuracy"])
        
        return model



# code to train the neural networks

parameters = ['mouth', 'face', 'eye', 'head']
epochs = [500,500,500,500]

imagePaths = [list(paths.list_images("dataset_new/categorized_data/mouth/")),
              list(paths.list_images("dataset_new/categorized_data/face/")),
              list(paths.list_images("dataset_new/categorized_data/eye/")),
              list(paths.list_images("dataset_new/categorized_data/head/"))]
data = []
labels = []
precision = []
counter = -1

for imagePaths_per_feature in imagePaths:
    counter = counter + 1
    data = []
    labels = []
    for imagePath in imagePaths_per_feature:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (64, 64))
        image = img_to_array(image)
        data.append(image)
        
        open_or_close_label = imagePath.split(os.path.sep)[-2]
        labels.append(open_or_close_label)

    LB = LabelBinarizer()
    labels = LB.fit_transform(labels)
    print("3")
    data = np.array(data, dtype="float") / 255.0
    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    # build the model
    model = DrowsyNetwork.build_model_binary_classifier()
    #print(model)
    
    aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2,
                         height_shift_range=0.2, shear_range=0.15,horizontal_flip=True, fill_mode="nearest")
    
    # fit the data
    model.fit_generator(aug.flow(trainX, trainY, batch_size=20), steps_per_epoch=10, nb_epoch = epochs[counter],
                    validation_data = (testX, testY))

    print("[INFO] serializing network...")
    model.save("output_new/drowsy_"+parameters[counter]+".model")

    # save the binarizer to disk
    print("[INFO] serializing "+parameters[counter]+ " label binarizer...")
    f = open("output_new/"+parameters[counter]+"_lb.pickle", "wb")
    f.write(pickle.dumps(LB))
    f.close()
    
    y_pred = model.predict(testX)
    y_pred = y_pred > 0.5
    y_test = testY
    precision.append(precision_score(y_test, y_pred))
    print("[INFO] precision score for "+parameters[counter]+ " network is... ")
    print(precision)
    

