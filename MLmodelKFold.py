
import keras.applications.xception
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2.cv2 as cv2
import os
import random
import shutil

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pathlib
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix, recall_score
import math

# create a new folder where the data can be found
# Undersample the good print and interrupted print data set to 243 samples in total, to create a balanced data set
# Assign 195 samples to the training folder of the corresponding class and 48 samples to the testing folder
DatasetFolder = "C:/Users/janly/Documents/Master Thesis/Undersampled_data"
save_dir = "C:/Users/janly/Documents/Master Thesis/Undersampled_data/Callbacks/exp_21_01_01/"
#model_save_dir = "C:/Users/janly/Documents/Master Thesis/Undersampled_data/Saved_model"
class_labels = ["double_print", "good_print", "interrupted_print"]

source_files = []
X = []
Y = []

# height and width of the images from the dataset
img_height = 360
img_width = 220

# create subfolders within the "KFold_Data/DS" folder
train_path = DatasetFolder+"/training/"
validation_path = DatasetFolder+"/validation/"
test_path = DatasetFolder+"/testing/"


def transferBetweenFolders(source, dest, split_rate):
    # transfer files between folders randomly with a predefined split rate
    global source_files
    source_files = os.listdir(source)
    if len(source_files) != 0:
        transfer_file_numbers = int(len(source_files)*split_rate)
        transfer_index = random.sample(range(0, len(source_files)), transfer_file_numbers)
        for index in transfer_index:
            shutil.move(source+str(source_files[index]), dest+str(source_files[index]))

    else:
        print("No file moved. Source empty")


def transferClassBetweenFolders(source, dest, split_rate):
    # make sure that files for all the classes are transfered to the correct folder
    for label in class_labels:
        transferBetweenFolders(DatasetFolder+"/"+source+"/"+label+"/",
                               DatasetFolder+"/"+dest+"/"+label+"/",
                               split_rate)


# Check if there are files in the testing folder, otherwise move them all to the training folder
#transferClassBetweenFolders("testing", "training", 1.0)

# Divide the files among the training and testing folders
#transferClassBetweenFolders("training", "testing", 0.2)


def prepareNameWithLabels(folder_name):
    source_files = os.listdir(DatasetFolder+"/training/"+folder_name)
    for val in source_files:
        X.append(val)
        for count, value in enumerate(class_labels):
            if folder_name == class_labels[count]:
                Y.append(count)


# organise file names and class labels to the X and Y
for i in range(len(class_labels)):
    prepareNameWithLabels(class_labels[i])

X = np.array(X)
Y = np.array(Y)

# parameters for the model are defined
num_classes = len(class_labels)
epochs = 7
batch_size = 32


def getmodel():
    # defining the model with different layers. There is a maxpooling layer in each layer and a fully connected layer
    model = Sequential([
        layers.Conv2D(16, 3, activation="relu", input_shape=(img_width, img_height, 3)),
        layers.MaxPooling2D(),
        layers.Dropout(0.1, seed=2022),

        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.15, seed=2022),

        #layers.Conv2D(64, 3, padding="same", activation="relu"),
        #layers.MaxPooling2D(),
        #layers.Dropout(0.2),

        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2, seed=2022),
        layers.Dense(num_classes, activation="sigmoid")
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    return model


def get_model_name(k):
    return 'model_'+str(k)+'.h5'


model = getmodel()

# K-FOLD CROSS VALIDATION to mitigate the risk of overfitting

class_weights = {0: 1.5, 1: 1, 2: 4.5}

skf = StratifiedKFold(n_splits=5, shuffle=True)
skf.get_n_splits(X, Y)
fold_number = 0


for train_index, val_index in skf.split(X, Y):
    # cut all images from validation to training, if they exist
    transferClassBetweenFolders("validation", "training", 1.0)
    fold_number += 1
    print("Result for fold", fold_number)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]

    # creating callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=save_dir+get_model_name(fold_number),
                                                    monitor="val_categorical_accuracy",
                                                    verbose=1,
                                                    save_best_only=True, mode="max")
    callback_list = [checkpoint]
    # Move validation images from training to validation map
    for Xindex, Xvalue in enumerate(X_val):
        class_label = ""
        for Yindex, Yvalue in enumerate(class_labels):
            if Y_val[Xindex] == Yindex:
                class_label = class_labels[Yindex]
        shutil.move(DatasetFolder+"/training/"+class_label+"/"+X_val[Xindex],
                    DatasetFolder+"/validation/"+class_label+"/"+X_val[Xindex])

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.1, fill_mode="nearest")
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Start image classification model
    train_generator = train_datagen.flow_from_directory(directory=train_path,
                                                    target_size=(img_width, img_height),
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    seed=42,
                                                    class_mode="categorical")

    validation_generator = validation_datagen.flow_from_directory(directory=validation_path,
                                                                 target_size=(img_width, img_height),
                                                                 batch_size=batch_size,
                                                                 seed=42,
                                                                 class_mode="categorical")

    history = model.fit(train_generator, validation_data=validation_generator, epochs=epochs,
                        callbacks=callback_list, class_weight=class_weights)

    val_results = model.evaluate(validation_generator, verbose=1)


    #print("validation results:", val_results)


# yPredictions = np.argmax(predictions, axis=1)
# true_classes = validation_generator.classes

# val_acc = history.history["val_categorical_accuracy"]
# val_prec = history.history["val_precision"]
# val_rec = history.history["val_recall"]



# Testing phase
print("==========TEST RESULTS==========")
n_folds = 5

for n in range(n_folds):
    model_id = n+1
    model = load_model(save_dir+get_model_name(model_id))
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_path, target_size=(img_width, img_height),
                                                      batch_size=batch_size, shuffle=False, class_mode=None)

    predictions = model.predict(test_generator, verbose=1)
    yPredictions = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes

    test_acc = accuracy_score(true_classes, yPredictions)
    test_prec = precision_score(true_classes, yPredictions, average="weighted")
    test_reca = recall_score(true_classes, yPredictions, average="weighted")
    test_Fscore = f1_score(true_classes, yPredictions, average="weighted")
    test_cm = confusion_matrix(true_classes, yPredictions)
    print("Accuracy: {0} \n Precision: {1} \n Recall: {2} \n F1 score: {3} \nConfusion Matrix: {4}".format(test_acc,
                                                                                                test_prec, test_reca,
                                                                                                test_Fscore, test_cm))
#for fold in range(1, 11):
#    model.load_weights(save_dir+"/model"+str(fold))
#    test_datagen = ImageDataGenerator(rescale=1./255)
#    test_generator = test_datagen.flow_from_directory(test_path, target_size=(img_height, img_width),
#                                                  batch_size=batch_size, class_mode=None)

#    predictions = model.predict(test_generator, verbose=1)
#    yPredictions = np.argmax(predictions, axis=1)
#    true_classes = test_generator.classes

#    test_acc = accuracy_score(true_classes, yPredictions)
#    test_prec = precision_score(true_classes, yPredictions, average="weighted")
#    test_reca = recall_score(true_classes, yPredictions, average="weighted")
#    test_Fscore = f1_score(true_classes, yPredictions, average="weighted")
#    test_cm = confusion_matrix(true_classes, yPredictions)
#    print("Accuracy: {0} \n Precision: {1} \n Recall: {2} \n F1 score: {3} \nConfusion Matrix: {4}".format(test_acc,
#                                                                                               test_prec, test_reca,
#                                                                                                test_Fscore, test_cm))
#save the model in a seperate folder within the working directory
#model.save("modelKFoldV2")






# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
