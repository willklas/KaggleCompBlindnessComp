import re
from glob import glob
import pandas as pd
import random
import numpy as np
import cv2
import tqdm

from keras.models import Sequential, model_from_json, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout, Input, Concatenate, GlobalAvgPool2D, GlobalMaxPool2D, Subtract, Multiply
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras.callbacks import ModelCheckpoint

# global variables
learning_rate       = 0.00001
epochs              = 100
batch_size          = 16
steps_per_epoch     = 200
validation_steps    = 100
input_shape         = (224, 224, 3)
num_classes         = 5


# create model
def create_model(model_name):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='sigmoid'))

    return model


# load image into numpy array
def read_image(image_path):
    image = cv2.imread(image_path)
    # TO DO: proper resizing!!!
    image = cv2.resize(image,(224,224))
    image = np.array(image).astype(np.float)
    # return preprocess_input(image, version=2)
    return image


# generator for producing batches of images to train on
def my_gen(training_set, image_names_to_paths, batch_size=16):
    # TO DO: gather more data by manipuating images (rotations) with same label
    while True:
        training_batch = random.sample(training_set, batch_size)
        labels = []
        batch_image_paths = []
        for training_sample in training_batch:
            batch_image_paths.append(image_names_to_paths[training_sample[0]])
            labels.append(training_sample[1])
        
        batch_images = np.array([read_image(image_path) for image_path in batch_image_paths])

        yield ([np.array(batch_images)], [np.array(labels)])

    return


# load a previosly saved model (for training or predicting)
def load_saved_model(model_name):
    model_structure_path = "models/" + model_name.split("_all_epochs")[0] + ".json"
    model_weights_path = "models/" + model_name + ".h5"
    model_structure_file = open(model_structure_path, "r")
    # load the model structure
    model = model_from_json(model_structure_file.read())
    # load the model weights
    model.load_weights(model_weights_path)
    model_structure_file.close()

    return model


# train the model (after creating) and save the weights
def train(model_name, continue_training):
    print("beginning training routine with model " + model_name + "...")
    training_labels = pd.read_csv("input/train/train.csv")
    all_image_paths = glob("input/train/*.png") 
    print(type(all_image_paths), len(all_image_paths))

    image_names_to_paths = {}
    for image_path in all_image_paths:
        # for dealing with annoying windows path formatting
        edited_image_path = image_path.replace('\\', '/')
        parsed_path = edited_image_path.split('/')
        key = parsed_path[-1].split('.')[0]
        image_names_to_paths[key] = edited_image_path
    
    training_set = []
    validation_set = []
    num_validation = len(training_labels) // 5
    validation_indicies = random.sample(range(len(training_labels)), num_validation)

    for i in range(len(training_labels.id_code.values)):
        image_name = training_labels.id_code.values[i]
        label      = [0]*5
        label[training_labels.diagnosis.values[i]] = 1
        if image_name in image_names_to_paths:
            if i in validation_indicies:
                validation_set.append((image_name, label))
            else:
                training_set.append((image_name, label))
    
    print("length of training set is", len(training_set), "and lengths of validation set is", len(validation_set))

    if (not continue_training):
        print('creating new model...')
        model = create_model(model_name)
        # save model to json file
        model_file = open("models/" + model_name + ".json", "w")
        model_file.write(model.to_json())
        model_file.close()
    else:
        print('loading saved model to continue training...')
        model = load_saved_model(model_name)
    
    # print a summary of the model    
    model.summary()
    
    model.compile(loss="categorical_crossentropy", metrics=['acc'], optimizer=Adam(learning_rate))
    
    # set up checkpoint so we can save the model (if better accuracy) after EVERY epoch. So if it crashes, or converges early, we're all good.
    model_weights_file = "models/" + model_name + ".h5"

    # accuracy is determined based on validation data, not train data, and model save file will only be updated if accuracy on val data is increased
    checkpoint = ModelCheckpoint(model_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    print("length of training set is", len(training_set), "and length if validation set is", len(validation_set))
    print("fitting model...")

    model.fit_generator(my_gen(training_set, image_names_to_paths, batch_size=batch_size), validation_data=my_gen(validation_set, image_names_to_paths, batch_size=batch_size), \
                        callbacks=callbacks_list, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    # this will be the final model from training on all epocs (probably overfitted), the above checkpointed saved model will be the model based on best validation set accuracy (early convergence)
    model.save_weights("models/" + model_name + "_all_epochs.h5")

    return


# predict test and save results
def predict(model_name):
    print("beginning testing routine with model " + model_name + "...")
    test_data_set = pd.read_csv("input/test/test.csv")

    print('loading model...')
    model = load_saved_model(model_name)

    predictions = []
    print("predicting... ")
    for i in tqdm.tqdm(range(len(test_data_set.id_code.values))):
        image = test_data_set.id_code.values[i]
        numpy_image = read_image("input/test/" + image + ".png")
        prediction = model.predict([[numpy_image]])
        prediction = prediction[0]
        # print(prediction)

        max = -1
        max_id = 0
        for j in range(len(prediction)):
            if prediction[j] > max:
                max = prediction[j]
                max_id = j
                # print('here')
        predictions.append(max_id)

    test_data_set["diagnosis"] = predictions
    test_data_set.to_csv("submissions/" + model_name + ".csv", index=False)

    return


""" 
    - use train and predict as you like (comment out train if you only want to predict and vice versa)
    - don't include path or extension for model name: "model_name".csv is submission file, "model_name".json is model structure, "model_name".h5 is model weights
    - if training a new model (new training method or new model structure), or wanting to not overwrite previous models, use a new model name in 'train', and reference it in 'predict'
"""
if __name__ == "__main__":
    # trains model and saves it with given name (if continuing training on a model, set 2nd argmnt to True)
    train(model_name = "model_1", continue_training=False)
    # takes the referenced model and creates submission file from testing images
    predict(model_name = "model_1")