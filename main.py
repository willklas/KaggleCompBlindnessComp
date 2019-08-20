
# import re
from glob import glob
import pandas as pd
import random
import numpy as np
import cv2
# import tqdm

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout

# from keras.layers import Input, Concatenate, GlobalAvgPool2D, GlobalMaxPool2D, Subtract, Multiply, Dense, Flatten, Dropout
# from keras.models import Model, model_from_json
from keras.optimizers import Adam
# from keras_vggface.utils import preprocess_input
from keras.callbacks import ModelCheckpoint

# global variables
learning_rate       = 0.00001
epochs              = 100
batch_size          = 7
steps_per_epoch     = 200
validation_steps    = 200
input_shape         = (224, 224, 3)
num_classes         = 5
# validation_families = ["F001", "F005", "F021", "F023", "F044", "F048", "F063", "F070", "F071", "F086", "F094", "F097", "F099"]


# create and save the model structure
def create_model(model_name):
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # model.add(Conv2D(64, (5, 5), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dense(num_classes, activation='sigmoid'))


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

    # save model to json file
    model_file = open("models/" + model_name + ".json", "w")
    model_file.write(model.to_json())
    model_file.close()

    # print a summary of the model    
    model.summary()

    return model


# load image into numpy array
def read_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image,(224,224))
    image = np.array(image).astype(np.float)
    # print(image.shape)
    # input()
    # return preprocess_input(image, version=2)
    return image


# generator for producing batches of images realtion arrays to train on
def my_gen(training_set, image_names_to_paths, batch_size=16):
    # persons = list(image_names_to_paths.keys())
    # num_persons = len(persons)
    # print(num_persons)
    # input()
    while True:
        training_batch = random.sample(training_set, batch_size)
        labels = []
        batch_image_paths = []
        for training_sample in training_batch:
            batch_image_paths.append(image_names_to_paths[training_sample[0]])
            labels.append(training_sample[1])
        # print('________________')
        # print(batch_image_paths)
        # print('----------------')
        # print(labels)
        # print('________________')
        # input()
        
        # batch_images = np.array([read_image(image_path) for image_path in batch_image_paths])
        batch_images = []
        for image_path in batch_image_paths:
            # print('__', image_path)
            temp = read_image(image_path)
            # print('__', temp.shape)
            batch_images.append(temp)
        # batch_images = np.array(batch_images)

        print(np.array([batch_images]).shape)
        print(np.array([labels]).shape)
        input()

        # training_batch_final = ([batch_images], labels)

        yield ([np.array(batch_images)], [np.array(labels)])

    return


# load a previosly saved model (for training or predicting)
# def load_saved_model(model_name):
#     model_structure_path = "models/" + model_name.split("_all_epochs")[0] + ".json"
#     model_weights_path = "models/" + model_name + ".h5"
#     model_structure_file = open(model_structure_path, "r")
#     # load the model structure
#     model = model_from_json(model_structure_file.read())
#     # load the model weights
#     model.load_weights(model_weights_path)
#     model_structure_file.close()

#     return model


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
    for i in range(len(training_labels.id_code.values)):
        image_name = training_labels.id_code.values[i]
        label      = [0]*5
        label[training_labels.diagnosis.values[i]] = 1
        if image_name in image_names_to_paths:
            training_set.append((image_name, label))
    # print(training_set)


    # input()

    # print(image_names_to_paths)

    #     key = my_split[-3] + '/' + my_split[-2]
    #     if    key in image_names_to_paths: image_names_to_paths[key].append(edited_image_path)
    #     else: image_names_to_paths[key] = [edited_image_path]

    

    # training_set = []
    # validation_set = []
    # for relationship in relationships_list:
    #     if relationship[0] in image_names_to_paths and relationship[1] in image_names_to_paths:
    #         training_set.append(relationship)
    #         for fam in validation_families:
    #             if fam in relationship[0]: 
    #                 validation_set.append(relationship)
    #                 training_set.pop()
    #                 break

    if (not continue_training):
        print('creating new model...')
        model = create_model(model_name)
    else:
        print('loading saved model to continue training...')
        # model = load_saved_model(model_name)
        model = create_model(model_name)
    
    model.compile(loss="categorical_crossentropy", metrics=['acc'], optimizer=Adam(learning_rate))
    
    # set up checkpoint so we can save the model (if better accuracy) after EVERY epoch. So if it crashes, or converges early, we're all good.
    model_weights_file = "models/" + model_name + ".h5"

    # checkpoint = ModelCheckpoint(model_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    checkpoint = ModelCheckpoint(model_weights_file, monitor='acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    print("length of training set is", len(training_set), "and length if validation set is", len(validation_set))
    print("fitting model...")
    model.fit_generator(my_gen(training_set, image_names_to_paths, batch_size=batch_size), \
                        callbacks=callbacks_list, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch)




    # model.fit_generator(my_gen(training_set, image_names_to_paths, batch_size=batch_size), validation_data=my_gen(validation_set, image_names_to_paths, batch_size=batch_size), \
    #                     callbacks=callbacks_list, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    # # this will be the final model from training on all epocs (probably overfitted), the above checkpointed saved model will be the model based on best validation set accuracy (early convergence)
    # model.save_weights("models/" + model_name + "_all_epochs.h5")

    return


# predict test and save results
# def predict(model_name):
#     print("beginning testing routine with model " + model_name + "...")
#     relationships = pd.read_csv("input/test_relationships.csv")

#     print('loading model...')
#     model = load_saved_model(model_name)

#     predictions = []
#     print("predicting... ")
#     for i in tqdm.tqdm(range(len(relationships.img_pair.values))):
#         image_pair = relationships.img_pair.values[i].split("-")
#         image_pair = [read_image("input/test/" + image) for image in image_pair]
#         prediction = model.predict(  [ [image_pair[0]], [image_pair[1]] ]   )

#         if prediction[0][0] > 0.5:
#             predictions.append(1)
#         else:
#             predictions.append(0)

#     relationships["is_related"] = predictions
#     relationships.to_csv("submissions/" + model_name + ".csv", index=False)

#     return


""" 
    - use train and predict as you like (comment out train if you only want to predict and vice versa)
    - don't include path or extension for model name: "model_name".csv is submission file, "model_name".json is model structure, "model_name".h5 is model weights
    - if training a new model (new training method or new model structure), or wanting to not overwrite previous models, use a new model name in 'train', and reference it in 'predict'
"""
if __name__ == "__main__":
    # trains model and saves it with given name (if continuing training on a model, set 2nd argmnt to True)
    train(model_name = "model_1", continue_training=False)
    # takes the referenced model and creates submission file from testing images
# predict(model_name = "model_1")