import re
from glob import glob
import pandas as pd
import random
import numpy as np
import cv2
import tqdm

from keras.models import Sequential, model_from_json, Model
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Activation, Dropout, Input, Concatenate, GlobalAvgPool2D, GlobalMaxPool2D, Subtract, Multiply, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras_vggface.utils import preprocess_input
from keras.callbacks import Callback, ModelCheckpoint
from keras.applications import DenseNet121

from sklearn.metrics import cohen_kappa_score

# global variables
learning_rate       = 0.00001
epochs              = 100
batch_size          = 16
steps_per_epoch     = 100
validation_steps    = 100
num_classes         = 4
label_dict          = {0:[0,0,0,0], 1:[1,0,0,0], 2:[1,1,0,0], 3:[1,1,1,0], 4:[1,1,1,1]}
label_dict_rev      = {'[0 0 0 0]':0, '[1 0 0 0]':1, '[1 1 0 0]':2, '[1 1 1 0]':3, '[1 1 1 1]':4}

densenet = DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224,224,3)
)


class Metrics(Callback):
    def __init__(self, val_data, batch_size = 16):
        super().__init__()
        self.validation_data = val_data
        self.batch_size = batch_size

    def on_train_begin(self, logs={}):
        self.val_kappas = []

    def on_epoch_end(self, epoch, logs={}):
        val_data_set, val_labels = next(self.validation_data)
        
        val_predictions = self.model.predict(val_data_set)

        predictions = []

        for prediction in val_predictions:
            min_diff = 100
            min_diff_id = 0

            for key in label_dict:
                abs_diff = sum(map(lambda x: x*(-1) if x < 0 else x, prediction-label_dict[key]))
                if abs_diff < min_diff:
                    min_diff = abs_diff
                    min_diff_id = key

            predictions.append(min_diff_id)

        actual = []
        for label in val_labels[0]:
            actual.append(label_dict_rev[str(label)])

        _val_kappa = cohen_kappa_score(
            predictions,
            actual, 
            weights='quadratic'
        )

        self.val_kappas.append(_val_kappa)

        print(f"val_kappa: {_val_kappa:.4f}")
        
        if _val_kappa == max(self.val_kappas):
            print("Validation Kappa has improved. Saving model.")
            self.model.save('models/model.h5')
        else:
            print("Validation Kappa has NOT improved...")

        return


# create model
def create_model(model_name):
    # .... model 1 begin
    model = Sequential()
    model.add(densenet)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=Adam(0.00001))
    # .... model 1 end


    # .... model 2 begin
    
    # .... model 2 end


    # .... model 3 begin
    
    # .... model 3 end

    return model


def resize_image(image):
    height, width, num_channels = map(float, image.shape)
    
    if height > width: 
        scaled_height = 224
        scaled_width = int(width*(224/height))
    else:              
        scaled_height = int(height*(224/width))
        scaled_width = 224

    scaled_image = cv2.resize(image, (scaled_width, scaled_height), interpolation = cv2.INTER_AREA)

    final_image = np.ones((500,500,3), dtype=np.uint8)
    final_image = np.array([[[0]*3]*224]*224, dtype=np.uint8)
    
    x_offset = (final_image.shape[0] - scaled_image.shape[0]) // 2
    y_offset = (final_image.shape[1] - scaled_image.shape[1]) // 2
    
    final_image[x_offset:(x_offset+scaled_image.shape[0]), y_offset:(y_offset+scaled_image.shape[1]), 0:3] = scaled_image

    return final_image


# load image into numpy array
def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = resize_image(image)
    # image = np.array(image).astype(np.float)

    # cv2.imshow("scaled_image image", image) 
    # cv2.waitKey(0)
    # cv2.destroyAllWindows() 

    return image

# read_image('C:/Users/will/code/python/kaggle_comp_blindness/input/train/0a4e1a29ffff.png')


# generator for producing batches of images to train on
def my_gen(training_set, image_names_to_paths, batch_size=16):
    flip_v = random.randint(0,1)
    flip_h = random.randint(0,1)
    while True:
        training_batch = random.sample(training_set, batch_size)
        labels = []
        batch_image_paths = []
        for training_sample in training_batch:
            batch_image_paths.append(image_names_to_paths[training_sample[0]])
            labels.append(training_sample[1])
        
        batch_images = []
        for image_path in batch_image_paths:
            np_image = read_image(image_path)
            # apply transformations
            # if flip_v: np_image = np.flip(np_image, 0)
            # if flip_h: np_image = np.flip(np_image, 1)

            # np_array_transformation_1 = np.flip(np_image, 0)
            # np_array_transformation_2 = np.flip(np_image, 1)
            # np_array_transformation_3 = np.flip(np_array_transformation_2, 0)

            batch_images.append(np_image)
            # batch_images.append(np_array_transformation_1)
            # batch_images.append(np_array_transformation_2)
            # batch_images.append(np_array_transformation_3)

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

    image_names_to_paths = {}
    for image_path in all_image_paths:
        # for dealing with annoying windows path formatting
        edited_image_path = image_path.replace('\\', '/')
        parsed_path = edited_image_path.split('/')
        key = parsed_path[-1].split('.')[0]
        image_names_to_paths[key] = edited_image_path
    
    training_set = []
    validation_set = []

    label_distribution = {0:0, 1:0, 2:0, 3:0, 4:0}
    val_label_distribution = {0:0, 1:0, 2:0, 3:0, 4:0}
    train_label_distribution = {0:0, 1:0, 2:0, 3:0, 4:0}

    num_validation = len(training_labels) // 5
    validation_indicies = random.sample(range(len(training_labels)), num_validation)

    for i in range(len(training_labels.id_code.values)):
        image_name = training_labels.id_code.values[i]
        if image_name in image_names_to_paths:
            label = label_dict[training_labels.diagnosis.values[i]]
            # label[training_labels.diagnosis.values[i]] = 1
            label_distribution[training_labels.diagnosis.values[i]] += 1
            if i in validation_indicies:
                validation_set.append((image_name, label))
                val_label_distribution[training_labels.diagnosis.values[i]] += 1
            else:
                training_set.append((image_name, label))
                train_label_distribution[training_labels.diagnosis.values[i]] += 1
    
    if (not continue_training):
        print('creating new model structure...')
        model = create_model(model_name)
        # save model to json file
        print('saving new model structure...')
        model_file = open("models/" + model_name + ".json", "w")
        model_file.write(model.to_json())
        model_file.close()
    else:
        print('loading saved model to continue training...')
        model = load_saved_model(model_name)
    
    # print a summary of the model    
    model.summary()
    
    # model.compile(loss="binary_crossentropy", metrics=['accuracy'], optimizer=Adam(learning_rate))
    
    # set up checkpoint so we can save the model (if better accuracy) after EVERY epoch. So if it crashes, or converges early, we're all good.
    model_weights_file = "models/" + model_name + ".h5"

    # kappa accuracy
    # kappa_metrics = Metrics(val_data = my_gen(validation_set, image_names_to_paths, batch_size=batch_size))
    # callbacks_list = [kappa_metrics]

    # accuracy is determined based on validation data, not train data, and model save file will only be updated if accuracy on val data is increased
    checkpoint = ModelCheckpoint(model_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    print("overall label distribution\n", label_distribution)
    print("train label distribution\n", train_label_distribution)
    print("val label distribution\n", val_label_distribution)

    print("length of training set is", len(training_set), "and length of validation set is", len(validation_set))

    print("fitting model...")

    model.fit_generator(my_gen(training_set, image_names_to_paths, batch_size=batch_size), validation_data=my_gen(validation_set, image_names_to_paths, batch_size=batch_size), \
                        callbacks=callbacks_list, epochs=epochs, verbose=1, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

    # this will be the final model from training on all epocs (probably overfitted), the above checkpointed saved model will be the model based on best validation set accuracy (early convergence)
    # model.save_weights("models/" + model_name + "_all_epochs.h5")

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

        min_diff = 100
        min_diff_id = 0

        for key in label_dict:
            abs_diff = sum(map(lambda x: x*(-1) if x < 0 else x, prediction-label_dict[key]))
            if abs_diff < min_diff:
                min_diff = abs_diff
                min_diff_id = key

        predictions.append(min_diff_id)

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
    train(model_name = "model", continue_training=False)
    # takes the referenced model and creates submission file from testing images
    predict(model_name = "model")