import os
import random
from pathlib import PureWindowsPath, Path

import cv2
import keras
import numpy as np
import seaborn as sns
import pandas as pd
from keras.applications import ResNet50, InceptionV3
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, Callback
from keras.layers import BatchNormalization, Dropout, Dense, GlobalAveragePooling2D, Multiply, Conv2D, MaxPool2D, \
    Flatten, ReLU, concatenate, Add, UpSampling2D, MaxPooling2D
from keras.utils import to_categorical
from PIL import Image
from keras.applications.densenet import DenseNet169, DenseNet121, DenseNet201
from keras import Model, backend, Sequential
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB3, EfficientNetB7
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

"""
This file was used to train the KL predictor model that uses the scores of the 
selected features to predict the KL score. It was trained with the ground truth labels
of the training examples
"""


"""
This class serves as a data generator for the validation dataset.
"""
class CustomDataGenerator(tf.keras.utils.Sequence):
    """
    Initializer of the dataloader.
    Args:
        - df (pandas df): validation dataset
        - batch_size (int): batch size within a load, usually 1
        - n (int): size of dataset
        - shuffle (Boolean): whether we need to shuffle at the end of an epoch
    """
    def __init__(self, df, batch_size, shuffle=True):
        self.df = df.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(self.df)

    """
        At epoch end, you can shuffle the dataset
    """
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    """
        This function is a getter function. It just returns the list of features scores as an numpy array
        Args:
            - features (list): list containing the feature severities of hip joint
        Return
            - features (np.array): previous list as numpy array 
    """
    def __get_input(self, features):
        return np.array(features)

    """
    This function is used to get the images and grades from defined batch
    Args:
        - batches (Array): names and grades of images. we need to get the true image as array and 
        y as one hot encoding
    Return:
        - X (numpy array): array of image points
        - y (numpy array): array of grades
    """
    def __get_data(self, batches):
        # Generates data containing batch_size samples
        feature_batch = batches["data"]
        label = batches["label"]

        X_batch = np.asarray([self.__get_input(x) for x in feature_batch])
        y_batch = np.asarray([to_categorical(y, 3) for y in label])

        return X_batch, y_batch

    """
    This function is used to get the images from batch
    Args:
        - index (int): index in df
    Return:
        - X (numpy array): array of image points
        - y (numpy array): array of grades
    """
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)
        return X, y

    """
    This function returns the number of iterations in an epoch
    """
    def __len__(self):
        return self.n // self.batch_size

"""
This class serves as a data generator for the training dataset.
"""
class CustomTrainingGenerator(tf.keras.utils.Sequence):

    """
    Initializer of the dataloader.
    Args:
        - df (list of the datasets): dataset of class0, 1 and 2
        - batch_size (list): tells how many times a class must be represented within the mini-batches
        - n (int): size of dataset
        - shuffle (Boolean): whether we need to shuffle at the end of an epoch
    """
    def __init__(self, df, batch_size, n, shuffle=True):

        self.class0_df, self.class2_df, self.class3_df = df
        self.batch_size = batch_size
        self.n = n
        self.shuffle = shuffle
        self.steps = 0

    """
    At epoch end, you can shuffle the dataset
    """
    def on_epoch_end(self):
        self.steps = 0
        if self.shuffle:
            for j in range(0, 3):
                self.class0_df = self.class0_df.sample(frac=1).reset_index(drop=True)
                self.class2_df = self.class2_df.sample(frac=1).reset_index(drop=True)
                self.class3_df = self.class3_df.sample(frac=1).reset_index(drop=True)

    """
    This function is a getter function. It just returns the list of features scores as an numpy array
    Args:
        - features (list): list containing the feature severities of hip joint
    Return
        - features (np.array): previous list as numpy array 
    """
    def __get_input(self, features):
        return np.array(features)

    """
    This function is used to get the (pixel wise) images and grades from defined batch
    Args:
        - batches (Array): names and grades of images. we need to get the true image as array and 
        y as one hot encoding
    Return:
        - X (numpy array): array of image points
        - y (numpy array): array of grades
    """
    def __get_data(self, batches0, batches2, batches3):
        feature_batch = [*batches0["data"], *batches2["data"], *batches3["data"]]
        label = [*batches0["label"], *batches2["label"], *batches3["label"]]
        X_batch = np.asarray([self.__get_input(x) for x in feature_batch])
        y_batch = np.asarray([to_categorical(y, 3) for y in label])
        return X_batch, y_batch

    """
    This function is used to get the images from batch
    Args:
        - index (int): index in df
    Return:
        - X (numpy array): array of image points
        - y (numpy array): array of grades
    """
    def __getitem__(self, index):
        # in the thesis, we mentioned that we balanced our mini-batches. for this reason
        # we defined the three pandas dataframes class0_df, class1_df and class2_df
        # now, we shuffled these dfs in the beginning. therefore, we can just go through the
        # df's and use the datapoints rowwise, from top to bottom. So in the first iteration,
        # we use the first 10 images from each df etc. Once we reach the bottom of one df,
        # we just reshuffle it many times and start from the top. This is what this code
        # is doing
        # get start pos and end pos. images in df between them will be selected
        start = self.steps * self.batch_size[0]
        end = (self.steps + 1) * self.batch_size[0]
        # if we reach the bottom of a df, start from above
        class0_start = start % len(self.class0_df)
        class0_end = end % len(self.class0_df)
        # end pointer is at the beginning. this means that we went through entire df
        # therefore, collect data and shuffle dataframe
        if class0_end < class0_start:
            d = [self.class0_df[class0_start:], self.class0_df[:class0_end]]
            batches0 = pd.concat(d)
            self.class0_df = self.class0_df.sample(frac=1).reset_index(drop=True)
        # if we did not reach end of df, just collect the names and labels of images
        # in new mini-batch
        else:
            batches0 = self.class0_df[class0_start:class0_end]

        # redo the same as above for the other two datasets
        start = self.steps * self.batch_size[1]
        end = (self.steps + 1) * self.batch_size[1]
        class2_start = start % len(self.class2_df)
        class2_end = end % len(self.class2_df)
        if class2_end < class2_start:
            d = [self.class2_df[class2_start:], self.class2_df[:class2_end]]
            batches2 = pd.concat(d)
            self.class2_df = self.class2_df.sample(frac=1).reset_index(drop=True)
        else:
            batches2 = self.class2_df[class2_start:class2_end]

        start = self.steps * self.batch_size[2]
        end = (self.steps + 1) * self.batch_size[2]
        class3_start = start % len(self.class3_df)
        class3_end = end % len(self.class3_df)
        if class3_end < class3_start:
            d = [self.class3_df[class3_start:], self.class3_df[:class3_end]]
            batches3 = pd.concat(d)
            self.class3_df = self.class3_df.sample(frac=1).reset_index(drop=True)
        else:
            batches3 = self.class3_df[class3_start:class3_end]

        # until now, we only have the names and labels of the current paths
        # use this function to get the original images and the one hot encodings
        # of the labels
        X, y = self.__get_data(batches0, batches2, batches3)
        self.steps += 1
        return X, y

    def __len__(self):
        return self.n // (sum(self.batch_size))


"""
This function is used to get the data. The datapoint consists of a list containing the features.
The labels are the KL scores. 
Args:
    - train_path (Path): folder where training hip joints are located
    - valid_path (Path): folder where valid hip joints are located
    - test_path (Path): folder where test hip joints are located
Returns:
    - class0_df (pandas df): df containing features and kl score for all training examples with kl score 0
    - class1_df (pandas df): df containing features and kl score for all training examples with kl score 1
    - class2_df (pandas df): df containing features and kl score for all training examples with kl score 2 and above
    - valid_df (pandas df): df containing features and kl score for all validation examples
    - test_df (pandas df): df containing features and kl score for all test examples 
"""
def get_data(score_path, train_path, valid_path, test_path):
    # access the file containing the feature grades
    data = pd.read_csv(score_path)
    # create dictionary to faster access severity grades given name of file
    row_lookup = {}
    for i in data.index:
        row_lookup[data['id'][i]] = str(i)
    data.fillna(-1, inplace=True)

    # we will assign the training hip joints to the list of their respective class
    # important to make later the df for the individual classes
    class0_data = []
    class0_labels = []
    class1_data = []
    class1_labels = []
    class2_data = []
    class2_labels = []
    for d in [valid_path, train_path, test_path]:
        image_data = []
        labels = []
        classes = os.listdir(d)

        for class_name in classes:
            class_path = os.path.join(d, class_name)
            image_ids = os.listdir(class_path)
            # go through each hip joint image and add a data point to the respective dataset
            for id in image_ids:
                # get name of image to get the row from dictionary
                key = str(id).replace("Cropped_", "")
                key = key.replace(".jpg", "")
                val = key.split("_")
                key = str(int(val[0])) + "_" + val[1] + "_" + val[2]
                # get row id
                row_id = int(row_lookup[key])
                # access all 6 feature and summarize their scores in list
                try:
                    features = [float(data.iloc[row_id]['AP_JSN_sup']), float(data.iloc[row_id]['AP_JSN_med']),
                               max(float(data.iloc[row_id]['OSTI_acet']), float(data.iloc[row_id]['OSTI_fem'])),
                               float(data.iloc[row_id]['OSTS_acet']), float(data.iloc[row_id]['OSTS_fem'])]
                except Exception:
                    continue

                # access kl score
                label = min(2, float(data.iloc[row_id]['KL_def']))

                if any(x in features for x in [-1.0, 0.1, 5, 3.5, 6, 9]):
                    continue

                if any(label == x for x in [-1, 0.1, 5, 3.5, 6, 9]):
                    continue

                # store features scores and label in respective dataset
                if d == train_path:
                    if label == 0:
                        class0_data.append(features)
                        class0_labels.append(0)
                    elif label == 1:
                        class1_data.append(features)
                        class1_labels.append(1)
                    elif label == 2:
                        class2_data.append(features)
                        class2_labels.append(2)
                    elif label == 3:
                        class2_data.append(features)
                        class2_labels.append(2)
                    else:
                        continue
                else:
                    labels.append(label)
                    image_data.append(features)

        # create datasets
        if d == train_path:
            class0_df = pd.concat([pd.Series(class0_data, name='data'), pd.Series(class0_labels, name='label')],
                                  axis=1)
            class1_df = pd.concat([pd.Series(class1_data, name='data'), pd.Series(class1_labels, name='label')],
                                  axis=1)
            class2_df = pd.concat([pd.Series(class2_data, name='data'), pd.Series(class2_labels, name='label')],
                                  axis=1)
        else:
            image_series = pd.Series(image_data, name='data')
            label_series = pd.Series(labels, name="label")
            pdf = pd.concat([image_series, label_series], axis=1)
            if d == test_path:
                test_df = pdf
            else:
                valid_df = pdf
    return class0_df, class1_df, class2_df, valid_df, test_df


"""
This function is used to get the model
Return:
    - model
"""
def get_model():
    model = Sequential()
    model.add(Dense(256, input_dim=5, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(3, activation='softmax', name="KL"))
    print(model.summary())
    return model


"""
This function does the training procedure of the kl formula predictor
"""
if __name__ == '__main__':
    # path to file in which the feature severity of CHECK is stored
    score_check_path = Path(
        PureWindowsPath(r"feature_scores_original.csv"))
    # path to the folder with the training examples. Only for CHECK since
    # we used in check this particular folder structure mentioned in readme
    train_path = Path(
        PureWindowsPath(r'Proper_Dataset\Training'))
    # path to the folder with the CHECK validation examples.
    valid_path = Path(
        PureWindowsPath(r'Proper_Dataset\Validation'))
    # path to the folder with the CHECK test examples.
    test_path = Path(PureWindowsPath(r'Proper_Dataset\Test'))

    # get data
    class0_df, class1_df, class2_df, valid_df, test_df = get_data(score_check_path, train_path, valid_path, test_path)
    print(len(class0_df), len(class1_df), len(class2_df))
    print(len(valid_df), len(test_df))

    # shuffle data
    for j in range(0, 3):
        class0_df = class0_df.sample(frac=1).reset_index(drop=True)
        class1_df = class1_df.sample(frac=1).reset_index(drop=True)
        class2_df = class2_df.sample(frac=1).reset_index(drop=True)

    # create data generators
    valgen = CustomDataGenerator(valid_df,
                                 batch_size=1)

    traingen = CustomTrainingGenerator([class0_df, class1_df, class2_df],
                                       batch_size=[10, 10, 10], n=4662)

    # get model
    model = get_model()

    anne = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1, min_lr=0.00001)
    checkpoint = ModelCheckpoint('KL_model.h5', verbose=1, save_best_only=True)


    # train model
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics='accuracy')
    model.fit(traingen,
              validation_data=valgen,
              epochs=200,
              callbacks=[checkpoint, anne],
              verbose=1)



