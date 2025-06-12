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
from keras.applications.densenet import DenseNet169, DenseNet121, DenseNet201, preprocess_input
from keras import Model, backend, Sequential
from keras.optimizers import Adam, SGD
import tensorflow as tf
from keras.applications.efficientnet import EfficientNetB3, EfficientNetB7
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

"""
This file can be used to get the results when we try to predict KL score based on the predicted, individual
feature scores
"""

"""
This function used to get the region for osteophytes
Args:
    - path (String): path to examined hip joint image
    - c_row_lookup (dictionary): dictionary that tells us the row which contains the 1-feature region coordinates of joint is stored
    - cropped_image_path (String): path to folder in which all the cropped images are stored
    - isolated (Boolean): are we using isolated region or 6-feature region
    - coordinates (pd df): dataframe which contains the coordinates of the 1-feature regions
    - mean (Float): mean of z-score normalization
    - std (Float): std of z-score normalization
    - type (String): which feature are you examining: SAO, SFO, IO or KL?   
Return:
    - image (np array): pixel wise image
    - original_image (cv2 image): image before z-score normalisation. Only needed for gradcam
"""


def get_input_osteo(path, c_row_lookup, cropped_image_path, isolated, coordinates, mean, std, type):
    # access the location where 1-feature regions coordinates are stored in coordinates file
    key = os.path.basename(path)
    key = key.replace(".jpg", "")
    row_id = int(c_row_lookup[key])

    name = os.path.basename(path)
    new_path = os.path.join(cropped_image_path, name)

    # access image and normalise orientation
    image = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
    if str(path).__contains__("APL"):
        image = cv2.flip(image, 1)
    height, width = image.shape

    # get type
    if type == "SAO":
        type = "OSTS_acet"
    elif type == "SFO":
        type = "OSTS_fem"
    elif type == "IO":
        type = "OSTI"
    else:
        type = "FULL"
        isolated = False

    # get region
    if isolated:
        xmin, ymin, xmax, ymax = map(float, coordinates.iloc[row_id][type].strip(
            '][').split(', '))
    else:
        xmin = 5 / 224 * width
        ymin = 5 / 224 * height
        xmax = 180 / 224 * width
        ymax = 180 / 224 * height

    if xmax - xmin > ymax - ymin:
        diff = (xmax - xmin) - (ymax - ymin)
        s = 0
        if int(ymin - diff / 2) < 0:
            s = int(ymin - diff / 2) * -1
        new_ymin = max(int(ymin - diff / 2), 0)
        new_ymax = min(int(ymax + diff / 2) + s, height - 1)
        new_xmin = int(xmin)
        new_xmax = int(xmax)
    else:
        diff = (ymax - ymin) - (xmax - xmin)
        s = 0
        if int(xmin - diff / 2) < 0:
            s = int(xmin - diff / 2) * -1
        new_xmin = max(int(xmin - diff / 2), 0)
        new_xmax = min(int(xmax + diff / 2) + s, width - 1)
        new_ymin = int(ymin)
        new_ymax = int(ymax)

    # pre-process image
    image = image[new_ymin:new_ymax, new_xmin:new_xmax]
    image = cv2.resize(image, (224, 224))
    original_image = image
    image = image.astype(np.float32)
    image = image - mean
    image = image / std
    image = np.dstack([image, image, image])
    return image, original_image


"""
This function used to get the region for JSN
Args:
    - path (String): path to examined hip joint image
    - c_row_lookup (dictionary): dictionary that tells us the row which contains the 1-feature region coordinates of joint is stored
    - cropped_image_path (String): path to folder in which all the cropped images are stored
    - isolated (Boolean): are we using isolated region or 6-feature region
    - coordinates (pd df): dataframe which contains the coordinates of the 1-feature regions
    - mean (Float): mean of z-score normalization
    - std (Float): std of z-score normalization
    - type (String): which feature are you examining: SAO, SFO, IO or KL?   
Return:
    - image (np array): pixel wise image
    - original_image (cv2 image): image before z-score normalisation. Only needed for gradcam
"""


def get_input_jsn(path, c_row_lookup, cropped_image_path, isolated, coordinates, mean, std, type):
    # access the location where 1-feature regions coordinates are stored in coordinates file
    key = os.path.basename(path)
    key = key.replace(".jpg", "")
    row_id = int(c_row_lookup[key])

    name = os.path.basename(path)
    new_path = os.path.join(cropped_image_path, name)

    # access image and normalise orientation
    image = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
    if str(path).__contains__("APL"):
        image = cv2.flip(image, 1)
    height, width = image.shape

    # get type
    if type == "sJSN":
        type = type
    elif type == "mJSN":
        type = type
    else:
        type = "FULL"
        isolated = False

    # get region
    if isolated:
        xmin_jsn, ymin_jsn, xmax_jsn, ymax_jsn = map(float, coordinates.iloc[row_id]["Flattening"].strip(
            '][').split(', '))
        if type == "sJSN":
            x1, y1, x2, y2 = map(float, coordinates.iloc[row_id]["OSTS_acet"].strip(
                '][').split(', '))
            xmin = x1 + (x2 - x1) / 2 - 5
            xmax = xmax_jsn - 5
            ymax = ymin_jsn + (ymax_jsn - ymin_jsn) / 2
            ymin = ymin_jsn - (ymax_jsn - ymin_jsn) / 2
        else:
            y_mid = ymin_jsn
            s = 0
            if xmax_jsn + 60 / 224 * width > width:
                s = xmax_jsn + 60 / 224 * width - width
            xmin = xmax_jsn - 80 / 224 * width - s
            xmax = xmax_jsn + 60 / 224 * width
            s = 0
            if y_mid - 30 / 224 * height < 0:
                s = (y_mid - 30 / 224 * height) * -1

            ymin = y_mid - 30 / 224 * height
            ymax = y_mid + 110 / 224 * height + s
    else:
        xmin = 15 / 224 * width
        ymin = 15 / 224 * height
        xmax = 200 / 224 * width
        ymax = 200 / 224 * height

    if xmax - xmin > ymax - ymin:
        diff = (xmax - xmin) - (ymax - ymin)
        s = 0
        if int(ymin - diff / 2) < 0:
            s = int(ymin - diff / 2) * -1
        new_ymin = max(int(ymin - diff / 2), 0)
        new_ymax = min(int(ymax + diff / 2) + s, height - 1)
        new_xmin = int(xmin)
        new_xmax = int(xmax)
    else:
        diff = (ymax - ymin) - (xmax - xmin)
        s = 0
        if int(xmin - diff / 2) < 0:
            s = int(xmin - diff / 2) * -1
        new_xmin = max(int(xmin - diff / 2), 0)
        new_xmax = min(int(xmax + diff / 2) + s, width - 1)
        new_ymin = int(ymin)
        new_ymax = int(ymax)

    # pre-process image
    image = image[new_ymin:new_ymax, new_xmin:new_xmax]
    image = cv2.resize(image, (224, 224))
    original_image = image
    image = image.astype(np.float32)
    image = image - mean
    image = image / std
    image = np.dstack([image, image, image])
    return image, original_image


"""
This function is used to evaluate model on test/validation dataset
Args:
    - cropped_image_path (String): path to folder in which all the cropped images are stored
    - valid_df (pd df): contains validation/test data
    - coordinates_path (Path): path where the file with the coordinates of all isolated regions is located
Return:
    - image (np array): pixel wise image
    - original_image (cv2 image): image before z-score normalisation. Only needed for gradcam
"""


def eval(cropped_image_path, valid_df, coordinates_path):
    coordinates = pd.read_csv(coordinates_path)
    c_row_lookup = {}
    for i in coordinates.index:
        c_row_lookup[coordinates['image_paths'][i]] = str(i)
    coordinates.fillna(-1, inplace=True)

    # please specify the path of your models
    model_sfo = keras.models.load_model("SFO.h5")
    model_io = keras.models.load_model("IO.h5")
    model_sao = keras.models.load_model("SAO.h5")
    model_sup = keras.models.load_model("sJSN.h5")
    model_med = keras.models.load_model("mJSN.h5")
    model_kl = keras.models.load_model("KL_model1.h5")

    # please specify the means for z-score normalisation
    mean_sfo = 132.21507421253838
    mean_io = 132.21507421253838
    mean_sao = 132.21507421253838
    mean_sup = 132.21507421253838
    mean_med = 132.21507421253838

    # please specify the std for z-score normalisation
    std_sfo = 51.81800506461007
    std_io = 51.81800506461007
    std_sao = 51.81800506461007
    std_sup = 51.81800506461007
    std_med = 51.81800506461007

    # please specify what region you want to use
    isolated_sfo = 51.81800506461007
    isolated_io = 51.81800506461007
    isolated_sao = 51.81800506461007
    isolated_sup = 51.81800506461007
    isolated_med = 51.81800506461007

    rows = len(valid_df.axes[0])
    cats = ["JSN"]
    count = 0
    targets = []
    preds = []
    # go through each file
    for j in range(0, 1):
        for i in range(0, rows):
            target = int(valid_df.iloc[i][cats[j]])
            count += 1
            path = valid_df.iloc[i]["image_paths"]

            # get prediction for IO
            image, original_image = get_input_osteo(path, c_row_lookup, cropped_image_path, isolated_io, coordinates,
                                                    mean_io, std_io, "IO")
            image = np.expand_dims(image, axis=0)
            result_osti = model_io.predict(image)
            outcome_osti = np.argmax(result_osti[0])

            # get prediction for SFO
            image, original_image = get_input_osteo(path, c_row_lookup, cropped_image_path, isolated_sfo, coordinates,
                                                    mean_sfo, std_sfo, "SFO")
            image = np.expand_dims(image, axis=0)
            result_sfo = model_sfo.predict(image)
            outcome_sfo = np.argmax(result_sfo[0])

            # get prediction for SAO
            image, original_image = get_input_osteo(path, c_row_lookup, cropped_image_path, isolated_sao, coordinates,
                                                    mean_sao, std_sao, "SAO")
            image = np.expand_dims(image, axis=0)
            result_sao = model_sao.predict(image)
            outcome_sao = np.argmax(result_sao[0])

            # get prediction for superior JSN
            image, original_image = get_input_jsn(path, c_row_lookup, cropped_image_path, isolated_sup, coordinates,
                                                  mean_sup, std_sup, "sJSN")
            image = np.expand_dims(image, axis=0)
            result_sup = model_sup.predict(image)
            outcome_sup = np.argmax(result_sup[0])

            # get prediction for medial JSN
            image, original_image = get_input_jsn(path, c_row_lookup, cropped_image_path, isolated_med, coordinates,
                                                  mean_med, std_med, "mJSN")
            image = np.expand_dims(image, axis=0)
            result_med = model_med.predict(image)
            outcome_med = np.argmax(result_med[0])

            # define input with the predictions. keep same format as in KL_Formula.py
            # and predict kl score
            input_val = [outcome_sup, outcome_med, outcome_osti, outcome_sao, outcome_sfo]
            input_val = np.expand_dims(np.asarray(input_val), axis=0)
            result = model_kl.predict(input_val)
            outcome = np.argmax(result[0])

            preds.append(outcome)
            targets.append(target)

        # print confusion matrix
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(3) + .5, [0, 1, 2], rotation=90)
        plt.yticks(np.arange(3) + .5, [0, 1, 2], rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()


"""
This function is used to get the dataframes class0_df, class1_df, class2_df and the ones for the 
validation and test datasets. These df's contain only examples from their category. Example: class0_df
contains the path and the label of all training hip joints which has severity 0 for the examined feature.
This structure is necessary to implement the balanced mini-batches

Args:
    score_check_path (Path): path to the csv file that contains the feature scores for all CHECK hip joints. called feature_scores.
    train_path (Path): path to the folder with the training examples. Only for CHECK since we used in check this particular folder structure 
    mentioned in readme
    valid_path (Path): path to the folder with the validation examples. Only for CHECK. 
    test_path (Path): path to the folder with the test examples. Only for CHECK.
Returns:
    class0_df (Pandas df): contains two columns, image_paths and JSN. image_paths is path to image. JSN is respective severity of hip joint.
    contains all training joints with label = 0 
    class1_df (Pandas df): contains two columns, image_paths and JSN. image_paths is path to image. JSN is respective severity of hip joint.
    contains all training joints with label = 1
    class2_df (Pandas df): contains two columns, image_paths and JSN. image_paths is path to image. JSN is respective severity of hip joint.
    contains all training joints with label = 2
    valid_df (Pandas df): contains two columns, image_paths and JSN. image_paths is path to image. JSN is respective severity of hip joint.
    contains all validation joints
    test_df (Pandas df): contains two columns, image_paths and JSN. image_paths is path to image. JSN is respective severity of hip joint.
    contains all test joints
"""


def get_data(score_check_path, train_path, valid_path, test_path):
    # get the feature score file
    data = pd.read_csv(score_check_path)
    row_lookup = {}
    for i in data.index:
        row_lookup[data['id'][i]] = str(i)
    data.fillna(-1, inplace=True)

    # define list for the separation
    class0_paths = []
    class0_labels = []
    class1_paths = []
    class1_labels = []
    class2_paths = []
    class2_labels = []
    for d in [valid_path, train_path, test_path]:
        count = 0
        image_paths = []
        labels = []
        classes = os.listdir(d)

        for class_name in classes:
            class_path = os.path.join(d, class_name)
            image_ids = os.listdir(class_path)
            # go through each image and collect all the features. if a features severity is not defined in
            # the ground truth label, skip image
            for id in image_ids:

                key = str(id).replace("Cropped_", "")
                key = key.replace(".jpg", "")
                val = key.split("_")
                key = str(int(val[0])) + "_" + val[1] + "_" + val[2]
                row_id = int(row_lookup[key])

                # get IO score
                label = max(float(data.iloc[row_id]['OSTI_acet']), float(data.iloc[row_id]['OSTI_fem']))
                if label not in [0, 1, 2, 3]:
                    continue

                # get SAO score
                label = float(data.iloc[row_id]['OSTS_acet'])
                if label not in [0, 1, 2, 3]:
                    continue

                # get SFO score
                label = float(data.iloc[row_id]['OSTS_fem'])
                if label not in [0, 1, 2, 3]:
                    continue

                # get superior JSN score
                label = float(data.iloc[row_id]['AP_JSN_sup'])
                if label not in [0, 1, 2, 3]:
                    continue

                # get medial JSN score
                label = float(data.iloc[row_id]['AP_JSN_med'])
                if label not in [0, 1, 2, 3]:
                    continue

                # get KL score
                label = float(data.iloc[row_id]['KL_def'])
                if label not in [0, 1, 2, 3]:
                    continue

                # assign to respective set
                if d == train_path:
                    image_path = os.path.join(class_path, id)
                    if label == 0:
                        class0_paths.append(image_path)
                        class0_labels.append(0)
                    elif label == 1:
                        class1_paths.append(image_path)
                        class1_labels.append(1)
                    elif label == 2 or label == 3:
                        class2_paths.append(image_path)
                        class2_labels.append(2)
                    else:
                        continue
                else:
                    image_path = os.path.join(class_path, id)
                    if label == 0:
                        label = 0
                    elif label == 1:
                        label = 1
                    elif label == 2:
                        label = 2
                    elif label == 3:
                        label = 2
                    else:
                        continue
                    labels.append(label)
                    image_paths.append(image_path)

        # create datasets
        if d == train_path:
            class0_df = pd.concat([pd.Series(class0_paths, name='image_paths'), pd.Series(class0_labels, name='JSN')],
                                  axis=1)
            class1_df = pd.concat([pd.Series(class1_paths, name='image_paths'), pd.Series(class1_labels, name='JSN')],
                                  axis=1)
            class2_df = pd.concat([pd.Series(class2_paths, name='image_paths'), pd.Series(class2_labels, name='JSN')],
                                  axis=1)
        else:
            image_series = pd.Series(image_paths, name='image_paths')
            label_series = pd.Series(labels, name="JSN")
            pdf = pd.concat([image_series, label_series], axis=1)
            if d == test_path:
                test_df = pdf
            else:
                valid_df = pdf
    return class0_df, class1_df, class2_df, valid_df, test_df


if __name__ == '__main__':
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
    # path to the file with the coordinates of the 1-feature regions. but, they shouldn't be normalized
    coordinates_path = Path(
        PureWindowsPath(r"coordinates_Proper.csv"))
    # path to folder with unnormalized cropped hip joint images of CHECK
    cropped_image_path = r"Cropped_Images"

    class0_df, class1_df, class2_df, valid_df, test_df = get_data(score_check_path, train_path, valid_path, test_path)
    # in eval, you have to define some things
    eval(cropped_image_path, test_df, coordinates_path)
