import os
import random
from pathlib import PureWindowsPath, Path

import cv2
import keras
import numpy as np
import seaborn as sns
import pandas as pd
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import BatchNormalization, Dense, GlobalAveragePooling2D
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet121
from keras import Model
from keras.optimizers import Adam
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

"""
This folder was used to train the models of the experiments for the osteophytes and KL score.
In the code, we explaine how to you can use this file.
"""

"""
This class was taken from https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/.
"""
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()
            print(self.layerName)

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        count = 0
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            count -= 1
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)

            loss = predictions[:, tf.argmax(predictions[0])]

        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)

"""
This class serves as a data generator for the validation dataset.
"""
class CustomDataGenerator(tf.keras.utils.Sequence):

    """
    This is the Initializer function. It needs many inputs to work properly.
    Args:
        - df (Panda Dataframe): This dataframe contains two columns. In the first column, the name of the image is mentioned.
        In the second column is the label. It consists of all datapoint of the validation dataset
        - X_col (String): name of the column in which the names of the images are stored
        - y_col (String): name of the column in which the grades are stored
        - batch_size (int): batch size to compute the loss. Since validation dataset, use 1.
        - input_size (tuple): shape of normalized image.
        - coordinates_path (Path): path where the file with the coordinates of all isolated regions is located
        - cropped_image_path (String): path to folder in which all the cropped images are stored
        - isolated (Boolean): are we using isolated region or 6-feature region
        - mean (Float): mean of z-score normalization
        - std (Float): std of z-score normalization
        - shuffle (Boolean): should we shuffle dataset, yes or no
    """
    def __init__(self, df, X_col, y_col, batch_size, input_size, coordinates_path, cropped_image_path, isolated, mean, std, type, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        self.n = len(self.df)
        self.mean = mean
        self.std = std

        if type == "SAO":
            self.type = "OSTS_acet"
        elif type == "SFO":
            self.type = "OSTS_fem"
        elif type == "IO":
            self.type = "OSTI"
        else:
            self.type = "FULL"
            isolated = False
        self.coordinates = pd.read_csv(coordinates_path)
        self.c_row_lookup = {}
        # get the coordinates of the 1-feature regions
        for i in self.coordinates.index:
            self.c_row_lookup[self.coordinates['image_paths'][i]] = str(i)
        self.coordinates.fillna(-1, inplace=True)

        self.cropped_image_path = cropped_image_path
        self.isolated = isolated

    """
    This function is called at the end of an epoch. We can use it if we want to shuffle the dataset.
    Not used for validation dataset. 
    """
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    """
    This function is used to get the input image. 
    Args:
        - path (String): path where image is stored
    Return:
        - image(numpy array): image
    """
    def __get_input(self, path):
        key = os.path.basename(path)
        key = key.replace(".jpg", "")
        # check in which row image is in the coordinate file
        row_id = int(self.c_row_lookup[key])

        name = os.path.basename(path)
        # get path to cropped hip joint image
        new_path = os.path.join(self.cropped_image_path, name)

        # read image
        image = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
        # normalize orientation of joint
        if str(path).__contains__("APL"):
            image = cv2.flip(image, 1)
        height, width = image.shape

        # if isolated region, access coordinates from self.coordinates
        if self.isolated:
            xmin, ymin, xmax, ymax = map(float, self.coordinates.iloc[row_id][self.type].strip(
                '][').split(', '))
        else:
            xmin = 5 / 224 * width
            ymin = 5 / 224 * height
            xmax = 180 / 224 * width
            ymax = 180 / 224 * height


        # ensure that regions are square shaped
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

        # get from original cropped image the aimed region
        image = image[new_ymin:new_ymax, new_xmin:new_xmax]
        # normalize its shape
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)
        # preprocess it
        image = image - self.mean
        image = image / self.std
        image = np.dstack([image, image, image])
        return image

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
        path_batch = batches[self.X_col]
        jsn = batches[self.y_col]
        X_batch = np.asarray([self.__get_input(x) for x in path_batch])
        # one hot encoding for softmax
        y_jsn_batch = np.asarray([to_categorical(y, 3) for y in jsn])

        return X_batch, y_jsn_batch

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
This class serves as a data generator for the validation dataset.
"""
class CustomTrainingGenerator(tf.keras.utils.Sequence):
    """
        This is the Initializer function. It needs many inputs to work properly.
        Args:
            - df (Panda Dataframe): This dataframe contains two columns. In the first column, the name of the image is mentioned.
            In the second column is the label. It consists of all datapoint of the training dataset
            - X_col (String): name of the column in which the names of the images are stored
            - y_col (String): name of the column in which the grades are stored
            - batch_size (array of three entries): batch size to compute the loss.
            - input_size (tuple): shape of normalized image.
            - coordinates_path (Path): path where the file with the coordinates of all isolated regions is located
            - cropped_image_path (String): path to folder in which all the cropped images are stored
            - isolated (Boolean): are we using isolated region or 6-feature region
            - mean (Float): mean of z-score normalization
            - std (Float): std of z-score normalization
            - shuffle (Boolean): should we shuffle dataset, yes or no
        """
    def __init__(self, df, X_col, y_col, batch_size, n, input_size, coordinates_path, cropped_image_path, isolated, mean, std, type, shuffle=True):

        self.class0_df, self.class2_df, self.class3_df = df
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.n = n

        self.input_size = input_size
        self.shuffle = False
        self.steps = 0

        self.mean = mean
        self.std = std

        if type == "SAO":
            self.type = "OSTS_acet"
        elif type == "SFO":
            self.type = "OSTS_fem"
        elif type == "IO":
            self.type = "OSTI"
        else:
            self.type = "FULL"
            isolated = False

        self.coordinates = pd.read_csv(coordinates_path)
        self.c_row_lookup = {}
        # get the coordinates of the 1-feature regions
        for i in self.coordinates.index:
            self.c_row_lookup[self.coordinates['image_paths'][i]] = str(i)
        self.coordinates.fillna(-1, inplace=True)

        self.cropped_image_path = cropped_image_path
        self.isolated = isolated

    """
        This function is called at the end of an epoch. We can use it if we want to shuffle the dataset.
        Not used. 
    """
    def on_epoch_end(self):
        if self.shuffle:
            for j in range(0, 20):
                self.class0_df = self.class0_df.sample(frac=1).reset_index(drop=True)
                self.class2_df = self.class2_df.sample(frac=1).reset_index(drop=True)
                self.class3_df = self.class3_df.sample(frac=1).reset_index(drop=True)

    """
    This function is used to get the input image. 
    Args:
        - path (String): path where image is stored
        - label (int): grade of joint. only used for debuggin purposes
    Return:
        - image (numpy array): image
    """
    def __get_input(self, path, label):
        key = os.path.basename(path)
        key = key.replace(".jpg", "")
        row_id = int(self.c_row_lookup[key])

        name = os.path.basename(path)
        new_path = os.path.join(self.cropped_image_path, name)

        # get image and normalize orientation
        image = cv2.imread(new_path, cv2.IMREAD_GRAYSCALE)
        if str(path).__contains__("APL"):
            image = cv2.flip(image, 1)

        height, width = image.shape
        # augmentate with probability
        k = random.randint(0, 5)
        transition = False
        zoom_rotate = False
        aug = False

        # select type of augmentation
        if k >= 1:
            k = random.randint(0, 2)
            if k == 0:
                aug = True
            k = random.randint(0, 2)
            if k == 0:
                zoom_rotate = True
            elif 1 <= k <= 2:
                transition = True

        # get region
        if self.isolated:
            xmin, ymin, xmax, ymax = map(float, self.coordinates.iloc[row_id][self.type].strip(
                '][').split(', '))
        else:
            xmin = 5 / 224 * width
            ymin = 5 / 224 * height
            xmax = 180 / 224 * width
            ymax = 180 / 224 * height

        # make sure that region is square shaped
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

        # increase region so that we can rotate better without creating artefacts in final region
        if zoom_rotate:
            right = -16
            if new_xmin + int(right / 2) >= 0 and new_xmax - int(right / 2) <= width - 2 and new_ymin + int(
                    right / 2) >= 0 and new_ymax - int(right / 2) <= height - 2:
                new_xmin = max(0, new_xmin + int(right / 2))
                new_xmax = min(width - 1, new_xmax - int(right / 2))
                new_ymin = max(0, new_ymin + int(right / 2))
                new_ymax = min(height - 1, new_ymax - int(right / 2))
            else:
                zoom_rotate = False

        # translate image
        if transition:
            s = random.randint(max(int(new_xmin / width * 224) * -1, -8), 8)
            u = random.randint(-5, 2)
            s = int(s / 224 * width)
            u = int(u / 224 * height)

            if new_xmin + s >= 0 and new_xmax + s <= width - 2 and new_ymin + u >= 0 and new_ymax + u <= height - 2:
                new_xmin = max(0, new_xmin + s)
                new_xmax = min(width - 2, new_xmax + s)
                new_ymin = max(0, new_ymin + u)
                new_ymax = min(height - 2, new_ymax + u)

        # crop out region
        image = image[max(0, new_ymin):min(new_ymax, height - 1), max(0, new_xmin):min(new_xmax, width - 1)]

        # rotate image
        if zoom_rotate:
            angle = random.randint(-6, 6)
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REPLICATE)
            image = image[8:-8, 8:-8]

        # normalize image size
        image = cv2.resize(image, (224, 224))

        # apply color augmentation or gaussian noise
        if aug:
            k = random.randint(0, 2)
            if k == 0:
                noise = np.zeros(image.shape, np.uint8)
                cv2.randn(noise, 0, 0.7)
                img_noised = cv2.add(image, noise)
                image = np.clip(img_noised, 0, 255).astype(np.uint8)
            if k == 1:
                max_val = image.max()
                min_val = image.min()
                s = random.randint(0, 1)
                if s == 0:
                    alpha = random.uniform(0.6, 1.2)
                    beta = random.uniform(-3, 10)
                else:
                    alpha = random.uniform(0.99, 1.01)
                    beta = random.uniform(min_val * -1 + 20, 255 - max_val - 10)
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        """print(path)
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()"""
        # preprocess image
        image = image.astype(np.float32)
        image = image - self.mean
        image = image / self.std
        image = np.dstack([image, image, image])
        return image

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
        path_batch = [*batches0[self.X_col], *batches2[self.X_col], *batches3[self.X_col]]
        jsn = [*batches0[self.y_col], *batches2[self.y_col], *batches3[self.y_col]]
        X_batch = np.asarray([self.__get_input(x, y) for x, y in zip(path_batch, jsn)])
        y_jsn_batch = np.asarray([to_categorical(y, 3) for y in jsn])

        return X_batch, y_jsn_batch

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
        # if start == 0, shuffle it. mostly gets active in the very first epoch
        if class0_start == 0:
            for i in range(0, 100):
                self.class0_df = self.class0_df.sample(frac=1).reset_index(drop=True)
        # end pointer is at the beginning. this means that we went through entire df
        # therefore, collect data and shuffle dataframe
        if class0_end < class0_start:
            d = [self.class0_df[class0_start:], self.class0_df[:class0_end]]
            batches0 = pd.concat(d)
            for i in range(0, 100):
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
        if class2_start == 0:
            for i in range(0, 100):
                self.class2_df = self.class2_df.sample(frac=1).reset_index(drop=True)
        if class2_end < class2_start:
            d = [self.class2_df[class2_start:], self.class2_df[:class2_end]]
            batches2 = pd.concat(d)
            for i in range(0, 100):
                self.class2_df = self.class2_df.sample(frac=1).reset_index(drop=True)
        else:
            batches2 = self.class2_df[class2_start:class2_end]

        start = self.steps * self.batch_size[2]
        end = (self.steps + 1) * self.batch_size[2]
        class3_start = start % len(self.class3_df)
        class3_end = end % len(self.class3_df)
        if class3_start == 0:
            for i in range(0, 100):
                self.class3_df = self.class3_df.sample(frac=1).reset_index(drop=True)
        if class3_end < class3_start:
            d = [self.class3_df[class3_start:], self.class3_df[:class3_end]]
            batches3 = pd.concat(d)
            for i in range(0, 100):
                self.class3_df = self.class3_df.sample(frac=1).reset_index(drop=True)
        else:
            batches3 = self.class3_df[class3_start:class3_end]

        # until now, we only have the names and labels of the current paths
        # use this function to get the original images and the one hot encodings
        # of the labels
        X, y = self.__get_data(batches0, batches2, batches3)
        self.steps += 1
        return X, y

    """
        This function returns the number of iterations in an epoch
    """
    def __len__(self):
        return self.n // (sum(self.batch_size))

"""
This function is used to get the dataframes class0_df, class1_df, class2_df and the ones for the 
validation and test datasets. These df's contain only examples from their category. Example: class0_df
contains the path and the label of all training hip joints which has severity 0 for the examined feature.
This structure is necessary to implement the balanced mini-batches

Args:
    score_check_path (Path): path to the csv file that contains the feature scores for all CHECK hip joints. called feature_scores.
    score_oi_path (Path): path to the csv file that contains the features scores for all OI hip joints. called OI_scores.
    oi_path (Path): path to the folder with the OI data. 
    train_path (Path): path to the folder with the training examples. Only for CHECK since we used in check this particular folder structure 
    mentioned in readme
    valid_path (Path): path to the folder with the validation examples. Only for CHECK. 
    test_path (Path): path to the folder with the test examples. Only for CHECK.
    oi (Boolean): are you working with OI or with CHECK? If OI, specifiy True. Otherwise, False.
    type (String): which feature are you examining: SAO, SFO, IO or KL?
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
def get_data(score_check_path, score_oi_path, oi_path, train_path, valid_path, test_path, oi, type):

    # access the CHECK feature and KL scores and store them in data df
    # additionally, create a dictionary called row_lookup. It stores the location
    # of each image's row with the feature scores in the data dataframe.
    data = pd.read_csv(score_check_path)
    row_lookup = {}
    for i in data.index:
        row_lookup[data['id'][i]] = str(i)
    data.fillna(-1, inplace=True)

    # access the OI feature and KL scores and store them in oi_data df
    # additionally, create a dictionary called oi_row_lookup. It stores the location
    # of each image's row with the feature scores in the data dataframe.
    oi_data = pd.read_csv(score_oi_path)
    oi_row_lookup = {}
    for i in oi_data.index:
        oi_row_lookup[oi_data['id'][i]] = str(i)
    oi_data.fillna(-1, inplace=True)

    # we will assign the training hip joints to the list of their respective class
    # important to make later the df for the individual classes
    class0_paths = []
    class0_labels = []
    class1_paths = []
    class1_labels = []
    class2_paths = []
    class2_labels = []
    inconsistent = []

    # go through each folder
    for d in [valid_path, train_path, test_path]:
        count = 0
        image_paths = []
        labels = []
        classes = os.listdir(d)

        # CHECK dataset
        if not oi:
            # go through each class folder (see readme)
            for class_name in classes:
                class_path = os.path.join(d, class_name)
                image_ids = os.listdir(class_path)
                # go through each image
                for id in image_ids:

                    # look the row id in dictionary
                    key = str(id).replace("Cropped_", "")
                    key = key.replace(".jpg", "")
                    val = key.split("_")
                    key = str(int(val[0])) + "_" + val[1] + "_" + val[2]
                    row_id = int(row_lookup[key])

                    # access label. only access the severity of examined feature specified by type
                    if type == "SAO":
                        label = float(data.iloc[row_id]['OSTS_acet'])
                    elif type == "SFO":
                        label = float(data.iloc[row_id]['OSTS_fem'])
                    elif type == "IO":
                        label = max(float(data.iloc[row_id]['OSTI_acet']), float(data.iloc[row_id]['OSTI_fem']))
                    elif type == "KL":
                        label = float(data.iloc[row_id]['KL_def'])
                    else:
                        continue
                    # remove redundant labels
                    if label not in [0, 1, 2, 3]:
                        continue

                    # This uncommented code is a template that you can use to output the id of
                    # participants who have inconsistent labels in examined feature
                    # afterward, you can use the id and add it to the list called "inconsistent".
                    # with the below, uncommented if, you can remove them from your training and
                    # validation set.
                    """if key.__contains__("T10"):
                        l = ["T08", "T05", "T02", "T00"]
                        for e in l:
                            previous_hip = key.replace("T10", e)
                            previous_id = int(row_lookup[previous_hip])
                            try:
                                if label < float(data.iloc[previous_id]['OSTS_acet']) and float(
                                        data.iloc[previous_id]['OSTS_acet']) in [0, 1, 2, 3]:
                                    # z.append(str(int(v[0])))
                                    out = True
                                    elements = key.split("_")
                                    print((str(elements[0]), str(elements[2])), ",")
                            except Exception:
                                continue
                    elif key.__contains__("T08"):
                        l = ["T05", "T02", "T00"]
                        for e in l:
                            previous_hip = key.replace("T08", e)
                            previous_id = int(row_lookup[previous_hip])
                            try:
                                if label < float(data.iloc[previous_id]['OSTS_acet']) and float(
                                        data.iloc[previous_id]['OSTS_acet']) in [0, 1, 2, 3]:
                                    # z.append(str(int(v[0])))
                                    out = True
                                    # print(key, label, float(data.iloc[previous_id]['OSTS_fem']))
                                    elements = key.split("_")
                                    print((str(elements[0]), str(elements[2])), ",")
                            except Exception:
                                continue
                    elif key.__contains__("T05"):
                        l = ["T02", "T00"]
                        for e in l:
                            previous_hip = key.replace("T05", e)
                            previous_id = int(row_lookup[previous_hip])
                            try:
                                if label < float(data.iloc[previous_id]['OSTS_acet']) and float(
                                        data.iloc[previous_id]['OSTS_acet']) in [0, 1, 2, 3]:
                                    # z.append(str(int(v[0])))
                                    out = True
                                    # print(key, label, float(data.iloc[previous_id]['OSTS_fem']))
                                    elements = key.split("_")
                                    print((str(elements[0]), str(elements[2])), ",")
                            except Exception:
                                continue
                    elif key.__contains__("T02"):
                        l = ["T00"]
                        for e in l:
                            previous_hip = key.replace("T02", e)
                            previous_id = int(row_lookup[previous_hip])
                            try:
                                if label < float(data.iloc[previous_id]['OSTS_acet']) and float(
                                        data.iloc[previous_id]['OSTS_acet']) in [0, 1, 2, 3]:
                                    # z.append(str(int(v[0])))
                                    out = True
                                    # print(key, label, float(data.iloc[previous_id]['OSTS_fem']))
                                    elements = key.split("_")
                                    print((str(elements[0]), str(elements[2])), ",")
                            except Exception:
                                continue"""

                    """if any(str(key).__contains__(u[0]) and str(key).__contains__(u[1]) for u in inconsistent):
                        continue"""

                    # separate images according to their labels. Relevant for balancing the mini-batches
                    if d == train_path:
                        image_path = os.path.join(class_path, id)
                        # if label == 0, assign datapoint to list of class 0
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
                    # if not training dataset, then mini-batch structure is not needed
                    # therefore, just store path of image and its label in the labels and image_paths lists
                    else:
                        image_path = os.path.join(class_path, id)
                        # merge classes 2 and 3
                        if label == 2 or label == 3:
                            label = 2
                        labels.append(label)
                        image_paths.append(image_path)
        else:
            # For oi, we didn't have access to the KL-grades. Therefore, all hip joint images
            # were added to one folder without any further separation
            image_ids = os.listdir(oi_path)
            for id in image_ids:
                # get in which row image is located in the oi_data using dictionary
                key = str(id).replace("Cropped_", "")
                key = key.replace(".jpg", "")
                val = key.split("_")

                val_image = False
                test_image = False
                number = 0
                # we did the splitting directly with the id of patients
                if key.__contains__("OI00"):
                    key = val[0] + "_" + val[1] + "_" + val[3]
                    number = int(val[1])
                    if int(val[1]) > 9900000:
                        val_image = True
                    if 9900000 >= int(val[1]) > 9800000:
                        test_image = True
                else:
                    number = int(val[0])
                    if int(val[0]) > 9900000:
                        val_image = True
                    if 9900000 >= int(val[0]) > 9800000:
                        test_image = True
                    key = str(int(val[0])) + "_" + val[2]

                try:
                    row_id = int(oi_row_lookup[key])
                    # access severity score of examined feature
                    if type == "SAO":
                        label = float(oi_data.iloc[row_id]['OSTS_acet'])
                    elif type == "SFO":
                        label = float(oi_data.iloc[row_id]['OSTS_acet'])
                    elif type == "IO":
                        label = max(float(oi_data.iloc[row_id]['OSTI_acet']), float(oi_data.iloc[row_id]['OSTI_fem']))
                    else:
                        continue
                except Exception:
                    continue

                # remove redundant scores like 9 (Technical issues) or 5 (implant)
                if label not in [0, 1, 2, 3]:
                    count += 1
                    continue

                # same structure as above. assign image paths and label to lists of respective class
                if d == train_path:
                    if val_image or test_image:
                        continue
                    image_path = os.path.join(oi_path, id)
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
                    image_path = os.path.join(oi_path, id)
                    if d == test_path and not test_image:
                        continue
                    if d == valid_path and not val_image:
                        continue
                    # merge classes 2 and 3
                    if label == 2 or label == 3:
                        label = 2
                    labels.append(label)
                    image_paths.append(image_path)
                    # To balance the mini-batches, you can add a for loop
                    # here to increase the number of samples from the minority
                    # classes in the validation dataset

        # create the df's
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
    # return the df's
    return class0_df, class1_df, class2_df, valid_df, test_df


"""
This function is used to get our DenseNet-121 model.
Return:
    - model (DenseNet121): the denseNet-121 model used in the thesis
"""
def get_model():
    # access densenet 121 pre-trained on ImageNet
    base_model = DenseNet121(include_top=False, input_shape=(224, 224, 3), weights="imagenet")
    regularizer = tf.keras.regularizers.L1L2(0.00001, 0.5)
    # ensure that all convolutional layers are frozen
    base_model.trainable = False

    count = 0
    # only make batch normalization layers trainable
    for layer in base_model.layers:
        if isinstance(layer, BatchNormalization):
            layer.trainable = True
        count += 1
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add classification model with l1l2 penalty
    JSN = Dense(3, activation="softmax", name='JSN', kernel_regularizer=regularizer)(x)
    # create model and return it
    model = Model(inputs=base_model.input, outputs=JSN)
    return model

"""
This function is only used during evaluation. it basically is the getter function from the validation dataloader.
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
def get_input(path, c_row_lookup, cropped_image_path, isolated, coordinates, mean, std, type):
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
This function is used to evaluate model on test/validation dataset
Args:
    - modelpath (String): path where model is stored
    - cropped_image_path (String): path to folder in which all the cropped images are stored
    - valid_df (pd df): contains validation/test data
    - coordinates_path (Path): path where the file with the coordinates of all isolated regions is located
    - isolated (Boolean): are we using isolated region or 6-feature region
    - mean (Float): mean of z-score normalization
    - std (Float): std of z-score normalization
    - type (String): which feature are you examining: SAO, SFO, IO or KL?   
"""
def eval(model_path, cropped_image_path, valid_df, coordinates_path, isolated, mean, std, type):
    # create pd df for the coordinates of the 1-feature regions
    coordinates = pd.read_csv(coordinates_path)
    c_row_lookup = {}
    for i in coordinates.index:
        c_row_lookup[coordinates['image_paths'][i]] = str(i)
    coordinates.fillna(-1, inplace=True)

    # load model
    model = keras.models.load_model(model_path)
    #model = keras.models.load_model("SAO.h5")
    model.trainable = False
    rows = len(valid_df.axes[0])
    cats = ["JSN"]
    count = 0
    targets = []
    preds = []
    for j in range(0, 1):
        for i in range(0, rows):
            target = int(valid_df.iloc[i][cats[j]])
            count += 1
            path = valid_df.iloc[i]["image_paths"]
            # access image and predict
            image, original_image = get_input(path, c_row_lookup, cropped_image_path, isolated, coordinates, mean, std, type)
            image = np.expand_dims(image, axis=0)
            result = model.predict(image)
            outcome = np.argmax(result[0])

            print(count - 1, path, result, int(valid_df.iloc[i][cats[j]]))
            # store predictions and ground truth
            preds.append(outcome)
            targets.append(target)

            # create gradcam images
            """icam = GradCAM(model, 0)
            heatmap = icam.compute_heatmap(image)
            heatmap = cv2.resize(heatmap, (224, 224))

            image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

            (heatmap, output) = icam.overlay_heatmap(heatmap, image, alpha=0.5)
            plt.imshow(output)
            plt.axis('off')
            plt.savefig("GradCam\\" + str(i) + ".jpg")"""

        # create confusion matrix
        cm = confusion_matrix(targets, preds)
        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, vmin=0, fmt='g', cmap='Blues', cbar=False)
        plt.xticks(np.arange(3) + .5, [0, 1, 2], rotation=90)
        plt.yticks(np.arange(3) + .5, [0, 1, 2], rotation=0)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        plt.close()


"""
    This function is called to train the model.
    Args:
        - score_check_path (Path): path to the csv file that contains the feature scores for all CHECK hip joints. called feature_scores.
        - score_oi_path (Path): path to the csv file that contains the features scores for all OI hip joints. called OI_scores.
        - cropped_image_path (String): path to folder in which all the cropped images are stored
        - oi_path (Path): path to the folder with the OI data. 
        - train_path (Path): path to the folder with the training examples. Only for CHECK since we used in check this particular folder structure 
        mentioned in readme
        - valid_path (Path): path to the folder with the validation examples. Only for CHECK. 
        - test_path (Path): path to the folder with the test examples. Only for CHECK.
        - coordinates_path (Path): path where the file with the coordinates of all isolated regions is located
        - oi (Boolean): are you working with OI or with CHECK? If OI, specifiy True. Otherwise, False.
        - isolated (Boolean): are we using isolated region or 6-feature region
        - mean (Float): mean of z-score normalization
        - std (Float): std of z-score normalization
        - type (String): which feature are you examining: SAO, SFO, IO or KL?      
"""
def training(score_check_path, score_oi_path, cropped_image_path, oi_path, train_path, valid_path, test_path, coordinates_path, oi, isolated, mean, std, type, model_path):
    # get the dataframes
    class0_df, class1_df, class2_df, valid_df, test_df = get_data(score_check_path, score_oi_path, oi_path, train_path, valid_path, test_path, oi, type)

    # shuffle the training dataframes
    for j in range(0, 100):
        class0_df = class0_df.sample(frac=1).reset_index(drop=True)
        class1_df = class1_df.sample(frac=1).reset_index(drop=True)
        class2_df = class2_df.sample(frac=1).reset_index(drop=True)

    # define the validation data loader
    valgen = CustomDataGenerator(valid_df,
                                 'image_paths',
                                 'JSN',
                                 batch_size=1, input_size=(224, 224, 3), coordinates_path=coordinates_path, cropped_image_path=cropped_image_path,
                                 isolated=isolated, mean=mean, std=std, type=type, shuffle=False)

    n = min(len(class0_df), len(class1_df), len(class2_df))
    m = max(len(class0_df), len(class1_df), len(class2_df))
    print(n)
    # define training data loader
    traingen = CustomTrainingGenerator([class0_df, class1_df, class2_df],
                                       'image_paths',
                                       'JSN',
                                       batch_size=[10, 10, 10], n=3*n, input_size=(224, 224, 3),
                                       coordinates_path=coordinates_path, cropped_image_path=cropped_image_path,
                                       isolated=isolated, mean=mean, std=std, type=type
                                       )

    # get model
    model = get_model()

    # ensure to reduce learning rate after 3 epochs and always store best model
    anne = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3*int(m/(3*n)), verbose=1, min_lr=0.00001)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)

    # train model with adam and categorical_crossentropy
    opt = Adam(learning_rate=0.0001)
    model.compile(optimizer=opt,
                  loss={'JSN': "categorical_crossentropy"},
                  metrics={'JSN': 'accuracy'})
    model.fit(traingen,
              validation_data=valgen,
              epochs=200,
              callbacks=[checkpoint, anne],
              verbose=1)

if __name__ == '__main__':
    # path to file in which the feature severity of CHECK is stored
    score_check_path = Path(PureWindowsPath(r"feature_scores_original.csv"))
    # path to file in which the feature severity of OI is stored
    score_oi_path = Path(PureWindowsPath(r"OI_scores.csv"))
    # path to the folder with the training examples. Only for CHECK since
    # we used in check this particular folder structure mentioned in readme
    train_path = Path(PureWindowsPath(r'Proper_Dataset\Training'))
    # path to the folder with the CHECK validation examples.
    valid_path = Path(PureWindowsPath(r'Proper_Dataset\Validation'))
    # path to the folder with the CHECK test examples.
    test_path = Path(PureWindowsPath(r'Proper_Dataset\Test'))
    # path to the file with the coordinates of the 1-feature regions. but, they shouldn't be normalized
    coordinates_path = Path(PureWindowsPath(r"coordinates_Proper.csv"))
    # path to folder with unnormalized cropped hip joint images of CHECK
    cropped_image_path = r"Cropped_Images"
    # path to folder with unnormalized cropped hip joint images of OI
    oi_path = Path(PureWindowsPath(r'New_Dataset\OI'))
    # specify feature:
    # KL for Kellgren-Lawrence,
    # SAO for superior acetabular osteophyte,
    # SFO for superior femoral osteophyte,
    # IO for inferior osteophytes
    type = "SFO"
    # oi data yes or no
    oi = False
    # z-score normalisation
    mean = 132.21507421253838
    std = 51.81800506461007
    # path were model is stored
    model_path = "SFO.h5"
    # 1-feature region -> True, else -> False
    isolated = True
    training(score_check_path, score_oi_path, cropped_image_path, oi_path, train_path, valid_path, test_path, coordinates_path, oi, isolated, mean, std, type, model_path)
    """class0_df, class1_df, class2_df, valid_df, test_df = get_data(score_check_path, score_oi_path, oi_path, train_path, valid_path, test_path, oi, type)

    eval(model_path, cropped_image_path, test_df, coordinates_path, isolated, mean, std, type)"""