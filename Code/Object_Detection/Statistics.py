import os
from os.path import isfile, join
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights
import numpy as np

"""
This file contains functions to compute the parameters of z-score normalization 
and to compute the mean size of the ground truth bounding boxes. 
"""


"""
This function can be used to compute the parameters for z-score normalization
Args:
    image_path (String): folder in which the images are 
"""
def image_statistics(image_path):
    os.getcwd()
    # get all images in folder
    directory = [file for file in os.listdir(image_path) if isfile(join(image_path, file))]

    pixel_count = 0
    pixel_mean = 0
    M2 = 0

    count = 0
    for file in directory:
        image = cv2.imread(image_path + file, cv2.IMREAD_GRAYSCALE)
        # go through image and compute running mean and std
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                pixel_count += 1
                delta = image[i][j] - pixel_mean
                pixel_mean = pixel_mean + delta / pixel_count
                M2 = M2 + delta * (image[i][j] - pixel_mean)
        count += 1

    pixel_variance = M2 / (pixel_count - 1)
    print(pixel_mean, pow(pixel_variance, 0.5), pixel_count)


"""
This function can be used to compute the anchor sizes of the Faster R-CNN. For this purpose, 
it needs access to the ground truth bounding boxes of the training set
Args:
    image_path (String): folder in which the images are 
    annotation_path (String): folder in which the annotations are 
"""
def bounding_box_statistics(image_path, annotation_path):
    os.getcwd()
    directory = [file for file in os.listdir(annotation_path) if isfile(join(annotation_path, file))]

    # initialize model and change its header
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 3)

    bb_widths = []
    normal_widths = []
    bb_heights = []
    bb_ratios = []
    ratios = []

    # go through each annotation
    for file in directory:
        path = image_path + file.replace(".xml", '.jpg')
        try:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        except Exception:
            continue

        orig_width = image.shape[1]
        orig_height = image.shape[0]
        ratios.append(orig_width / orig_height)

        # transform image so that we can feed it to the model
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)

        # check how Faster-RCNN transforms image. Faster R-CNN has its own function to
        # normalize size of images. We need to know what the sizes of the bb's are
        # in that normalized/transformed image
        # If you don't want to use the default transformation and keep the original sizes of
        # the images, then change the min_size and max_size of the Faster R-CNN
        # accordingly.

        transformed_image, _ = model.transform(image)
        transformed_width = transformed_image.tensors[0].shape[2]
        transformed_height = transformed_image.tensors[0].shape[1]

        tree = ET.parse(annotation_path + file)
        root = tree.getroot()
        for obj in root.iter('object'):
            det_width = 0
            det_height = 0
            for size in root.iter('size'):
                det_width = float(size.find('width').text)
                det_height = float(size.find('height').text)

            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)

            # check whether there is an issue with the size of the image
            if orig_width != det_width or orig_height != det_height:
                print("ATTENTION: ", file, orig_height, det_height, orig_width, det_width)
                """xmin = int((xmin / det_width) * orig_width)
                xmax = int((xmax / det_width) * orig_width)
                ymin = int((ymin / det_height) * orig_height)
                ymax = int((ymax / det_height) * orig_height)

            # BB needs to be squares
            a = (ymax - ymin) - (xmax - xmin)
            if a > 0:
                a = int(abs(a) / 2)
                if xmin - a > 0 and xmax + a < orig_width:
                    xmin -= a
                    xmax += a
            elif a < 0:
                a = int(abs(a) / 2)
                if ymin - a > 0 and ymax + a < orig_height:
                    ymin -= a
                    ymax += a

            # BB needs to be larger than in the annotations
            increase = int((orig_width / 100) * 1.5)
            if ymin - increase > 0 and ymax + increase < orig_height and xmin - increase > 0 and xmax + increase < orig_width:
                xmin -= increase
                xmax += increase
                ymin -= increase
                ymax += increase"""

            # Resize bb according to the transformed image shape
            new_width = (xmax - xmin) / orig_width * transformed_width
            new_height = (ymax - ymin) / orig_height * transformed_height

            bb_widths.append(new_width)
            bb_heights.append(new_height)
            bb_ratios.append(new_height / new_width)
            normal_widths.append(ymax - ymin)

    # print stats
    print("Mean WIDTH : ", np.mean(bb_widths), ", STD: ", np.std(bb_widths), ", Min: ",
          min(bb_widths), ", Max: ", max(bb_widths))
    print("Mean HEIGHT: ", np.mean(bb_heights), ", STD: ", np.std(bb_heights), ", Min: ",
          min(bb_heights), ", Max: ", max(bb_heights))

    print("Mean NORMAL HEIGHT: ", np.mean(normal_widths), ", STD: ", np.std(normal_widths), ", Min: ",
          min(normal_widths), ", Max: ", max(normal_widths))
    print("Mean RATIO: ", np.mean(bb_ratios), ", STD: ", np.std(bb_ratios), ", Min: ",
          min(bb_ratios), ", Max: ", max(bb_ratios))
    print("Mean RATIO: ", np.mean(ratios), ", STD: ", np.std(ratios), ", Min: ",
          min(ratios), ", Max: ", max(ratios))


if __name__ == '__main__':
    # location of your annotation
    annotation_path = 'Dataset\\Training\\Annotations\\'
    # location of your images
    image_path = 'Dataset\\Training\\Images\\'
    bounding_box_statistics(image_path, annotation_path)
    #image_statistics(image_path)
