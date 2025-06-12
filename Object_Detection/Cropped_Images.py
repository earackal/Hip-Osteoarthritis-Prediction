import os
from os.path import isfile, join
from pathlib import PureWindowsPath, Path
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead


"""
This file was used to save the cropped hip joint images
"""

"""
This function is called to get image
Args: 
    image_path(PureWindowsPath): folder where image is located
    file(String): name of file
Return:
    image(torch): pre-processed image for the object detection model    
"""
def get_image(image_path, file):
    try:
        image = cv2.imread(os.path.join(image_path, file)).astype(np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        return image
    except Exception:
        return None


"""
    This is a very important function. It redefines the RPN head of the pre-trained Faster R-CNN. 
    If you want to change the z-normalization, the anchor sizes or the aspect ratios, please change
    the respective values. 
    Args:
        path (String): path to an .pth file of an already trained Faster R-CNN. Then, we can load its weights
        to this model
    Return:
        model (fasterrcnn_resnet50_fpn): The Faster R-CNN with its new modifications.
"""
def get_model(path=None):
    # define model, specify mean and std for z-score normalization
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 image_mean=[0.4289522339884313, 0.4289522339884313,
                                                                             0.4289522339884313],
                                                                 image_std=[0.25704046834492883, 0.25704046834492883,
                                                                            0.25704046834492883])
    # define anchor sizes and ratios
    sizes = ((64,), (156,), (286,), (512,), (800,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    # create new object of Anchorgenerator
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios)
    # redefine the rpn head
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # adapt number of classes (3 = background, left hip, right hip)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)

    # load weights of specified model
    if path is not None:
        model_path = PureWindowsPath(path)
        model.load_state_dict(torch.load(Path(model_path)))
    return model


"""
    This function applies non maximum suppression. In this way, bounding boxes that depict the same
    objects can be removed. Only the best is kept. 
    Args:
        orig_prediction (dictionary): the prediction of the model
        iou_threshold (float): define how much intersection over union is allowed between two predicted 
        bounding boxes. 
    Return:
        final_prediction (dictionary): new set of predicted bounding boxes and its details after nms
"""
def apply_nms(orig_prediction, iou_thresh=0.5):
    # torchvision returns the indices of the bboxes to keep
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

"""
In this function, we apply the model on the datasets. Afterward, the cropped images are saved.
Args: 
    image_path(Path): folder where all the images are located that needs to be cropped
    storage_path(Path): folder where the cropped images should be stored
    model_path(String): path to .pth file (with extension) 
"""
def crop_images(image_path, storage_path, model_path):
    model = get_model(model_path)
    device = torch.device('cpu')
    model.to(device)

    os.getcwd()
    directory = [file for file in os.listdir(image_path) if isfile(join(image_path, file))]

    # go through each image in directory
    for file in directory:
        best_bb = [-1, -1]
        # get image
        image = get_image(image_path, file)

        if image is None:
            continue
        model.eval()
        # make prediction
        with torch.no_grad():
            prediction = model([image.to(device)])[0]

        # apply nms to remove redundant bb's
        nms_prediction = apply_nms(prediction, iou_thresh=0.1)
        for i in range(0, len(nms_prediction["boxes"])):
            pred_label = nms_prediction["labels"][i]
            score = nms_prediction["scores"][i]

            # store the best bounding box for each class
            if best_bb[pred_label-1] == -1 and score > 0.7:
                best_bb[pred_label-1] = i

            if best_bb[pred_label-1] != -1 and score > 0.7:
                if nms_prediction["scores"][best_bb[pred_label-1]] < score:
                    best_bb[pred_label - 1] = i

        for i in best_bb:
            if i != -1:
                pred_label = nms_prediction["labels"][i]
                orig_image = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
                # compute coordinates of bb
                xmin = int(nms_prediction["boxes"][i][0])
                ymin = int(nms_prediction["boxes"][i][1])
                bb_width = int(nms_prediction["boxes"][i][2]) - xmin
                bb_height = int(nms_prediction["boxes"][i][3]) - ymin
                a = 0
                # make bb square shaped
                if bb_width > bb_height:
                    a = int((bb_width - bb_height) / 2)
                    if ymin - a <= 0:
                        ymin = 0
                    elif ymin + bb_height + 2 * a >= image.shape[1] - 1:
                        bb_height = image.shape[1] - 1 - ymin
                    else:
                        ymin -= a
                        bb_height += 2 * a
                elif bb_width < bb_height:
                    a = int((bb_height - bb_width) / 2)
                    if xmin - a <= 0:
                        xmin = 0
                    elif xmin + bb_width + 2 * a >= image.shape[2] - 1:
                        bb_width = image.shape[2] - 1 - xmin
                    else:
                        xmin -= a
                        bb_width += 2 * a
                # crop the relevant area out of original image
                region_of_interest = orig_image[ymin:ymin + bb_height, xmin:xmin + bb_width]
                file_names = str(file).split("_")
                # define what hip joint it is
                if pred_label.item() == 1:
                    extension = "APR.jpg"
                else:
                    extension = "APL.jpg"
                # store image in folder
                print(storage_path + "\Cropped_" + file_names[0] + "_" + file_names[1] + "_" + extension)
                path = PureWindowsPath(storage_path+"\Cropped_" + file_names[0] + "_" + file_names[1] + "_" + extension)
                cv2.imwrite(Path(path).absolute().as_posix(), region_of_interest)


if __name__ == '__main__':
    # location where all images are
    image_path = r"Images"
    # where should the images be stored
    storage_path = "Cropped"
    # where is your best model stored
    model_path = r"Models\FasterRCNN_Final.pth"
    crop_images(image_path, storage_path, model_path)