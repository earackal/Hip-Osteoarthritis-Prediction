import os
from os.path import isfile, join
from pathlib import PureWindowsPath, Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import RPNHead
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image

""" 
This file was used to compute the gradcam images for the hip joint/feature detector. 
It was adapted from Jaco Gildenblat's implementation. The original code can be
found at https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection%20With%20Faster%20RCNN.ipynb.
Only the documented methods were implemented by me. The lines in the orginial code, that I've changed
are marked with 'ADAPTATION'.
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
        image = cv2.imread(os.path.join(image_path, file), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        image_float_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        image = image_float_np.transpose(2, 0, 1)
        image = torch.from_numpy(image)
        return image, image_float_np
    except Exception:
        return None, None


"""
Gets hip Joint detector.
Args:
    path (String): path to an .pth file of an already trained Faster R-CNN. Then, we can load its weights
    to this model
Return:
    model (fasterrcnn_resnet50_fpn): The Faster R-CNN with its new modifications.
For further details, please refer file Faster R-CNN
"""
def get_model_hip(path=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
                                                                 image_mean=[0.4289522339884313, 0.4289522339884313,
                                                                             0.4289522339884313],
                                                                 image_std=[0.25704046834492883, 0.25704046834492883,
                                                                            0.25704046834492883])
    sizes = ((64,), (156,), (260,), (468,), (800,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=3)

    if path is not None:
        model_path = PureWindowsPath(path)
        model.load_state_dict(torch.load(Path(model_path)))
    return model


"""
Gets Feature detector.
Args:
    path (String): path to an .pth file of an already trained Faster R-CNN. Then, we can load its weights
    to this model
Return:
    model (fasterrcnn_resnet50_fpn): The Faster R-CNN with its new modifications.
For further details, please refer file Faster R-CNN
"""
def get_model_feature(path=None):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 image_mean=[0.5, 0.5, 0.5],
                                                                 image_std=[0.2, 0.2, 0.2])

    sizes = ((8,), (16,), (32,), (64,), (112,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)

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
def apply_nms(orig_prediction, iou_thresh):
    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction

def draw_boxes(boxes, labels, classes, image):
    # ADAPTATION
    class_names = [0, 1, 2]
    # if using feature detector, then use
    # class_names = [0, 1, 2, 3, 4]
    COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return image


def predict(input_tensor, model, device, detection_threshold):
    outputs = model(input_tensor)
    # ADAPTATION
    class_names = ["background", "Right Hip", "Left Hip"]
    # if using feature detector, then use
    # class_names = ["background", "OSTS_acet", "OSTS_fem", "OSTI", "Flattening"]
    pred_classes = [class_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    pred_labels = outputs[0]['labels'].cpu().numpy()
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

    boxes, classes, labels, indices = [], [], [], []
    for index in range(len(pred_scores)):
        if pred_scores[index] >= detection_threshold:
            boxes.append(pred_bboxes[index].astype(np.int32))
            classes.append(pred_classes[index])
            labels.append(pred_labels[index])
            indices.append(index)
    boxes = np.int32(boxes)
    return boxes, classes, labels, indices


def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam, labels, classes):
    # Normalize the CAM to be in the range [0, 1]
    # inside every bounding boxes, and zero outside of the bounding boxes.
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
        images.append(img)

    renormalized_cam = np.max(np.float32(images), axis=0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    image_with_bounding_boxes = draw_boxes(boxes, labels, classes, eigencam_image_renormalized)
    return image_with_bounding_boxes

"""
This function is called to generate the grad cams. 
Args: 
    image_path(Path): Path to folder where the images are located
    mode_path(String): Path to .pth file
    storage_path(String): folder, where the gradcams should be stored. You must include the \\ symbol
    as shown in the main function
"""
def create_grad_cam(image_path, model_path, storage_path):
    model = get_model_hip(model_path)
    # if you are using feature detector, use:
    # model = get_model_feature(model_path)

    device = torch.device('cpu')
    model.to(device)

    os.getcwd()
    directory = [file for file in os.listdir(image_path) if isfile(join(image_path, file))]

    # go through each file in the specified image and compute heat maps
    for file in directory:
        image, image_float_np = get_image(image_path, file)
        if image is None:
            continue
        image = image.to(device)
        image = image.unsqueeze(0)
        model.eval()

        # make the predictions
        boxes, classes, labels, indices = predict(image, model, device, 0.7)
        target_layers = [model.backbone]
        targets = [FasterRCNNBoxScoreTarget(labels=labels, bounding_boxes=boxes)]
        # create object eigencam
        cam = EigenCAM(model,
                       target_layers,
                       use_cuda=torch.cuda.is_available(),
                       reshape_transform=fasterrcnn_reshape_transform)

        grayscale_cam = cam(image, targets=targets)
        # Take the first image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        # apply heatmap to image as well
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
        # rewrite cam image to an image
        im = Image.fromarray(cam_image)

        path = PureWindowsPath(storage_path + str(file))
        orig_image = cv2.imread(os.path.join(image_path, file))
        f = plt.figure()

        # save original image next to image with heatmap as in thesis
        f.add_subplot(1, 2, 1)
        plt.axis('off')
        plt.imshow(orig_image)
        f.add_subplot(1, 2, 2)
        plt.axis('off')
        plt.imshow(im)
        f.savefig(Path(path))


if __name__ == '__main__':
    # folder where images are for which you want to generate the eigencams
    image_path = Path(PureWindowsPath(r"Dataset\Test\Images"))
    # path to folder
    model_path = "Models\FasterRCNN_FINAL_16.pth"
    # folder where eigencams should be stored
    storage_path = "GradCam\\"
    create_grad_cam(image_path, model_path, storage_path)