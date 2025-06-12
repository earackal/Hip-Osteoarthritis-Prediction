import os
import random
import numpy as np
import warnings
from pathlib import Path, PureWindowsPath
import cv2
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import torch
from sklearn.metrics import auc
warnings.filterwarnings('ignore')

"""
This is the code that was used to train and to evaluate the Hip Joint Detector. 
It was adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
To work with this code, it is very important that the annotations are in PASCAL VOC. If 
you want to generate annotations, a very helpful tool is label studios. They provide
the option to generate PASCAL VOC annotations. 
"""

"""
This is the HipDataset class. It can be used to create the training, validation and
test datasets. 
"""
class HipDataset(torch.utils.data.Dataset):

    """
    The init function initializes an object of the HipDataset
    Args:
        image_dir (Path): path where the images of a dataset can be found.
        annotation_dir (Path): path where the annotations in pascal_voc of a dataset can be found
        augment (Boolean): we can specify whether the images in the dataset should be augmented. Set this
        only true for the training dataset
    """
    def __init__(self, image_dir, annotation_dir, augment=False):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        # the three classes
        self.classes = ["_", "Right Hip", "Left Hip"]
        # collects the names of all the images in the specified image directory
        self.imgs = [image for image in sorted(os.listdir(self.image_dir)) if image[-4:] == '.jpg']
        self.augmentation = augment

    """
    The function returns the specified image and its annotation
    Args:
        index (int): the index of the image in the dataset. 
    Return:
        image (torch): torch array from specified image
        target (dictionary): annotation of image, contains information on the bounding boxes, labels,
        area, iscrowd and index. To understand what each element means, please visit 
        https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
    """
    def __getitem__(self, index):
        # Find the name of the image
        name = self.imgs[index]
        # Get the location where the image is stored
        path = os.path.join(self.image_dir, name)
        # read and transform image
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # access annotation file
        annotation_file = name.replace(".jpg", ".xml")
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        boxes = []
        labels = []

        width = image.shape[1]
        height = image.shape[0]

        # go through the annotation file and collect the bounding boxes
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            if cls_name == "Right Hip":
                cls_name = 1
            else:
                cls_name = 2
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_name)

        bb_amount = len(boxes)


        # In this part, we can apply some augmentation techniques on the image
        # with a certain probability
        k = random.randint(0, 5)
        if self.augmentation and k >= 1:
            # Add noise to the image
            k = random.randint(0, 2)
            if k == 0:
                noise = np.zeros(image.shape, np.uint8)
                cv2.randn(noise, 0, 7)
                img_noised = cv2.add(image, noise)
                img_noised = np.clip(img_noised, 0, 255).astype(np.uint8)
                image = cv2.cvtColor(img_noised, cv2.COLOR_BGR2RGB)

            to_PIL = transforms.Compose([
                transforms.ToPILImage()
            ])
            img = to_PIL(image)

            # Change brightness and contrast
            k = random.randint(0, 3)
            if k >= 1:
                Color_Transformation = transforms.Compose([
                    transforms.ColorJitter(brightness=0.1, contrast=0.5)
                ])
                img = Color_Transformation(img)

            image = np.array(img)
            image = image[:, :, ::-1].copy()

            # Apply rotation, scaling, translation
            k = random.randint(0, 3)
            if k >= 1:
                if bb_amount == 1:
                    bbs = BoundingBoxesOnImage([
                        BoundingBox(x1=boxes[0][0], y1=boxes[0][1], x2=boxes[0][2], y2=boxes[0][3])
                    ], shape=image.shape)
                else:
                    bbs = BoundingBoxesOnImage([
                        BoundingBox(x1=boxes[0][0], y1=boxes[0][1], x2=boxes[0][2], y2=boxes[0][3]),
                        BoundingBox(x1=boxes[1][0], y1=boxes[1][1], x2=boxes[1][2], y2=boxes[1][3])
                    ], shape=image.shape)

                up_scaling = random.randint(0, 1)
                if up_scaling == 0:
                    # zoom out = scale down image
                    resized_height = random.randint(int(0.5 * height), 2 * height)
                    seq = iaa.Sequential([
                        iaa.Resize({"height": resized_height, "width": "keep-aspect-ratio"})

                    ])
                else:
                    seq = iaa.Sequential([
                        iaa.Affine(scale=(1, 2))
                    ])
                image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
                check = True
                for i in range(len(bbs.bounding_boxes)):
                    new_bbs = bbs_aug.bounding_boxes[i]
                    if new_bbs.x1 < 0 or new_bbs.x2 > width-1 or new_bbs.y1 < 0 or new_bbs.y2 > height-1:
                        check = False
                if check:
                    for i in range(len(bbs.bounding_boxes)):
                        new_bbs = bbs_aug.bounding_boxes[i]
                        [xmin, ymin, xmax, ymax] = [new_bbs.x1, new_bbs.y1, new_bbs.x2, new_bbs.y2]
                        boxes[i][0] = xmin
                        boxes[i][1] = ymin
                        boxes[i][2] = xmax
                        boxes[i][3] = ymax
                    image = image_aug

            width = image.shape[1]
            height = image.shape[0]

            # Make single hip joint image
            k = random.randint(0, 1)
            if k > 0 and bb_amount == 2:
                bb_amount = 1
                if labels[0] == 1:
                    x_right_max = boxes[0][2]
                    x_left_min = boxes[1][0]
                else:
                    x_right_max = boxes[1][2]
                    x_left_min = boxes[0][0]

                midpoint = int(x_right_max + (x_left_min - x_right_max) / 2)
                s1 = image[:, :midpoint]
                s2 = image[:, midpoint:]
                k = random.randint(0, 1)
                if k == 0:
                    image = s1
                    if labels[0] == 1:
                        boxes = [[boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]]]
                    else:
                        boxes = [[boxes[1][0], boxes[1][1], boxes[1][2], boxes[1][3]]]
                    labels = [1]
                else:
                    image = s2
                    if labels[0] == 2:
                        boxes = [[boxes[0][0] - midpoint, boxes[0][1], boxes[0][2] - midpoint, boxes[0][3]]]
                    else:
                        boxes = [[boxes[1][0] - midpoint, boxes[1][1], boxes[1][2] - midpoint, boxes[1][3]]]
                    labels = [2]

            # Crop irrelevant parts of image out
            k = 1
            if k == 1:
                if bb_amount == 1:
                    [smallest_x, smallest_y, largest_x, largest_y] = [int(x) for x in boxes[0]]
                else:
                    smallest_x = int(min(boxes[0][0], boxes[1][0]))
                    smallest_y = int(min(boxes[0][1], boxes[1][1]))
                    largest_x = int(max(boxes[0][2], boxes[1][2]))
                    largest_y = int(max(boxes[0][3], boxes[1][3]))
                #left = random.randint(max(int((smallest_x - 20) * 0.4), 0), max(smallest_x - 20, 0))
                left = random.randint(0, max(smallest_x - 20, 0))
                down = random.randint(0, max(smallest_y - 20, 0))
                right = random.randint(0,  max(width - 1 - (largest_x + 20), 0))
                up = random.randint(0, max(height - 1 - (largest_y + 20), 0))
                image = image[down:height - 1 - up, left:width - 1 - right]

                for i in range(0, len(boxes)):
                    boxes[i][0] -= left
                    boxes[i][1] -= down
                    boxes[i][2] -= left
                    boxes[i][3] -= down

        # normalize image
        image = image.astype(np.float32)
        image = image / 255.0
        image = image.transpose(2, 0, 1)

        # define the target
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = index

        return torch.from_numpy(image), target

    """
    This function returns the number of images in HipDataset
    Return:
        - len (int): number of images in the dataset
    """
    def __len__(self):
        return len(self.imgs)

"""
This function can be used to show an image with its ground truth bounding boxes.
Args:
    image (PIL): PIL image
    target (dictionary): annotation of image
"""
def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in (target['boxes']):
        x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')
        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()


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
This is a function to convert a torchtensor back to PIL image
Args: 
    img (Torch): torch image
Returns:
    img (PIL): PIL image
"""
def torch_to_pil(img):
    return transforms.ToPILImage()(img).convert('RGB')


"""
    This is a very important function. It redefines the RPN head of the pre-trained Faster R-CNN. 
    If you want to change the z-score normalization parameters, the anchor sizes or the aspect ratios, please change
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
This function can be used to compute the model's coco mAP. It uses the function in 
engine.py. The mAP values will be printed on the console.
Args:
    val_image_path (Path): path where the images of your validation/test dataset can be found
    val_annot_path (Path): path where the annotations of your validation/test dataset can be found
    model_path (String): path where the .pth file of the model can be found
"""
def model_evaluation(val_image_path, val_annot_path, model_path):
    val_data = HipDataset(val_image_path, val_annot_path, False)

    val_loader = torch.utils.data.DataLoader(
        val_data, shuffle=False, num_workers=1, collate_fn=utils.collate_fn)

    model = get_model(model_path)

    device = torch.device('cpu')
    model.to(device)
    # evaluate model
    evaluate(model=model, data_loader=val_loader, device=device)



"""
This function is used to compute the iou of two bounding boxes
Args: 
    box 1 (List): must contain 4 values (xmin, ymin, xmax, ymax).
    box 2 (List): second bounding box
Return:
    iou (float): the intersection over union between the given boxes
"""
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = inter_area / float(box1_area + box2_area - inter_area)
    return iou


"""
This function can be used to compute the average IoU of your model in a specified
validation / test dataset. 
Args: 
    val_image_path (Path): path where the images of your validation/test dataset can be found
    val_annot_path (Path): path where the annotations of your validation/test dataset can be found
    model_path (String): path where the .pth file of the model can be found    
"""
def model_iou(val_image_path, val_annot_path, model_path):
    val_data = HipDataset(val_image_path, val_annot_path, False)
    model = get_model(model_path)

    device = torch.device('cpu')
    model.to(device)

    cases = 0
    misclassifications = 0
    iou_scores = []
    # go through each image in the dataset
    for i in range(0, val_data.__len__()):
        img, target = val_data[i]

        model.eval()
        # let the model make its prediction
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        # apply nms
        nms_prediction = apply_nms(prediction, iou_thresh=0.2)
        # if nms is applied, the left and right hip joint must be occupied in the best case by one box.
        # If that is not the case, the model recognized additional joints -> misclassifications
        # count the number of misclassifications
        if len(target["boxes"]) != len(nms_prediction["boxes"]):
            cases += 1
            misclassifications += abs(len(target["boxes"]) - len(nms_prediction["boxes"]))
            print("Misclassifications present in ", val_data.imgs[i])
        # only compute the IoU between the true positives and the ground truth bounding boxes
        best_scores = [-1, -1]
        for k in range(0, len(nms_prediction["boxes"])):
            # get label of prediction
            pred_label = nms_prediction["labels"][k]
            try:
                # compare predicted bb with the ground truth of respective class
                index = (target["labels"] == pred_label).nonzero(as_tuple=True)[0]
                iou = compute_iou(nms_prediction["boxes"][k], target["boxes"][index[0]])
                # find the best one
                if best_scores[pred_label - 1] < iou:
                    best_scores[pred_label - 1] = iou
            except Exception:
                continue
        # if bounding box was found, add it to list so that we can consider its iou during the
        # computation of the average iou of the dataset
        for val in best_scores:
            if val > -1:
                iou_scores.append(val)
    # Output the stats
    print("Total Number of Images with Misclassifications: ", cases)
    print("Total Number of Misclassifications: ", misclassifications)
    print("Mean IOU: ", np.mean(np.array(iou_scores)))
    print("Std IOU : ", np.std(np.array(iou_scores)))


"""
This function is used during training. It outputs the validation loss.
For more information, see the training method. 
Args:
    model (fasterrcnn_resnet50_fpn): current model 
    epoch (int): current epoch
    data_loader (data_loader): data_loader of the validation dataset 
    device (torch.device): where the model is stored
    scaler (boolean): necessary for autocast        
"""
def eval_forward(model, epoch, data_loader, device, scaler=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    loss = []
    # access batch of images from data_loader
    for images, targets in data_loader:
        # rewrite images and targets to compute loss
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # computation of loss. adapted from engine.py
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        # print the stats
        print("VALIDATION: TOTAL LOSS = ", loss_value, ". DETAILS: ", loss_dict)
        loss.append(float(loss_value))
    print("TOTAL LOSS = ", sum(loss) / len(loss))


"""
This function can be used to output image with its target annotation (red)
and predicted bounding boxes (green) and save them in the "Prediction" folder.
It is called from the show_predicted_bb function.
Args:
    imgs (PIL): PIL image
    predictions (dictionary): contains the predicted annotations
    target (dictionary): contains ground truth annotations
    file_name (String): name of image
"""
def plot_img_and_target_bbox(img, predictions, target, file_name):
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    # go through each bb in prediction, compute its width and height so that we
    # can draw the bb on the figure
    for i in range(len(predictions["labels"])):
        x = predictions["boxes"][i][0].item()
        y = predictions["boxes"][i][1].item()
        width = predictions["boxes"][i][2].item() - predictions["boxes"][i][0].item()
        height = predictions["boxes"][i][3].item() - predictions["boxes"][i][1].item()
        color = 'green'
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor=color,
                                 facecolor='none')
        # Draw the bounding box on top of the image
        a.add_patch(rect)

    # go through each bb in target, compute its width and height so that we
    # can draw the bb on the figure
    for i in range(len(target["labels"])):
        x = target["boxes"][i][0].item()
        y = target["boxes"][i][1].item()
        width = target["boxes"][i][2].item() - target["boxes"][i][0].item()
        height = target["boxes"][i][3].item() - target["boxes"][i][1].item()
        color = 'red'
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor=color,
                                 facecolor='none')
        # Draw the bounding box on top of the image
        a.add_patch(rect)
    # Save it in the Prediction folder
    path = PureWindowsPath("Prediction\Pred_" + file_name)
    plt.savefig(Path(path))
    plt.close()


"""
    This function can be used to output image with its target annotation (red)
    and predicted bounding boxes (green) and save them in the "Prediction" folder.
    It calls the plot_img_and_target_bbox function.
    Args:
        val_image_path (Path): path where the images of your validation/test dataset can be found
        val_annot_path (Path): path where the annotations of your validation/test dataset can be found
        model_path (String): path where the .pth file of the model can be found 
"""
def show_predicted_bb(val_image_path, val_annot_path, model_path):
    model = get_model(model_path)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    with torch.no_grad():
        # create dataset
        val_data = HipDataset(val_image_path, val_annot_path, False)

        # call function to create plot for each image in the dataset
        for image, target in val_data:
            prediction = model([image.to(device)])[0]
            plot_img_and_target_bbox(torch_to_pil(image), prediction, target, val_data.imgs[target['image_id']])

"""
This function is used to compute precision, recall and f1 scores for test dataset
Args:
    val_image_path (Path): path where the images of your validation/test dataset can be found
    val_annot_path (Path): path where the annotations of your validation/test dataset can be found
    model_path (String): path where the .pth file of the model can be found 
"""
def metrics(val_image_path, val_annot_path, model_path):
    model = get_model(model_path)
    model.eval()
    with torch.no_grad():
        val_data = HipDataset(val_image_path, val_annot_path, False)

        valid_loader = torch.utils.data.DataLoader(
            val_data, batch_size=4, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        fp_R = 0
        fp_L = 0

        # create batches
        for images, targets in valid_loader:
            # Forward pass through the model to get predictions
            predictions = model(images)

            # Iterate over each prediction and target
            for prediction, target in zip(predictions, targets):
                # Get the predicted bounding boxes and labels
                predicted_boxes = prediction['boxes']
                predicted_labels = prediction['labels']

                # Get the target bounding boxes and labels
                target_boxes = target['boxes']
                target_labels = target['labels']

                # Compute the true positives, false positives, and false negatives
                match_right = True
                match_left = True
                if target_labels.__contains__(1):
                    match_right = False
                if target_labels.__contains__(2):
                    match_left = False



                for i in range(len(predicted_boxes)):
                    # check if predicted label is in the target label set. if not,
                    # we have a false positive
                    if predicted_labels[i] in target_labels:
                        # If the predicted label matches a target label, check for overlap
                        iou = torchvision.ops.box_iou(predicted_boxes[i].unsqueeze(0),
                                                      target_boxes[target_labels == predicted_labels[i]])
                        # check if artefact
                        if predicted_boxes[i][3] - predicted_boxes[i][1] < 2:
                            print("Artefact: ", val_data.imgs[target['image_id']], prediction)
                            continue
                        # if not artefact, check with iou whether true positive of false positive
                        if torch.max(iou) >= 0.7:
                            true_positives += 1
                            if predicted_labels[i] == 1:
                                match_right = True
                            if predicted_labels[i] == 2:
                                match_left = True
                            print("OK: ", torch.max(iou))
                        else:
                            false_positives += 1
                            if predicted_labels[i] == 1:
                                fp_R += 1
                            if predicted_labels[i] == 2:
                                fp_L += 1
                            # DEBUGGER
                            print(val_data.imgs[target['image_id']], torch.max(iou))
                            print(prediction)
                            print(target)
                    else:
                        false_positives += 1
                        if predicted_labels[i] == 1:
                            fp_R += 1
                        if predicted_labels[i] == 2:
                            fp_L += 1
                        print("ERROR: ", val_data.imgs[target['image_id']], torch.max(iou))

                # Compute the false negatives (ground truth hip joint was not found)
                if not match_right:
                    false_negatives += 1
                if not match_left:
                    false_negatives += 1

        # Compute precision, recall, and sensitivity
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)

        print(precision, recall, f1, true_positives, false_positives, false_negatives, fp_R, fp_L)


"""
    This function is called to train a Faster R-CNN.
    Args: 
        train_image_path (Path): path where the images of your train dataset can be found
        train_annot_path (Path): path where the annotations of your train dataset can be found
        val_image_path (Path): path where the images of your validation dataset can be found
        val_annot_path (Path): path where the annotations of your validation dataset can be found
        model_path (String): path of folder where the .pth files of the model should be stored
"""
def training(train_image_path, train_annot_path, val_image_path, val_annot_path, model_storage_path):
    # create training and validation data set with HipDataset. Set augment to false for validation.
    train_data = HipDataset(train_image_path, train_annot_path, True)
    val_data = HipDataset(val_image_path, val_annot_path, False)

    # Create data loader for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=16, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=25, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # get model and place it in the respective memory storage
    model = get_model()
    model.to(device)

    # define parameter list, optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    epochs = 20

    for epoch in range(0, epochs):
        # train one epoch using engine.py
        train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, epoch=epoch, print_freq=5,
                        device=device)
        # apply scheduler
        scheduler.step()
        # store state of model after each epoch
        model_path = PureWindowsPath(model_storage_path + "\FasterRCNN_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), Path(model_path))

        # evaluate validation loss
        eval_forward(model, epoch, val_loader, device)

        # after 2 epochs, increase batch size to 32 (for more information, read thesis, section on training details)
        if epoch == 1:
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=32, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)


if __name__ == '__main__':
    train_image_path = PureWindowsPath("New_Dataset\Training\Images")
    train_annot_path = PureWindowsPath("New_Dataset\Training\Annotations")
    val_image_path = PureWindowsPath("New_Dataset\Validation\Images")
    val_annot_path = PureWindowsPath("New_Dataset\Validation\Annotations")
    test_image_path = PureWindowsPath("New_Dataset\Test\Images")
    test_annot_path = PureWindowsPath("New_Dataset\Test\Annotations")
    model_storage_path = "Models"
    model_path = r"Models\FasterRCNN_Final.pth"
    #training(train_image_path, train_annot_path, val_image_path, val_annot_path, model_storage_path)
    #model_evaluation(test_image_path, test_annot_path, model_path)
    #model_iou(test_image_path, test_annot_path, model_path)
    #metrics(test_image_path, test_annot_path, model_path)
    #show_predicted_bb(test_image_path, test_annot_path, model_path)