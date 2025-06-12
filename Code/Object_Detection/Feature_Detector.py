import os
import random
from os.path import isfile, join
import numpy as np
import warnings
from pathlib import Path, PureWindowsPath
import cv2
from xml.etree import ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import torchvision
from torchvision import transforms
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch
import utils
warnings.filterwarnings('ignore')
import torch

"""
This is the code that was used to train and to evaluate the Feature Detector (isolated region). 
It was adapted from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html.
To work with this code, it is very important that the annotations are in PASCAL VOC. If 
you want to generate annotations, a very helpful tool is label studios. They provide
the option to generate PASCAL VOC annotations. 

This code is basically the same as the Hip Joint Detector. Therefore, we will comment 
mostly the differences towards Hip_Joint_Detector.py
"""


"""
This is the FeatureDataset class. It can be used to create the training, validation and
test datasets. 
"""
class FeatureDataset(torch.utils.data.Dataset):
    """
    The init function initializes an object of the FeatureDataset
    Args:
        image_dir (Path): path where the images of a dataset can be found.
        annotation_dir (Path): path where the annotations in pascal_voc of a dataset can be found
        augment (Boolean): we can specify whether the images in the dataset should be augmented. Set this
        only true for the training dataset
    """
    def __init__(self, image_dir, annotation_dir, augment=False):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        # new set of classes
        self.classes = ["_", "OSTS_acet", "OSTS_fem", "OSTI", "Flattening"]
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
        name = self.imgs[index]
        path = os.path.join(self.image_dir, name)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotation_file = name.replace(".jpg", ".xml")
        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        boxes = []
        labels = []

        width = image.shape[1]
        height = image.shape[0]

        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.iter('object'):
            # since we have now more classes, we get a larger if clause
            cls_name = obj.find('name').text
            if cls_name == "OSTS_acet":
                cls_name = 1
            elif cls_name == "OSTS_fem":
                cls_name = 2
            elif cls_name == "OSTI":
                cls_name = 3
            else:
                cls_name = 4
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(cls_name)

        k = random.randint(0, 5)

        # In this part, we can apply some augmentation techniques on the image
        # with a certain probability
        if self.augmentation and k >= 1:
            # Add noise
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

            width = image.shape[1]
            height = image.shape[0]

            # Crop irrelevant parts of image out
            k = 1
            if k == 1:
                # don't forget to include the boxes of all features
                smallest_x = int(min(boxes[0][0], boxes[1][0], boxes[2][0], boxes[3][0]))
                smallest_y = int(min(boxes[0][1], boxes[1][1], boxes[2][1], boxes[3][1]))
                largest_x = int(max(boxes[0][2], boxes[1][2], boxes[2][2], boxes[3][2]))
                largest_y = int(max(boxes[0][3], boxes[1][3], boxes[2][3], boxes[3][3]))

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

        # define target
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
    This function returns the number of images in FeatureDataset
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
If you want to change the z-score normalization, the anchor sizes or the aspect ratios, please change
the respective values. 
Args:
    path (String): path to an .pth file of an already trained Faster R-CNN. Then, we can load its weights
    to this model
Return:
    model (fasterrcnn_resnet50_fpn): The Faster R-CNN with its new modifications.
"""
def get_model(path=None):
    # adapt mean and std for z-score normalisation
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                                 image_mean=[0.5, 0.5, 0.5],
                                                                 image_std=[0.2, 0.2, 0.2])

    # adapt your anchor sizes and ratios according to the isolated regions
    sizes = ((8,), (16,), (32,), (64,), (112,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(sizes)
    anchor_generator = AnchorGenerator(
        sizes=sizes,
        aspect_ratios=aspect_ratios)
    model.rpn.anchor_generator = anchor_generator
    model.rpn.head = RPNHead(256, anchor_generator.num_anchors_per_location()[0])
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Very Important: increase the number of classes!
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=5)

    if path is not None:
        model_path = PureWindowsPath(path)
        model.load_state_dict(torch.load(Path(model_path)))
    return model


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
def eval_forward(model, epoch, data_loader, device, print_freq, scaler=None):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    loss = []
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        print("VALIDATION: TOTAL LOSS = ", loss_value, ". DETAILS: ", loss_dict)
        loss.append(float(loss_value))
    print("TOTAL LOSS = ", sum(loss) / len(loss))

"""
This is the method that was used to create the coordinates file that contained
the coordinates of the isolated regions

Args:
    - image_path (PureWindowsPath): path to folder where ALL hip joint images are stored
    - model_path (String): path where model as .h5 file is stored
"""
def create_coordinates_file(image_path, model_path):
    # load model
    model = get_model(model_path)
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    # in these lists, we will store the predicted coordinates
    fem_labels = []
    acet_labels = []
    osti_labels = []
    flat_labels = []
    paths = []
    with torch.no_grad():
        os.getcwd()
        # collect all cropped hip joint images from the specified folder
        directory = [file for file in os.listdir(image_path) if isfile(join(image_path, file))]
        count = 0
        # go through each file
        for file in directory:
            # read image, normalize its scale and orientation
            # if you don't want to normalize it, no problem,
            # faster r-cnn will do it anyway
            image = cv2.imread(os.path.join(image_path, file))
            #cv2.resize(image, (224, 224))
            if str(file).__contains__("APL"):
                image = cv2.flip(image, 1)
            # pre-process image for prediction
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)
            # predict
            prediction = model([image.to(device)])[0]
            #prediction = apply_nms(prediction, iou_thresh=0.2)
            best = [-1, -1, -1, -1]
            score = [-1, -1, -1, -1]
            # get the best prediction for each class
            for k in range(0, len(prediction["boxes"])):
                pred_label = prediction["labels"][k] - 1
                if prediction["scores"][k] > score[pred_label]:
                    best[pred_label] = k
                    score[pred_label] = prediction["scores"][k]
            #prediction = apply_nms(prediction, 0.5)
            #plot_img_and_target_bbox(orig_image, prediction, file)
            # get the best prediction
            if best[0] != -1:
                acet = prediction["boxes"][best[0]].tolist()
            else:
                acet = -1
            if best[1] != -1:
                fem = prediction["boxes"][best[1]].tolist()
            else:
                fem = -1
            if best[2] != -1:
                osti = prediction["boxes"][best[2]].tolist()
            else:
                osti = -1
            if best[3] != -1:
                flattening = prediction["boxes"][best[3]].tolist()
            else:
                flattening = -1
            count += 1
            # append image name to paths list
            paths.append(str(file).replace(".jpg", ""))
            # store the best predictions in their respective lists from above
            fem_labels.append(fem)
            acet_labels.append(acet)
            osti_labels.append(osti)
            flat_labels.append(flattening)
            print(count)
        # create df and save the new coordinates file
        df = pd.concat([pd.Series(paths, name='image_paths'), pd.Series(fem_labels, name='OSTS_fem'),
                        pd.Series(acet_labels, name='OSTS_acet'), pd.Series(osti_labels, name='OSTI'),
                        pd.Series(flat_labels, name='Flattening')], axis=1)
        df.to_csv("coordinates_Proper.csv")


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
    train_data = FeatureDataset(train_image_path, train_annot_path, True)
    val_data = FeatureDataset(val_image_path, val_annot_path, False)

    # Create data loader for both datasets
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=10, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    # get model and place it in the respective memory storage
    model = get_model()
    model.to(device)

    # define parameter list, optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    epochs = 25

    for epoch in range(0, epochs):
        # train one epoch using engine.py
        train_one_epoch(model=model, optimizer=optimizer, data_loader=train_loader, epoch=epoch, print_freq=1,
                        device=device)
        scheduler.step()
        # store state of model after each epoch
        model_path = PureWindowsPath(model_storage_path + "\FasterRCNN_" + str(epoch) + ".pth")
        torch.save(model.state_dict(), Path(model_path))
        # evaluate validation loss
        eval_forward(model, epoch, val_loader, device, 1)
        # after 2 epochs, increase batch size to 8
        if epoch == 1:
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=8, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

"""
This function is used to compute precision, recall and f1 scores for test dataset
Args:
    val_image_path (Path): path where the images of your validation/test dataset can be found
    val_annot_path (Path): path where the annotations of your validation/test dataset can be found
    model_path (String): path where the .pth file of the model can be found 
    class_id (int): specify for which class you want to compute the metrics.
    Use the ids from the FeatureDataset
"""
def metrics(val_image_path, val_annot_path, model_path, class_id):
    model = get_model(model_path)
    model.eval()
    with torch.no_grad():
        val_data = FeatureDataset(val_image_path, val_annot_path, False)

        valid_loader = torch.utils.data.DataLoader(
            val_data, batch_size=4, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        fp_R = 0
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
                match_right = True
                if target_labels.__contains__(class_id):
                    match_right = False
                for i in range(len(predicted_boxes)):
                    if predicted_labels[i] != class_id:
                        continue
                    if predicted_labels[i] in target_labels:
                        # only consider predictions for specified class

                        # If the predicted label matches a target label, check for overlap
                        iou = torchvision.ops.box_iou(predicted_boxes[i].unsqueeze(0),
                                                      target_boxes[target_labels == predicted_labels[i]])
                        #iou = compute_iou(predicted_boxes[i], target_boxes[target_labels == predicted_labels[i]], name)
                        # bb of size less than 2 -> artefact (thesis)
                        if predicted_boxes[i][3] - predicted_boxes[i][1] < 2:
                            print("Artefact: ", val_data.imgs[target['image_id']], prediction)
                            continue
                        # true positive since iou large equal to 0.7
                        if torch.max(iou) >= 0.7:
                            true_positives += 1
                            if predicted_labels[i] == class_id:
                                match_right = True
                        # iou is lower -> false positive
                        else:
                            false_positives += 1
                            if predicted_labels[i] == class_id:
                                fp_R += 1
                    else:
                        # predicted label was not in set of target labels -> false positive
                        false_positives += 1
                        if predicted_labels[i] == class_id:
                            fp_R += 1

                # Compute the false negatives
                if not match_right:
                    false_negatives += 1

        # Compute precision, recall, and sensitivity and print them out
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)

        print(precision, recall, f1, true_positives, false_positives, false_negatives, fp_R)

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
    class_id (int): specify for which class you want to compute the metrics.
    Use the ids from the FeatureDataset
"""
def model_iou(val_image_path, val_annot_path, model_path, class_id):
    val_data = FeatureDataset(val_image_path, val_annot_path, False)
    model = get_model(model_path)

    device = torch.device('cpu')
    model.to(device)

    iou_scores = []
    # go through each image in the dataset
    for i in range(0, val_data.__len__()):
        img, target = val_data[i]

        model.eval()
        # let the model make its prediction
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        #nms_prediction = apply_nms(prediction, iou_thresh=0.2)
        nms_prediction = prediction
        """if len(target["boxes"]) != len(nms_prediction["boxes"]):
            cases += 1
            misclassifications += abs(len(target["boxes"]) - len(nms_prediction["boxes"]))
            print("Misclassifications present in ", val_data.imgs[i])"""
        best_scores = [-1, -1, -1, -1]
        best_conf = [-1, -1, -1, -1]
        #print(nms_prediction)
        # only compute the IoU between the true positives and the ground truth bounding boxes
        for k in range(0, len(nms_prediction["boxes"])):
            pred_label = nms_prediction["labels"][k]
            conf = nms_prediction["scores"][k]
            if pred_label != class_id:
                continue
            try:
                index = (target["labels"] == pred_label).nonzero(as_tuple=True)[0]
                # compute iou
                iou = compute_iou(nms_prediction["boxes"][k], target["boxes"][index[0]])
                # only consider the one prediction for which the model is most confident in
                if best_conf[pred_label - 1] < conf:
                    best_conf[pred_label - 1] = conf
                    best_scores[pred_label - 1] = iou
            except Exception:
                continue
        # if bounding box was found, add it to list so that we can consider its iou during the
        # computation of the average iou of the dataset
        for val in best_scores:
            if val > -1:
                iou_scores.append(val)
    # output stats
    print("Mean IOU: ", np.mean(np.array(iou_scores)))
    print("Std IOU : ", np.std(np.array(iou_scores)))


if __name__ == '__main__':
    # path where train images are
    train_image_path = PureWindowsPath("New_Dataset\Training\Images")
    # path where train annotations are
    train_annot_path = PureWindowsPath("New_Dataset\Training\Annotations")
    # path where validation images are
    val_image_path = PureWindowsPath("New_Dataset\Validation\Images")
    # path where validation annotations are
    val_annot_path = PureWindowsPath("New_Dataset\Validation\Annotations")
    # path where test images are
    test_image_path = PureWindowsPath("New_Dataset\Test\Images")
    # path where test annotations are
    test_annot_path = PureWindowsPath("New_Dataset\Test\Annotations")
    # folder in which the models should be stored during training
    model_storage_path = "Models"
    # location where final model is
    model_path = r"Models\FasterRCNN_Final.pth"
    # which class should be examined
    class_id = 2
    # folder in which all images are for which you want to create the coordinates file
    image_path = PureWindowsPath("Proper_Dataset\\Full")

    #training(train_image_path, train_annot_path, val_image_path, val_annot_path, model_storage_path)
    #create_coordinates_file(image_path, model_path)
    #model_iou(val_image_path, val_annot_path, model_path, class_id)
    #metrics(val_image_path, val_annot_path, model_path, class_id)
