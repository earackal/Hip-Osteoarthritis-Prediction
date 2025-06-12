import random
from os.path import isfile, join
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET
import os


"""
If working with Label-Studio, you need to be careful. Sometimes, it creates bounding boxes
that is somehow normalized to another image size. If that happens, feel free to use this
code to correct the bounding boxes. The trick is, that the size of the "normalized" image is also
specified in the annotation file. Use it to undo normalization by comparing it with the true image size.

Furthermore, we split in this function the training, validation and test set. 
    
Args:
    image_Path (String): path to folder where the pascal voc images are found
    annotation_Path (String): path to folder where the pascal voc annotations are found
    train_Path (String): path to train folder where the corrected annotations and images should be stored
    valid_Path (String): path to validadtion folder where the corrected annotations and images should be stored
    test_Path (String): path to test folder where the corrected annotations and images should be stored
    train_split (Float): train data ratio
    val_split (Float): validationd data ratio
    test_split (Float): test data ratio
"""
def correct_xml(image_Path, annotationPath, train_Path, valid_Path, test_Path, train_split, val_split, test_split):
    os.getcwd()
    # load the annotations
    directory = [file for file in os.listdir(annotationPath) if isfile(join(annotationPath, file))]
    # shuffle them so that we can later define the three datasets
    for i in range(0, 100):
        random.shuffle(directory)
    count = 1
    # go through each file and correct annotations
    for file in directory:
        name = str(file)
        names = name.split("-")
        true_name = names[1]

        # get place where corresponding image is stored
        path = image_Path + file.replace(".xml", '.jpg')
        # open image and compute its true shape
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        width = image.width
        height = image.height

        # go through annotation file and collect the bounding boxes
        tree = ET.parse(annotationPath + file)
        root = tree.getroot()
        annotations = []

        det_width = 0
        det_height = 0
        # check the size of the image in the annotation
        # then store in det_width and det_height the normalised shape of image
        for size in root.iter('size'):
            det_width = float(size.find('width').text)
            det_height = float(size.find('height').text)
            print(file, det_width, det_height, width, height)

        class_names = []
        # collect the bounding boxes and check if whether image
        # was actually normalized. if so, undo normalization
        for obj in root.iter('object'):
            class_names.append(obj.find('name').text)
            xml_box = obj.find('bndbox')
            xmin = float(xml_box.find('xmin').text)
            ymin = float(xml_box.find('ymin').text)
            xmax = float(xml_box.find('xmax').text)
            ymax = float(xml_box.find('ymax').text)

            # recompute bounding boxes
            if width != det_width or height != det_height:
                xmin = int((xmin / det_width) * width)
                xmax = int((xmax / det_width) * width)
                ymin = int((ymin / det_height) * height)
                ymax = int((ymax / det_height) * height)

            # BB needs to be squares
            a = (ymax - ymin) - (xmax - xmin)
            if a > 0:
                a = int(abs(a) / 2)
                if xmin - a > 0 and xmax + a < width:
                    xmin -= a
                    xmax += a
            elif a < 0:
                a = int(abs(a) / 2)
                if ymin - a > 0 and ymax + a < height:
                    ymin -= a
                    ymax += a

            # if you want to, you can do here a last attempt to increase or decrease the size
            # of the BB's
            increase = int((width / 100) * 1.5)
            if ymin - increase > 0 and ymax + increase < height and xmin - increase > 0 and xmax + increase < width:
                xmin -= increase
                xmax += increase
                ymin -= increase
                ymax += increase

            print(abs((xmax - xmin) - (ymax - ymin)), xmax - xmin, ymax - ymin)
            annotations.append([xmin, ymin, xmax, ymax])

        # Once bb's were corrected, store them back in to the annotation file
        # For this, we need to create the annotation tree. Each annotation file
        # has a format of a tree. Refer to PASCAL VOC format.
        root = ET.Element('annotation')

        size = ET.SubElement(root, 'size')
        tree_width = ET.SubElement(size, 'width')
        tree_height = ET.SubElement(size, 'height')
        depth = ET.SubElement(size, 'depth')
        tree_width.text = str(width)
        tree_height.text = str(height)
        depth.text = str(3)


        # if only one joint in image
        if len(class_names) == 1:
            [xmin, ymin, xmax, ymax] = annotations[0]
            object = ET.SubElement(root, 'object')
            name = ET.SubElement(object, 'name')
            pose = ET.SubElement(object, 'pose')
            truncated = ET.SubElement(object, 'truncated')
            difficult = ET.SubElement(object, 'difficult')
            name.text = class_names[0]
            pose.text = "Unspecified"
            truncated.text = "0"
            difficult.text = "0"

            bndbox = ET.SubElement(object, 'bndbox')
            xmin1 = ET.SubElement(bndbox, 'xmin')
            ymin1 = ET.SubElement(bndbox, 'ymin')
            xmax1 = ET.SubElement(bndbox, 'xmax')
            ymax1 = ET.SubElement(bndbox, 'ymax')

            xmin1.text = str(xmin)
            ymin1.text = str(ymin)
            xmax1.text = str(xmax)
            ymax1.text = str(ymax)

            tree = ET.ElementTree(root)

            # create train, validation, test datasets. assign corrected image and annotation to respective
            # directory. For train, always flip image and annotation additionally to increase dataset
            # Since we shuffled the directory earlier, we just assigned the first k images to training, the
            # next m images to validation and the last n images to test set.
            # training dataset
            if 1 < count <= int(len(directory) * train_split):
                # save original image and corrected annotation to training dataset
                image.save(train_Path + "Images\\" + file.replace(".xml", ".jpg"))
                annotation_place = train_Path + "Annotations\\" + file
                tree.write(annotation_place)

                # save flipped image and annotation as well. useful to increase training dataset
                flipped_image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
                xmin1.text = str(width - xmax)
                xmax1.text = str(width - xmin)

                if class_names[0] == "Left Hip":
                    name.text = "Right Hip"
                else:
                    name.text = "Left Hip"

                flipped_filename = train_Path + 'Annotations\\aug_' + file
                tree.write(flipped_filename)
                flipped_image.save(train_Path + "Images\\aug_" + file.replace(".xml", ".jpg"))
            elif int(len(directory) * train_split) < count <= int(len(directory) * train_split) + int(len(directory) * val_split):
                image.save(valid_Path + "Images\\" + file.replace(".xml", ".jpg"))
                annotation_place = valid_Path + "Annotations\\" + file
                tree.write(annotation_place)
            else:
                image.save(test_Path + "Test\\Images\\" + file.replace(".xml", ".jpg"))
                annotation_place = test_Path + "Annotations\\" + file
                tree.write(annotation_place)
        else:
            if class_names[0] == "Right Hip":
                [xmin_right, ymin_right, xmax_right, ymax_right] = annotations[0]
                [xmin_left, ymin_left, xmax_left, ymax_left] = annotations[1]
            else:
                [xmin_left, ymin_left, xmax_left, ymax_left] = annotations[0]
                [xmin_right, ymin_right, xmax_right, ymax_right] = annotations[1]

            object1 = ET.SubElement(root, 'object')
            name1 = ET.SubElement(object1, 'name')
            pose1 = ET.SubElement(object1, 'pose')
            truncated1 = ET.SubElement(object1, 'truncated')
            difficult1 = ET.SubElement(object1, 'difficult')
            name1.text = "Right Hip"
            pose1.text = "Unspecified"
            truncated1.text = "0"
            difficult1.text = "0"

            bndbox1 = ET.SubElement(object1, 'bndbox')
            xmin1 = ET.SubElement(bndbox1, 'xmin')
            ymin1 = ET.SubElement(bndbox1, 'ymin')
            xmax1 = ET.SubElement(bndbox1, 'xmax')
            ymax1 = ET.SubElement(bndbox1, 'ymax')

            xmin1.text = str(xmin_right)
            ymin1.text = str(ymin_right)
            xmax1.text = str(xmax_right)
            ymax1.text = str(ymax_right)

            object2 = ET.SubElement(root, 'object')
            name2 = ET.SubElement(object2, 'name')
            pose2 = ET.SubElement(object2, 'pose')
            truncated2 = ET.SubElement(object2, 'truncated')
            difficult2 = ET.SubElement(object2, 'difficult')
            name2.text = "Left Hip"
            pose2.text = "Unspecified"
            truncated2.text = "0"
            difficult2.text = "0"

            bndbox2 = ET.SubElement(object2, 'bndbox')
            xmin2 = ET.SubElement(bndbox2, 'xmin')
            ymin2 = ET.SubElement(bndbox2, 'ymin')
            xmax2 = ET.SubElement(bndbox2, 'xmax')
            ymax2 = ET.SubElement(bndbox2, 'ymax')

            xmin2.text = str(xmin_left)
            ymin2.text = str(ymin_left)
            xmax2.text = str(xmax_left)
            ymax2.text = str(ymax_left)

            tree = ET.ElementTree(root)

            if count <= int(len(directory) * train_split):
                image.save(train_Path + "Images\\" + file.replace(".xml", ".jpg"))
                annotation_place = train_Path + "Annotations\\" + file
                tree.write(annotation_place)

                flipped_image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
                xmin1.text = str(width - xmax_left)
                xmax1.text = str(width - xmin_left)
                ymin1.text = str(ymin_left)
                ymax1.text = str(ymax_left)

                xmin2.text = str(width - xmax_right)
                xmax2.text = str(width - xmin_right)
                ymin2.text = str(ymin_right)
                ymax2.text = str(ymax_right)

                flipped_filename = train_Path + 'Annotations\\aug_' + file
                tree.write(flipped_filename)
                flipped_image.save(train_Path + "Images\\aug_" + file.replace(".xml", ".jpg"))
            elif int(len(directory) * train_split) < count <= int(len(directory) * train_split) + int(len(directory) * val_split):
                image.save(valid_Path + "Images\\" + file.replace(".xml", ".jpg"))
                annotation_place = valid_Path + "Annotations\\" + file
                tree.write(annotation_place)
            else:
                image.save(test_Path + "Test\\Images\\" + file.replace(".xml", ".jpg"))
                annotation_place = test_Path + "Annotations\\" + file
                tree.write(annotation_place)
        count += 1


if __name__ == '__main__':
    # in the above method, it is explained how you should change these variables
    image_Path = "VOC\\images\\"
    annotationPath = "VOC\\Annotations\\"
    train_Path = "Dataset\\Training\\"
    valid_Path = "Dataset\\Validation\\"
    test_Path = "Dataset\\Test\\"
    train_split = 0.8
    valid_split = 0.1
    test_split = 0.1
    correct_xml(image_Path, annotationPath, train_Path, valid_Path, test_Path, train_split, valid_split, test_split)








