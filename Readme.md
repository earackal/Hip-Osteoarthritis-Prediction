# Project Code
Hip osteoarthritis is a disease that leads to the degeneration of the hip joint cartilage.
Consequently, affected patients experience pain and difficulty performing everyday activities.
For this reason, it is important to detect osteoarthritis at an early stage so that effective
treatment can be initiated. In this master’s thesis, we wanted to build a deep learning based 
pipeline that predicts the Kellgren-Lawrence grades for hip joints in an X-ray image.
For this purpose, we worked with the CHECK and OI datasets. The pipeline consisted
of a Faster R-CNN to localise the joints and a DenseNet-121 to predict the stage. The
Kellgren-Lawrence grading system is defined by the presence and stage of osteoarthritis
features. Therefore, we trained CNN models to first predict the severity of selected features.

This repository contains the code that we used during the project. This repository
consists of two folders: Experiment and Object_Detection. In the Object_Detection
folder, you will have access to all the files that are necessary to train
the Hip Joint recognition model and the Feature Detection model. The relevant files 
in that folder are Hip_Joint_Detector.py and Feature_Detector.py. If you want to
use these files, your dataset must be in the PASCAL_VOC Format.

## Hip Joint recognition model
Let us start with the Hip Joint recognition model. If you already have a training,
validation, and test dataset, you can skip the first step:

- First, go to Get_Files.py and use it to create your training, validation, and 
test datasets. 
- If you don't know what anchor sizes and parameters for the z-score normalization you want, 
you can get some inspiration from the Statistics.py file. 
- Once you created the datasets, you can access the Hip_Joint_Detector.py file. 
Specify the relevant paths. If you want to change the configuration of the Faster-RCNN,
you can do that in the get_model() function. 
- Use the training() function to train the model. 
- After training, you can evaluate the model with the model_iou or metrics function.
- With GradCam.py you can visualise the decision-making of your model using EigenCams. 

Once you trained the Hip Joint Detector, we need to apply it to all the images. To do so, 
you can access the Cropped_Images.py file. Specify again the relevant paths and crop the 
hip joints from the X-ray images. The cropped images will have as prefix "Cropped_". In my case, 
I stored the cropped hip joint images in a folder called Cropped_Images which was located 
in the same folder, where all the files necessary for the Experiments are stored 
(in this case, the Experiment folder).

## Feature Detection model
Once you cropped all your images, we need to get the isolated regions within the cropped 
hip joint image. For this purpose, we need to train the Feature Detection model. In my
case, I created the training, validation, and test dataset manually since the number of 
annotated images was low.

But, we still will keep the same folder structure as before: We have a Dataset folder.
In the Dataset folder, we have a Training, Validation, and Test folder. In each of these 
folders, we have the Annotations and Images folder.

- If you don't know what anchor sizes and parameters for the z-score normalization you want, 
you can get some inspiration from the Statistics.py file.
- Once you created the datasets, you can access the Feature_Detector.py file. 
Specify the relevant paths. If you want to change the configuration of the Faster-RCNN,
you can do that in the get_model() function. 
- Use the training() function to train the model. 
- After training, you can evaluate the model with the model_iou or metrics function.
- With GradCam.py you can visualise the decision-making of your model using EigenCams. 

After evaluation, apply the create_coordinates_file() function. This function will create
a file called coordinates.csv. You can use another name if you want. This file will contain
all the coordinates of the isolated regions. It has the following structure:

,image_paths,OSTS_fem,OSTS_acet,OSTI,Flattening

0,Cropped_0003088_T00_APL,"[125.67710876464844, 355.64044189453125, 618.5879516601562, 848.7530517578125]","[314.8020935058594, 243.04559326171875, 483.6934814453125, 418.4653625488281]","[616.6515502929688, 720.903564453125, 1013.135986328125, 1121.8330078125]","[256.96246337890625, 333.2124328613281, 919.5562133789062, 1006.1159057617188]"

OSTS_fem stands for SFO, OSTS_acet for SAO, OSTI for IO, and Flattening for the bounding box that
covers the femoral head. In other words, in each row, we have the coordinates of a particular 
hip joint image. 

## Experiments
After you created the coordinates.csv file, you can start with the experiments. For this, you need to
go to the Experiment folder. 

### Relevant Files and Folders
Now, everything gets tricky. Because the entire code depends on whether you get the same files as we did.
Otherwise, I can tell you how can create on your own the files needed for the Experiments. 

If you get the same files as I did:
- Access the Get_Images.py file. With the function read_feature_scores(), you can create the file 
feature_scores.csv for CHECK. This file will contain the feature scores of CHECK. To produce it, you need the 
feature_scores.sav file. 
- Afterward, use the create_proper_dataset function to create your train/val/test dataset.

If you don't have it, the structure of feature_scores.csv looks like the following:

id,BUT,CYS,FLA,SCL,AP_JSN_med,AP_JSN_sup,FP_JSN_pos,FP_JSN_sup,OSTI_acet,OSTI_fem,OSTS_acet,OSTS_fem,KL_def
3088_T02_APL,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,2.0,0.0,0.0

id consists of the identifier of the participant, the year and the side of the joint. 
T00 stands for baseline. T0i stands for years i = 2, 5, 8 and T10 stands for year 10. APL = left side, APR = right side.
In each row, we have the feature and KL scores for the participants at various years. 

Furthermore, the folder structure that we used for CHECK looks the following: We have the Proper_Dataset
folder. It contains three folders called Training, Validation, and Test. Each of the three folders contains
4 additional folders: 0, 1, 2, 3. In folder 0, all hip joints with KL score 0 are assigned, etc. 

Once you did this for CHECK, you can do the same for OI using 
the convert_to_csv() function. It will create the OI_scores.csv file. The id of participants are 
OI00_9760954_APL or 9760954_APL, where OI00 means baseline. To create this file, you need the
feature score files from NIH. Afterward, you can use the assign_oi function to store all images 
from OI in your specified folder. 

### Training of the models
- Once you created all of these folders and files, you can access the JSN.py file to train the models
for superior and medial JSN. In Other_Features.py, you can train the models for IO, SAO, SFO, and KL. 
- If you need to compute the parameters for z-score normalisation, use Statistics.py from above.
- With KL_Formula.py, you can train a model to output the KL score given the feature scores.
- In Ensemble.py, you can use the models from Experiment 1 to predict the KL score

I hope that the descriptions in the codes will help you further. All the best!
