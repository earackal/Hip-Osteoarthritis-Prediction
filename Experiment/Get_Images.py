import os
from os.path import isfile, join
from pathlib import PureWindowsPath, Path
import cv2
import numpy as np
import pandas as pd

"""
This file is was used to create the train, validation and test dataset for CHECK. 
Furthermore, it was used to create the various files including those containing the feature grades
of CHECK and OI.
"""

"""
This function was used to create the train/val/test datasets for CHECK

Args:
    - image_path (PureWindowsPath): path to folder where all cropped hip joint images are located
    - new_path (String): path to folder where you want to store the datasets
"""
def create_proper_dataset(image_path, new_path):
    # access all files
    directory = [file for file in os.listdir(image_path) if isfile(join(image_path, file))]

    # access to file with the labels
    score_path = PureWindowsPath(r"feature_scores.csv")
    data = pd.read_csv(Path(score_path))
    row_lookup = {}
    for i in data.index:
        row_lookup[data['id'][i]] = str(i)
    data.fillna(-1, inplace=True)

    # go through each cropped hip joint image, get its kl label
    # and store them in respective dataset
    for id in directory:
        key = str(id).replace("Cropped_", "")
        key = key.replace(".jpg", "")
        val = key.split("_")
        key = str(int(val[0])) + "_" + val[1] + "_" + val[2]
        try:
            row_id = int(row_lookup[key])
        except Exception:
            continue

        # assign to dataset
        if row_id < 8200:
            loc = "Training\\"
        elif 8200 <= row_id <= 9000:
            loc = "Validation\\"
        else:
            loc = "Test\\"

        label = int(float(data.iloc[row_id]['KL_def']))
        if label == 9 or label == 5 or label == 0.1 or label == 6.0 or label == 3.5 or label == -1:
            continue
        # store them in dataset
        new_path = new_path + loc + str(label) + "\\" + id
        image = cv2.imread(os.path.join(image_path, id))
        image = cv2.resize(image, (224, 224))
        cv2.imwrite(new_path, image)

"""
This function was used to transform provided labels from CHECK to new file. 
This is very hard coded and can only be used if working with the very same files
"""
def read_feature_scores():
    # access original sav file
    df = pd.read_spss("feature_scores.sav")
    rows = len(df.axes[0])

    main_cols = []
    cleaned_cols = ["id"]
    # go through columns and collect names of relevant columns
    for col in df.columns:
        if col.__contains__("T0") and col.__contains__("_H_"):
            main_cols.append(col)
            clean = col.replace("T0_H_", "")
            clean = clean.replace("li_", "")
            clean = clean.replace("re_", "")
            if clean not in cleaned_cols:
                cleaned_cols.append(clean)


    print(main_cols)
    cleaned_df = pd.DataFrame(columns=cleaned_cols)

    # go through each row and collect all the relevant feature grades.
    for i in range(0, rows):
        id = str(int(df.iloc[i]['nsin']))
        d00_left = {}
        d00_right = {}
        d02_left = {}
        d02_right = {}
        d05_left = {}
        d05_right = {}
        d08_left = {}
        d08_right = {}
        d10_left = {}
        d10_right = {}

        # go through each column name
        for j in main_cols:
            # don't forget the 5 timepoints
            column_name = j.replace("T0_H_", "")
            d00_left["id"] = id + "_T00_APL"
            d00_right["id"] = id + "_T00_APR"
            d02_left["id"] = id + "_T02_APL"
            d02_right["id"] = id + "_T02_APR"
            d05_left["id"] = id + "_T05_APL"
            d05_right["id"] = id + "_T05_APR"
            d08_left["id"] = id + "_T08_APL"
            d08_right["id"] = id + "_T08_APR"
            d10_left["id"] = id + "_T10_APL"
            d10_right["id"] = id + "_T10_APR"

            # access data
            if column_name.__contains__("li_"):
                column_name = column_name.replace("li_", "")
                d00_left[column_name] = df.iloc[i][j]
                d02_left[column_name] = df.iloc[i][j.replace("T0", "T2")]
                d05_left[column_name] = df.iloc[i][j.replace("T0", "T5")]
                d08_left[column_name] = df.iloc[i][j.replace("T0", "T8")]
                try:
                    d10_left[column_name] = df.iloc[i][j.replace("T0", "T10")]
                except Exception:
                    d10_left[column_name] = ""
            else:
                column_name = column_name.replace("re_", "")
                d00_right[column_name] = df.iloc[i][j]
                d02_right[column_name] = df.iloc[i][j.replace("T0", "T2")]
                d05_right[column_name] = df.iloc[i][j.replace("T0", "T5")]
                d08_right[column_name] = df.iloc[i][j.replace("T0", "T8")]
                try:
                    d10_right[column_name] = df.iloc[i][j.replace("T0", "T10")]
                except Exception:
                    # 9
                    d10_right[column_name] = ""
        # store data in fresh pandas df
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d00_left])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d00_right])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d02_left])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d02_right])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d05_left])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d05_right])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d08_left])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d08_right])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d10_left])], ignore_index=True)
        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d10_right])], ignore_index=True)

    # store pandas df
    cleaned_df.to_csv("feature_scores_original.csv", sep=',', index=False, encoding='utf-8')


"""
This function was used to transform provided labels from OI to new file. 
This is very hard coded and can only be used if working with the very same files
"""
def convert_to_csv():
    # read original file from oi and replace errors with 9
    x = open("Hip XR SQ ASCIIhxr_sq00.txt")
    s = x.read().replace("|", ",")
    s = s.replace(".J", "9")
    s = s.replace(".T", "9")
    s = s.replace(".U", "9")
    s = s.replace(".P", "5")
    x.close()

    # store error free file
    x = open("OI_00_scores.csv", "w")
    x.write(s)
    x.close()

    # read error free file, create new pd dataset
    df = pd.read_csv("OI_00_scores.csv")
    cleaned_df = pd.DataFrame(columns=["id", "FLA", "SCL", "CYS", "JSN_SM", "JSN_SL", "OSTI_acet", "OSTI_fem", "OSTS_acet", "OSTS_fem"])

    rows = len(df.axes[0])

    # go through each row in error free file. access the feature grades and store them in panda df.
    for i in range(0, rows):
        id = str(int(df.iloc[i]['ID']))
        if str(df.iloc[i]['hip']) == "L":
            side = "APL"
        else:
            side = "APR"
        d = {}
        d["id"] = "OI00" + "_" + id + "_" + side
        d["FLA"] = df.iloc[i]['V00HFLAT'].split(":")[0]
        d["SCL"] = df.iloc[i]['V00HSCF'].split(":")[0]
        d["CYS"] = df.iloc[i]['V00HCYA'].split(":")[0]
        d["JSN_SM"] = df.iloc[i]['V00HJSNSM'].split(":")[0]
        d["JSN_SL"] = df.iloc[i]['V00HJSNSL'].split(":")[0]
        d["OSTI_acet"] = df.iloc[i]['V00HAOSI'].split(":")[0]
        d["OSTI_fem"] = df.iloc[i]['V00HFOSI'].split(":")[0]
        d["OSTS_acet"] = df.iloc[i]['V00HAOSS'].split(":")[0]
        d["OSTS_fem"] = df.iloc[i]['V00HFOSS'].split(":")[0]

        cleaned_df = pd.concat([cleaned_df, pd.DataFrame([d])], ignore_index=True)
    # store folder.
    cleaned_df.to_csv("OI00_scores.csv", sep=',', index=False, encoding='utf-8')


"""
This function was used to retrieve all images from OI in Cropped_Images and 
store them in a new folder called New_Dataset\\OI

Args: 
    final_path (PureWindowsPath): path to folder New_Dataset\\OI
    data_path (PureWindowsPath): path to folder Cropped_Images
"""
def assign_oi(final_path, data_path):
    os.getcwd()
    data_directory = [file for file in os.listdir(data_path) if isfile(join(data_path, file))]
    for file in data_directory:
        name = str(file)
        # get the OI images and store them in new path
        if not name.__contains__("T00") and not name.__contains__("T02") and not name.__contains__(
                "T05") and not name.__contains__("T08") and not name.__contains__("T10"):
            image = cv2.imread(os.path.join(data_path, file), cv2.IMREAD_GRAYSCALE)
            image = np.dstack([image, image, image])
            image = cv2.resize(image, (224, 224))
            cv2.imwrite(os.path.join(final_path, file), image)


if __name__ == '__main__':
    # where are the current images located at
    image_path = PureWindowsPath(r"Proper_Dataset\Full")
    # this will be later the oi_path
    final_path = PureWindowsPath("New_Dataset\\OI")
    # the cropped_images folder
    data_path = PureWindowsPath("C:\\Users\\X\\Desktop\\pythonProject\\ObjectDetection\\Cropped_Images")

    # folder of the dataset for CHECK
    new_path = "Proper_Dataset\\"
    # read_feature_scores()
    # convert_to_csv()
    # create_proper_dataset(image_path, new_path)
    # assign_oi(final_path, data_path)

