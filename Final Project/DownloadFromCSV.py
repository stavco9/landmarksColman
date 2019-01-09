
# coding: utf-8

# In[ ]:


import tensorflow as tf
import pandas as pd
import random
import os
import requests
import shutil

# os.system("qsub -I -V -N kukk -l nodes=1:gpus=8:V100")

_TRAIN_FOLDER_ = "D:\Images\Train\\"


def getRandomClasses(file=None, total=100):
    if file is None:
        return []

    print("Loading " + file + "...")
    train = pd.read_csv(file, skiprows=1, header=None, names=["#", "Image", "Class"])

    print("Removing empty lines ...")
    train.dropna()

    print("Getting unique classes")
    classes = train['Class'].unique().tolist()

    print("Shuffling classes...")
    random.shuffle(classes)

    print("Scanning classes...")
    list = []
    for _class in classes:
        images = train.loc[train['Class'] == _class]
        if _class == 'None':
            continue

        if images.shape[0] < 100:
            continue

        list.append(_class)

        if len(list) >= 100:
            break

    return train, list[:total]

def createFolder(folder = None):
    if folder is None:
        return

    if not os.path.exists(folder):
        os.makedirs(folder)

def createTrainingCsvData(folder = None):
    if folder is None:
        print("No training data supplied")
        return

    # Create training folder
    createFolder(folder)

    # Generate 100 classes out of the landmark database
    train, classes = getRandomClasses('D:\Images\\train.csv')

    print("Found "+str(len(classes)) + " classes")

    # generate csv files with images
    for _class in classes:
        with open(_TRAIN_FOLDER_ + _class + ".csv", 'a+') as f:
            print("Exporting " + _class + "...")
            images = train.loc[train['Class'] == _class]
            images.to_csv(_TRAIN_FOLDER_ + _class + ".csv", header=False, index=False)

    print("landmark csv data generated successfully.")

def fetch_image(path,folder, name):
    url = path

    try:
        response = requests.get(url, stream=True)
        with open(folder + name + '.jpg', 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response
    except OSError:
        print("Can't download " + path)

def downloadTrainingImages(folder = None):
    if folder is None:
        print("No training data supplied")
        return

    if not os.path.exists(folder):
        print("Training data folder does not exists")
        return

    if len(os.listdir(folder)) == 0:
        print("Training folder is empty")
        return
    else:
        files = []
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            if os.path.isdir(path) or fname == ".DS_Store":
                # skip directories
                continue
            files.append(fname)

        counter = 1
        for file in files:
            new_folder = folder + file.replace(".csv", '')
            print("[" + str(counter) + "/" + str(len(files)) + "] Creating " + new_folder + " folder...")
            createFolder(new_folder)

            print("[" + str(counter) + "/" + str(len(files)) + "] Loading " + file + "...")
            images = pd.read_csv(folder + file, names=["#", "Image", "Class"])
            images.drop(['Class'], axis=1, inplace=True)
            # images = images['Image'].tolist()

            for index, img in images.iterrows():
                print("[" + str(counter) + "/" + str(len(files)) + "] - [" + str(index+1) + "/" + str(images.shape[0]) + "] Downloading " + img["Image"] + "...")
                fetch_image(img["Image"], new_folder+"/", img['#'])
                if (index+1) > 80:
                    break;

            counter = counter + 1
if __name__ == '__main__':
    #createTrainingCsvData(_TRAIN_FOLDER_)
    downloadTrainingImages(_TRAIN_FOLDER_)

