import os
import cv2
import pickle
import numpy as np
import skimage.feature
import skimage.color
import sklearn
import sklearn.svm
import sklearn.multiclass
import sklearn.preprocessing
from matplotlib import pyplot as plt

def get_default_parameters(data_path, class_indices):

    # Returns a dict containing the default experiment parameters
    # It has several fields, each itself a dict of parameters for the various experiment stages
    # These are ‘Split’, ‘Prepare’, ‘Train’, ‘Summary’, ‘Report’ (according to the needs)
    # Each struct is sent to the relevant function (i.e. Params[‘Train’] is sent to Train(), etc.)
    # Each experiment starts by configuring the experiment parameters:
    # Changing relevant parameters according to the specific experiments needs
    # Do not keep hidden constants in the code (use parameters to set them)

    parms = {
        "Data":
            {
                "DataPath": data_path,
                "ClassIndices": class_indices,
                "MaxNumOfImages": 40,
                "ResultsPath": os.path.join(os.getcwd(), 'Task1_Representation', 'Results'),
                "ResultsFileName": 'Results.pkl'
            },
        "Prepare":
            {
                "PixelsPerCell": 20,
                "CellsPerBlock": 4,
                "S": 200
            },
        "Train":
            {
                "SvmPenalty": 1,
                "PolyDegree": 2
            },
        "Split": {
            "DefaultSplit": 20,
            "ForTuneSplit": 10
        },
        "Pickle":
            {
                "PicklePath": os.path.join(os.getcwd(), 'Task1_Representation', 'Pickle'),
                "PickleFileName": 'data.pkl',
            }
    }
    return parms


def get_data(parms):
    # each time we change the data classes
    dand_l = _extract__images_from_folders(parms['Data'])
    pickle_file_name = os.path.join(parms['Pickle']['PicklePath'], parms['Pickle']['PickleFileName'])
    return  pickle.dump(dand_l, open(pickle_file_name, "wb"))

def _class_is_input(some_class, path_to_data):
    for i in data_details["class_indices"]:
        if i == os.listdir(path_to_data).index(some_class) + 1:
            return true
    return false

def _extract__images_from_folders(data_details):
    # Puts the data in DandL[‘Data’], the labels in DandL[‘Labels’]
    fixed_data = {
        "Data": [],
        "Lables": []
    }
    for class_name in os.listdir(data_details['data_path']):
        if  _class_is_input(class_name,data_details['data_path']):
            counter = 0
            for file in os.listdir(os.path.join(data_details['data_path'], folder)) and counter < data_details["MaxNumOfImages"]:
                image = cv2.imread(os.path.join(data_details['data_path'], class_name, file))
                fixed_data["Data"].append(image)
                fixed_data["Lables"].append(class_name)
                counter = counter + 1
    return fixed_data


def train_split_data(data, lables, split):
    # Splits the data and labels according to a ratio defined in Params
    # SplitData includes fields: TrainData, TestData, TrainLabels, TestLabels
    pass


def prepare(train_data, parms):
    # Compute the representation function: Turn the images into vectors for classification
    pass

def load_data(pickle_file_name):
    return pickle.load(open(pickle_file_name, "rb"))


