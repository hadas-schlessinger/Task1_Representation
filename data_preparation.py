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
                "ResultsFileName": 'ResultsOfEx1.pkl'
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
        "Cache":
            {
                "CachePath": os.path.join(os.getcwd(), 'Task1_Representation', 'Cache'),
                "CacheFileName": '1.pkl',
                "UseCacheFlagForXX": False
            }
    }
    return parms


def get_data(path_to_data):
    data = _load_data(path_to_data)
    cv2.imread()
    cv2.rgb2gray()
    cv2.imresize()
    _save_pickle(data)
    # Loads the data and subsets it if required
    # Puts the data in DandL[‘Data’], the labels in DandL[‘Labels’],
    # In our case: Params include a path for the data and sub - setting parameters

    pass


def train_split_data(data, lables, split):
    # Splits the data and labels according to a ratio defined in Params
    # SplitData includes fields: TrainData, TestData, TrainLabels, TestLabels
    pass


def prepare(train_data, parms):
    # Compute the representation function: Turn the images into vectors for classification
    pass


def _load_data(path):
    os.listdir(path)
    pass


def _save_pickle(data):
    pass
