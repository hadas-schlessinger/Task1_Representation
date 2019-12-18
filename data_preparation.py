import os
import cv2
import pickle
import numpy as np
import skimage.feature
import skimage.color
import sklearn
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
            "TrainTestSize": 20,
            "TuningSize": 10
        },
        "Pickle":
            {
                "PicklePath": os.path.join(os.getcwd(), 'Pickle'),
                "PickleFileName": 'data.pkl',
            }
    }
    return parms


def _extract__images_from_folders(data_details):
    # Puts the data in DandL[‘Data’], the labels in DandL[‘Labels’]
    fixed_data = {
        'Data': [],
        'Labels': []
    }
    class_indices = data_details['ClassIndices']
    for class_number in class_indices:
        class_name = sorted(os.listdir(data_details['DataPath']),key=str.lower)[class_number-1]
        counter = 0
        for file in sorted(os.listdir(os.path.join(data_details['DataPath'], class_name))):
            # print(sorted(os.listdir(os.path.join(data_details['DataPath'], class_name))))
            if file.endswith('.jpg') and counter < data_details['MaxNumOfImages']:
                image = cv2.imread(os.path.join(data_details['DataPath'], class_name, file))
                fixed_data['Data'].append(image)
                fixed_data['Labels'].append(class_name)
                counter = counter + 1
    return fixed_data


def set_data(parms):
    # each time we change the data classes
    dand_l = _extract__images_from_folders(parms['Data'])
    pickle_file_name = os.path.join(parms['Pickle']['PicklePath'], parms['Pickle']['PickleFileName'])
    pickle.dump(dand_l, open(pickle_file_name, "wb"))


def split_data(params):
    # Splits the data and labels according to a ratio defined in Params
    # SplitData includes fields: TrainData, TestData, TrainLabels, TestLabels
    '''
           converting the image into vector of features for further use. before
           that the image is turning into grayscale and resized
           :input Params:
               Data - a vector of images which need to be converted
               PrepareParams - see GetDefaultParameters()
           :output Params:
               Resualt: vector contating all the images and their features
        '''
    data = pickle.load(open(, "rb" ))
    DataRep = []
    Resualt = []
    for i in Data:
        i = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        i = cv2.resize(i, (PrepareParams["S"], PrepareParams["S"]))
        DataRep.append(i)

    for i in DataRep:
        ImageRep = hog(i, orientations=8,
                       pixels_per_cell=(PrepareParams["PixelsPerCell"], PrepareParams["PixelsPerCell"]),
                       cells_per_block=(PrepareParams["CellsPerBlock"], PrepareParams["CellsPerBlock"]))
        Resualt.append(ImageRep)
    return Resualt
    pass


def prepare(train_data, parms):
    # Compute the representation function: Turn the images into vectors for classification
    pass


def load_data(params):
    pickle_file_name = os.path.join(params['Pickle']['PicklePath'], params['Pickle']['PickleFileName'])
    return pickle.load(open(pickle_file_name, "rb"))


