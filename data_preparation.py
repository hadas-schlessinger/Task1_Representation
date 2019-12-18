import os
import cv2
import pickle
import numpy as np
import skimage.feature
import skimage.color
import sklearn
from matplotlib import pyplot as plt
from skimage.feature import hog



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
                'PicklePath': os.path.join(os.getcwd(), 'Pickle'),
                'PickleTrain': 'train.pkl',
                'PickleTest': 'test.pkl'
            }
    }
    return parms


def _extract__images_from_folders(data_details):

    fixed_train_data= {
        'train_data': [],
        'train_labels': []
    }
    fixed_test_data = {
        'test_data': [],
        'test_labels': []
    }
    class_indices = data_details['ClassIndices']

    for class_number in class_indices:
        class_name = sorted(os.listdir(data_details['DataPath']),key=str.lower)[class_number-1]
        list_files = sorted(os.listdir(os.path.join(data_details['DataPath'], class_name)))
        train_counter, test_counter = 0
        for file in list_files:
            image = cv2.imread(os.path.join(data_details['DataPath'], class_name, file))

            if file.endswith('.jpg') and train_counter < data_details['TrainTestSize']:
                fixed_train_data['train_data'].append(image)
                fixed_train_data['train_labels'].append(class_name)
                train_counter = train_counter + 1
            else:
                if file.endswith('.jpg') and test_counter < data_details['TrainTestSize']:
                    fixed_test_data['test_data'].append(image)
                    fixed_test_data['test_labels'].append(class_name)
                    test_counter = test_counter + 1
                else:
                    break
    return fixed_train_data, fixed_test_data


def set_and_split_data(parms):
    # each time we change the data classes
    train, test = _extract__images_from_folders(parms['Data'])
    pickle_train_file_name = os.path.join(parms['Pickle']['PicklePath'], parms['Pickle']['PickleTrain'])
    pickle_test_file_name = os.path.join(parms['Pickle']['PicklePath'], parms['Pickle']['PickleTest'])
    pickle.dump(train, open(pickle_train_file_name, "wb"))
    pickle.dump(test, open(pickle_test_file_name, "wb"))




def prepare(params, pkl_name):
    # Compute the representation function: Turn the images into vectors for classification
    data = load_data(params, pkl_name)
    ready_data = {
        'Data': [],
        'Labels': []
    }
    for img, label in data['Data'], data['Labels']:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (params['Prepare']['S'], params['Prepare']['S']))
        converted_image = hog(img, orientations=8,pixels_per_cell=(params['Prepare']["PixelsPerCell"], params['Prepare']["PixelsPerCell"]),
                              cells_per_block=(params['Prepare']["CellsPerBlock"], params['Prepare']["CellsPerBlock"]))
        ready_data['Data'].append(converted_image)
        ready_data['Labels'].append(label)
    return ready_data


def load_data(params, file_name):
    pickle_file_name = os.path.join(params['Pickle']['PicklePath'], file_name)
    return pickle.load(open(pickle_file_name, "rb"))


