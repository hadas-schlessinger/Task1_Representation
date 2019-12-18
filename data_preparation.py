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
    # These are ‘split’, ‘prepare’, ‘train’, ‘Summary’, ‘Report’ (according to the needs)
    # Each struct is sent to the relevant function (i.e. Params[‘train’] is sent to train(), etc.)
    # Each experiment starts by configuring the experiment parameters:
    # Changing relevant parameters according to the specific experiments needs
    # Do not keep hidden constants in the code (use parameters to set them)

    parms = {
        'data':
            {
                'data_path': data_path,
                'class_inc': class_indices,
                'results_path': os.path.join(os.getcwd(), 'Task1_Representation', 'Results'),
                'results_file_name': 'results.pkl'
            },
        'prepare':
            {
                'pixels_per_cell': 20,
                'cells_per_block': 4,
                'S': 200
            },
        'train':
            {
                'svm_penalty': 1,
                'poly_degree': 2
            },
        'split': {
            'train_test_size': 20,
            'tuning_size': 10
        },
        'pickle':
            {
                'pickle_path': os.path.join(os.getcwd(), 'pickle'),
                'pickle_train': 'train.pkl',
                'pickle_test': 'test.pkl'
            }
    }
    return parms


def _extract__images_from_folders(data_details):
    fixed_train_data = {
        'data': [],
        'labels': []
    }
    fixed_test_data = {
        'data': [],
        'labels': []
    }
    class_indices = data_details['class_indices']

    for class_number in class_indices:
        class_name = sorted(os.listdir(data_details['data_path']),key=str.lower)[class_number-1]
        list_files = sorted(os.listdir(os.path.join(data_details['data_path'], class_name)))
        train_counter, test_counter = 0
        for file in list_files:
            image = cv2.imread(os.path.join(data_details['data_path'], class_name, file))

            if file.endswith('.jpg') and train_counter < data_details['train_test_size']:
                fixed_train_data['data'].append(image)
                fixed_train_data['labels'].append(class_name)
                train_counter = train_counter + 1
            else:
                if file.endswith('.jpg') and test_counter < data_details['train_test_size']:
                    fixed_test_data['data'].append(image)
                    fixed_test_data['labels'].append(class_name)
                    test_counter = test_counter + 1
                else:
                    break
    return fixed_train_data, fixed_test_data


def set_and_split_data(parms):
    # each time we change the data classes
    train, test = _extract__images_from_folders(parms['data'])
    pickle_train_file_name = os.path.join(parms['pickle']['pickle_path'], parms['pickle']['pickle_train'])
    pickle_test_file_name = os.path.join(parms['pickle']['pickle_path'], parms['pickle']['pickle_test'])
    pickle.dump(train, open(pickle_train_file_name, 'wb'))
    pickle.dump(test, open(pickle_test_file_name, 'wb'))


def prepare(params, pkl_name):
    # Compute the representation function: Turn the images into vectors for classification
    data = load_data(params, pkl_name)
    ready_data = {
        'data': [],
        'labels': []
    }
    for img, label in data['data'], data['labels']:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (params['prepare']['S'], params['prepare']['S']))
        converted_image = hog(img, orientations=8,pixels_per_cell=(params['prepare']['pixels_per_cell'], params['prepare']['pixels_per_cell']),
                              cells_per_block=(params['prepare']['cells_per_block'], params['prepare']['cells_per_block']))
        ready_data['data'].append(converted_image)
        ready_data['labels'].append(label)
    return ready_data


def load_data(params, file_name):
    pickle_file_name = os.path.join(params['pickle']['pickle_path'], file_name)
    return pickle.load(open(pickle_file_name, 'rb'))


