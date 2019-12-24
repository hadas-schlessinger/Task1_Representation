import os
import numpy as np
import os
import cv2
import pickle
import numpy as np
import skimage.feature
import skimage.color as color
import sklearn
import sklearn.svm
from matplotlib import pyplot as plt
from skimage.feature import hog


####################### data preparation #######################


def get_default_parameters():
    '''
        Returns a dict containing the default experiment parameters.
        Each experiment starts by configuring the experiment parameters.
    '''

    parms = {
        'data':
            {
                'data_path': os.path.join(os.getcwd(), '101_ObjectCategories'),
                'class_indices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'results_path': os.path.join(os.getcwd(), 'Task1_Representation', 'Results'),
                'results_file_name': 'results.pkl',
                'train_test_size': 20,
                'tuning_size': 10
            },
        'prepare':
            {
                'pixels_per_cell': 20,
                'cells_per_block': 4,
                'S': 200
            },
        'train':
            {
                'c': 1,
                'kernel': 'rbf',
                'gamma': 'scale',
                'degree': 2

            },
        'pickle':
            {
                'pickle_path': os.path.join(os.getcwd(), 'pickle'),
                'pickle_train': 'train.pkl',
                'pickle_test': 'test.pkl',
                'first_run': False
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
        train_counter = 0
        test_counter = 0
        for file in list_files:
            image = cv2.cvtColor(cv2.imread(os.path.join(data_details['data_path'], class_name, file)), cv2.COLOR_BGR2GRAY)
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
    if (parms['pickle']['first_run']):
        train, test = _extract__images_from_folders(parms['data'])
        pickle_train_file_name = os.path.join(parms['pickle']['pickle_path'], parms['pickle']['pickle_train'])
        pickle_test_file_name = os.path.join(parms['pickle']['pickle_path'], parms['pickle']['pickle_test'])
        pickle.dump(train, open(pickle_train_file_name, 'wb'))
        pickle.dump(test, open(pickle_test_file_name, 'wb'))
        return train, test
    return load_data(parms, parms['pickle']['pickle_train']), load_data(parms, parms['pickle']['pickle_test'])


def prepare(params, data):
    # Compute the representation function: Turn the images into vectors for classification
    ready_data = {
        'data': [],
        'labels': []
    }
    for img in data['data']:
        img = cv2.resize(img, (params['prepare']['S'], params['prepare']['S']))
        converted_image = hog(img, orientations=8,pixels_per_cell=(params['prepare']['pixels_per_cell'], params['prepare']['pixels_per_cell']),
                              cells_per_block=(params['prepare']['cells_per_block'], params['prepare']['cells_per_block']))
        ready_data['data'].append(converted_image)
    for label in data['labels']: ready_data['labels'].append(label)
    return ready_data


def load_data(params, file_name):
    pickle_path = os.path.join(params['pickle']['pickle_path'], file_name)
    return pickle.load(open(pickle_path, 'rb'))

####################### model #######################




def _create_labels(labels, current_class):
    '''
        this func is for -1 and 1 classifies for 1vs all svm
        for each class the samples of the class get the label 1 and the rest of the classes get -1
    '''

    fixed_labels = np.zeros((len(labels),))
    for i in range(len(labels)):
        if (labels[i] == current_class):
            fixed_labels[i] = 1
        else:
            fixed_labels[i] = -1
    return fixed_labels


def _svm(hog_data, fixed_labels, training_params):
    # C-Support Vector Classification
    svm = sklearn.svm.SVC(kernel=training_params['kernel'], C=training_params['c'], gamma=training_params['gamma'],
                          degree=training_params['degree'], probability=True)
    model = svm.fit(hog_data, fixed_labels)  # fit the model per each binary classifier
    return model


def _m_classes_svm_train(hog_data, data_labels, params):
    all_svms = []
    for current_class in (params['data']['class_indices']):
        class_name = sorted(os.listdir(params['data']['data_path']), key=str.lower)[current_class - 1]  # extract the class name for matching the lables
        fixed_labels = _create_labels(data_labels, class_name)  # appending -1 or 1 if the class matched
        all_svms.append(_svm(hog_data, fixed_labels, params['train']))
    return all_svms


def _m_classes_predict(hog_data, m_classes_svm, class_indices, data_path):
    predictions = []
    score_matrix = np.zeros((len(hog_data), len(class_indices)))
    for current_class in range(len(class_indices)):
        prob = m_classes_svm[current_class].predict_proba(hog_data)  # probability for an image to be in 1 of 2 classes
        for sample in range(len(prob)):
            score_matrix[sample, current_class] = prob[sample, 1]  # the score for sample i is to be from class 1 (j)
    for y in range(len(hog_data)):
        results_per_image = score_matrix[y, :] # takes all m probabilities for each sample
        # find max probability and put the labels of it in predictions
        predictions.append(sorted(os.listdir(data_path), key=str.lower)[class_indices[np.argmax(results_per_image)]-1])
    return score_matrix, predictions


def tuning(params, train):
    train_data = prepare(params, train)
    pass


def train_model(train_data, data_labels, params):
    return _m_classes_svm_train(train_data, data_labels, params)


def test_model(test_data, trained_model, data_details):
    return _m_classes_predict(test_data, trained_model, data_details['class_indices'], data_details['data_path'])


def evaluate(score_matrix, predictions, test_labels):
    # Compute the results statistics - error rate and confusion matrix
    confusion_matrix = sklearn.metrics.confusion_matrix(test_labels, predictions)
    # i = 0
    # error = 0
    # for i in range(len(predictions)):
    #     if (predictions[i] != labels_test[i]):  # counts the number of unmatched true label array to prediction array
    #         error = error + 1
    #
    # error_rate = error / len(labels_test)
    return error_rate, confusion_matrix


def report_results(summary, params):
    # print the error results and confusion matrix and error images
    # Draws the results figures, reports results to the screen
    # Saves the results to the results path, to a file named according to the experiment name or number (e.g. to Results\ResultsOfExp_xx.pkl)
    return True


################# main ####################


def main():
    params = get_default_parameters()   # (experiment specific parameters override)
    np.random.seed(0)  # seed
    train, test = set_and_split_data(params)
#   tuning(train)
    train_data = prepare(params, train)
    test_data = prepare(params, test)
    model = train_model(train_data['data'], train_data['labels'], params)
    score_matrix, predictions = test_model(test_data['data'], model, params['data'])
    summary = evaluate(score_matrix, predictions, test_data['labels'])
    # report_results(summary)


if __name__ == "__main__":
    main()
