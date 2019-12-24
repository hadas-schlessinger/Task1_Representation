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
from sklearn.model_selection import train_test_split


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
                'cells_per_block': 10,
                'orientations_bins': 4,
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
        converted_image = hog(img, orientations= params['prepare']['number_of_bins'], pixels_per_cell=(params['prepare']['pixels_per_cell'], params['prepare']['pixels_per_cell']),
                              cells_per_block=(params['prepare'] ['cells_per_block'], params['prepare']['cells_per_block']))
        ready_data['data'].append(converted_image)
    for label in data['labels']: ready_data['labels'].append(label)
    return ready_data


def load_data(params, file_name):
    pickle_path = os.path.join(params['pickle']['pickle_path'], file_name)
    return pickle.load(open(pickle_path, 'rb'))

####################### model #######################


def train_model(train_data, data_labels, params):
    # the functions implementing the actual learning algorithm and the classifier
    m_classes_svm = _m_classes_svm_train(train_data, data_labels, params, params['data']['class_indices'])
    results = _m_classes_predict(m_classes_svm)
    return results


def tuning(params,train):
    train_of_cross_val, test_of_cross_val = train_test_split(train['data'], train['labels'], train_size=0.7)
    tuning_error_per_set = []
    for kernel in ['linear', 'rbf', 'poly']:
        params['prepare']['kernel'] = kernel
        for size in [100, 129, 150, 190, 220, 290, 300]:
            params['prepare']['S'] = size
            for pixels_per_cell in [8, 16, 32, 64]:
                params['prepare']['pixels_per_cell'] = pixels_per_cell
                for cells_per_block in range(2, 6, 1):
                    params['prepare']['cells_per_block'] = cells_per_block
                    for orientations in range(4,10,1):
                        params['prepare']['orientations_bins'] = orientations
                        hog_images_train = prepare(params, train_of_cross_val)
                        hog_images_test = prepare(params, test_of_cross_val)
                        for c in [0.01, 0.1, 0.2, 0.3, 1, 10]:
                            params['prepare']['c'] = c
                            if kernel == 'poly':
                                for degree in [2, 3, 4]:
                                    params['prepare']['degree'] = degree
                                    trained_model = train_model(hog_images_train['data'],hog_images_train['labels'],params)
                            else:
                                trained_model = train_model(hog_images_train['data'], hog_images_train['labels'], params)
                            score, predictions = test_model(hog_images_test['data'], trained_model, params['data'])
                            error_of_valid, matrix = _evaluate(predictions, hog_images_test['labels'])
                            tuning_error_per_set.append(kernel, size, pixels_per_cell, orientations, c, error_of_valid)
    return tuning_error_per_set


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


def train_model(train_data, data_labels, params):
    return _m_classes_svm_train(train_data, data_labels, params)


def test_model(test_data, trained_model, data_details):
    return _m_classes_predict(test_data, trained_model, data_details['class_indices'], data_details['data_path'])


def _evaluate(predictions, test_labels):
    # Compute the results statistics - error rate and confusion matrix
    error = sum(1 for predict, real in zip(predictions, test_labels) if predict == real)
    return error / len(test_labels), sklearn.metrics.confusion_matrix(test_labels, predictions)


def _calc_margins(score_matrix, test_lables):
    classesUniqueList = uniquelabels()  # creates a unique list of the classes- real classes - test
    margins = np.zeros((len(test_lables),))  # list of margins
    i = 0
    for i in range(len(labels_test)):
        # loops the images and calculate - the image score for the real label minus the max score for this image
        numbOfClass = numberOfClass(labels_test[i], classesUniqueList)
        # number of class returns for an image the real class it belongs
        # go to the probability and calc the score minus max in row
        margins[i] = score_matrix[i, numbOfClass] - numpy.amax(
            score_matrix[i, :])  # calc score matrix based on max proba
        # marginsVector is a number-of-images vector represents the margin for each image
    return margins


def _list_worst_images(margins, data_path):
    pass


def _present_and_save_images(images):
    pass



################# main ####################


def main():
    params = get_default_parameters()   # (experiment specific parameters override)
    np.random.seed(0)  # seed
    train, test = set_and_split_data(params)
    tuning(params, train)
    train_data = prepare(params, train)
    test_data = prepare(params, test)
    model = train_model(train_data['data'], train_data['labels'], params)
    score_matrix, predictions = test_model(test_data['data'], model, params['data'])
    report_results(predictions, score_matrix, params['data']['data_path'], test_data['labels'], params['data']['class_indices'])


if __name__ == "__main__":
    main()
