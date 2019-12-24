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
                'first_run': True
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
        converted_image = hog(img, orientations=(params['prepare']['orientations']), pixels_per_cell=(params['prepare']['pixels_per_cell'], params['prepare']['pixels_per_cell']),
                              cells_per_block=(params['prepare']['cells_per_block'], params['prepare']['cells_per_block']))
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

    image_size = [100, 129, 150, 190, 220, 290, 300]
    pixels_per_cell = [8, 16, 32, 64]
    c_params = [0.01, 0.1, 0.2, 0.3, 1, 10]
    degree_params = [2,3,4]
    train_of_cross_val, test_of_cross_val = train_test_split(train['data'], train['labels'], train_size=0.7)
    kernel_type =['linear', 'rbf', 'poly']
    hog_cells_per_block = [1]
    tests = []

    for k in kernel_type:
        params['prepare']['kernel'] = k
            for s in image_size:
                params['prepare']['S'] = s
                    for pixels_per_cell in hog_pixels_per_cell:
                        params['prepare']['pixels_per_cell'] = pixels_per_cell
                        for cells_per_block in hog_cells_per_block:
                            params['prepare']['cells_per_block'] = cells_per_block
                            hog_images_train = prepare(params, train_of_cross_val)
                            hog_images_test = prepare(params, test_of_cross_val)
                            for c in c_params:
                                params['prepare']['c'] = c
                                if k == 'poly':
                                    for degree in degree_params:
                                        params['prepare']['degree'] = degree
                                        trained_model = train_model(hog_images_train['data'],hog_images_train['labels'],params)
                                else:
                                    trained_model = train_model(hog_images_train['data'], hog_images_train['labels'], params)
                                results = test(trained_model, test_of_cross_val)
                                summary = evaluate(results)

def test(trained_model, test_data_rep):
    pass


def evaluate(results,split_data, params):
    # Compute the results statistics and return them as fields of Summary For classification these are:
    # Most important: the error rate In our case also:
    # Confusion matrix, the indices of the largest error images
    pass


def report_results(summary, params):
    # print the error results and confusion matrix and error images
    # Draws the results figures, reports results to the screen
    # Saves the results to the results path, to a file named according to the experiment name or number (e.g. to Results\ResultsOfExp_xx.pkl)
    return True


def _create_labels(labels, current_class):
    '''
        this func is for -1 and 1 classifies for 1vs all svm
        for each class the samples of the class get the label 1 and the rest of the classes get -1
    '''

    fixed_labels = np.zeros((len(labels),))
    for i in range(len(labels)):
        if (labels[i]==current_class):
            fixed_labels[i] = 1
        else:
            fixed_labels[i] = -1
    return fixed_labels


def _svm(hog_data, fixed_labels, training_params):
    # C-Support Vector Classification
    svm = sklearn.svm.SVC(kernel=training_params['kernel'], C=training_params['c'], gamma=training_params['gamma'], degree=training_params['degree'], probability=True)
    model = svm.fit(hog_data, fixed_labels)  # train the data - fit the model per each class binary classifier
    return model


def _m_classes_svm_train(hog_data, data_labels, params, class_indices):
    all_svms = []
    for current_class in class_indices:
        class_name = sorted(os.listdir(params['data']['data_path']), key=str.lower)[current_class-1] # extract the class name for matching the lables
        fixed_labels = _create_labels(data_labels, class_name)  # appending -1 or 1 if the class matched
        all_svms.append(_svm(hog_data, fixed_labels, params['train']))
    return all_svms


def _m_classes_predict(m_classes_svm):
    # classesUniqueList = uniquelabels()  # creates a unique list of the classes
    # numberOfImages = len(HOGTestdata)  # define the number of images
    # predictions = []
    # score_matrix = numpy.zeros((numberOfImages, len(classesUniqueList)))
    # results_per_image = numpy.zeros((1, 10))  # results for 1 image - what is the probability for each class
    # j = 0
    # # for per each class and calc the prob for each image
    # for j in range(len(classesUniqueList)):
    #     proba = numpy.zeros((
    #                         10,))  # for per each class and calc the probability for each image (what is the prob to be part of a class)
    #     proba = n_models[j].predict_proba(HOGTestdata)
    #     i = 0
    #     for i in range(len(proba)):
    #         score_matrix[i, j] = proba[i, 1]  # calc score matrix based on max proba
    # y = 0
    # for y in range(numberOfImages):
    #     results_per_image = score_matrix[y, :]
    #     # argmax is the calc by taking the argmax over the class score matrix columns
    #     max = numpy.argmax(results_per_image)
    #     predictions.append(classesUniqueList[max])  # find max argmax and put the lables of it in predictions
    #
    # return score_matrix, predictions
    pass



################# main ####################


def main():
    params = get_default_parameters()   # (experiment specific parameters override)
    np.random.seed(0)  # seed
    train, test = set_and_split_data(params)
    tuning(params, train)
    train_data = prepare(params, train)
    trained_model = train_model(train_data['data'], train_data['labels'], params)
    test_data = prepare(params,test)
    # results = test(trained_model, test_data)
    # summary = evaluate(results)
    # report_results(summary)


if __name__ == "__main__":
    main()
