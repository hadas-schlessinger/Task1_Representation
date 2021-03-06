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
import pandas as pd


####################### data preparation #######################


def get_default_parameters():
    '''
        Returns a dict containing the default experiment parameters.
        it contains all tuned parameters .
    '''

    parms = {
        'data':
            {
                'data_path': os.path.join(os.getcwd(), '101_ObjectCategories'),
                'image_path': [],
                'number_of_test_img': [],
                'class_indices': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                'train_test_size': 20
            },
        'prepare':
            {
                'pixels_per_cell': 16,
                'cells_per_block': 3,
                'orientations_bins': 14,
                'S': 196
            },
        'train':
            {
                'c': 0.01,
                'kernel': 'linear',
                'degree': 3

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
    '''
            The function extract the relevant classes according to the class_indices.
            It returns a dict containing the fixed data and its labels.
            The function davids the data into train (20 pics) and test (20 pics or less) and changes it to grey scale
            In addition, the function saves a list of the test images path.
            This list will be used later to present the worst error images
    '''

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
        class_name = sorted(os.listdir(data_details['data_path']), key=str.lower)[class_number-1]
        list_files = sorted(os.listdir(os.path.join(data_details['data_path'], class_name)))
        train_counter = 0
        test_counter = 0
        for file in list_files:
            image = cv2.cvtColor(cv2.imread(os.path.join(data_details['data_path'], class_name, file)), cv2.COLOR_BGR2GRAY)
            if file.endswith('.jpg') and train_counter < data_details['train_test_size']: #train
                fixed_train_data['data'].append(image)
                fixed_train_data['labels'].append(class_name)
                train_counter = train_counter + 1
            else: #test
                if file.endswith('.jpg') and test_counter < data_details['train_test_size']:
                    data_details['image_path'].append(os.path.join(data_details['data_path'], class_name, file))
                    fixed_test_data['data'].append(image)
                    fixed_test_data['labels'].append(class_name)
                    test_counter = test_counter + 1
                else:
                    break
        data_details['number_of_test_img'].append(test_counter)
    return fixed_train_data, fixed_test_data


def set_and_split_data(parms):
    '''
    This is a frame function, it calls the function for extracting and splitting the data.
    In addition, the function is used in order to save a pickle file or to load it.
    if this is the first run (on a different class_indices), then the params dics should contain the value "first_run"= true
    if "first_run" = true -  all new data will be extracted and saved as a pickle.
    if the value is false - the pickle will be loaded.
    '''
    # each time we change the data classes
    if (parms['pickle']['first_run']):
        train, test = _extract__images_from_folders(parms['data'])
        # pickle_train_file_name = os.path.join(parms['pickle']['pickle_path'], parms['pickle']['pickle_train'])
        # pickle_test_file_name = os.path.join(parms['pickle']['pickle_path'], parms['pickle']['pickle_test'])
        # pickle.dump(train, open(pickle_train_file_name, 'wb'))
        # pickle.dump(test, open(pickle_test_file_name, 'wb'))
        return train, test
    return load_data(parms, parms['pickle']['pickle_train']), load_data(parms, parms['pickle']['pickle_test'])


def prepare(params, data, labels):
    '''
     The function prepares the data.
     It receives a list of the data and it labels and resizing each of the data images.
     Then, the function applies the hog procedure on the data, according to the tuned parameters.
     the function returns a dict of the prepared data and it labels
    '''
    # Compute the representation function: Turn the images into vectors for classification
    ready_data = {
        'data': [],
        'labels': []
    }
    for img in data:
        img = cv2.resize(img, (params['prepare']['S'], params['prepare']['S']))
        converted_image = hog(img, orientations=params['prepare']['orientations_bins'],
                              pixels_per_cell=(params['prepare']['pixels_per_cell'], params['prepare']['pixels_per_cell']),
                              cells_per_block=(params['prepare']['cells_per_block'], params['prepare']['cells_per_block']))
        ready_data['data'].append(converted_image)
    for label in labels: ready_data['labels'].append(label)
    return ready_data


def load_data(params, file_name):
    '''
    helping function to load the pickle
    '''
    pickle_path = os.path.join(params['pickle']['pickle_path'], file_name)
    return pickle.load(open(pickle_path, 'rb'))

####################### model #######################

def tuning(params, train):
    '''
    this is the tunning functions, it runs over hyper parameters combinations and saves each combination and it error in a list
    In addition, it writes the results to an excel file.
    The function returnes a list of all combinations and a list of all validation errors for each combination.
    the combinations are made from: kernel, pixels per cell, number of orientation bins and c (penalty).
    '''
    train_of_cross_val, test_of_cross_val, train_label, test_label = train_test_split(train['data'], train['labels'], train_size=0.7)
    tuning_error_per_set = []
    errors =[]
    for kernel in ['linear', 'rbf', 'poly']:
        params['train']['kernel'] = kernel
        for size in [100, 129, 150, 196, 256]:
            params['prepare']['S'] = size
            for pixels_per_cell in [8, 16, 32, 64]:
                params['prepare']['pixels_per_cell'] = pixels_per_cell
                #for cells_per_block in range(2, 6, 1):
                for orientations in range(4, 16, 2):
                    params['prepare']['orientations_bins'] = orientations
                    hog_images_train = prepare(params, train_of_cross_val, train_label)
                    hog_images_test = prepare(params, test_of_cross_val, test_label)
                    for c in [0.01, 0.1, 0.2, 0.3, 1, 3]:
                        params['train']['c'] = c
                        trained_model = train_model(hog_images_train['data'], hog_images_train['labels'], params)
                        score, predictions = test_model(hog_images_test['data'], trained_model, params['data'])
                        error_of_valid, matrix = _evaluate(predictions, hog_images_test['labels'])
                        print(f'this round parameters are {[kernel, size, pixels_per_cell, orientations, c,  error_of_valid]}')
                        tuning_error_per_set.append([kernel, size, pixels_per_cell, orientations, c, error_of_valid])
                        errors.append(error_of_valid)
    # excel writing
    tuning_error_per_set=pd.Series(tuning_error_per_set, name='value').to_frame()
    writer = pd.ExcelWriter(os.path.join(params['pickle']['pickle_path'], 'tune.xlsx'))
    tuning_error_per_set.to_excel(writer, 'df')
    writer.save()
    return tuning_error_per_set, errors

def tuning_blocks(params, train):
    train_of_cross_val, test_of_cross_val, train_label, test_label = train_test_split(train['data'], train['labels'],                                                                          train_size=0.7)
    tuning_error_per_set = []
    errors = []
    for block_size in range(1,7,1):
        params['prepare']['cells_per_block'] = block_size
        hog_images_train = prepare(params, train_of_cross_val, train_label)
        hog_images_test = prepare(params, test_of_cross_val, test_label)
        trained_model = train_model(hog_images_train['data'], hog_images_train['labels'], params)
        score, predictions = test_model(hog_images_test['data'], trained_model, params['data'])
        error_of_valid, matrix = _evaluate(predictions, hog_images_test['labels'])
        print(f'this round block size is {[block_size, error_of_valid]}')
        tuning_error_per_set.append([block_size, error_of_valid])
        errors.append(error_of_valid)
    # excel writing
    tuning_error_per_set = pd.Series(tuning_error_per_set, name='value').to_frame()
    writer = pd.ExcelWriter(os.path.join(params['pickle']['pickle_path'], 'tune2.xlsx'))
    tuning_error_per_set.to_excel(writer, 'df')
    writer.save()
    return tuning_error_per_set, errors

def _create_labels(labels, current_class):
    '''
        this function creates -1 and 1 labels. the function gets the current class and all the labels.
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
    '''
            This function applies a binary svm according to the tuned parameters
            The function return svm classifiers for a specific class
    '''
    svm = sklearn.svm.SVC(kernel=training_params['kernel'], C=training_params['c'],
                          degree=training_params['degree'], probability=True)
    model = svm.fit(hog_data, fixed_labels)  # fit the model per each binary classifier
    return model


def _m_classes_svm_train(hog_data, data_labels, params):
    '''
               This function applies a multi classes svm according to the tuned parameters
               the function claculates m SVMs, according to the number of classes
               The function return multi svm classifiers for all classes
    '''
    all_svms = []
    for current_class in (params['data']['class_indices']):
        class_name = sorted(os.listdir(params['data']['data_path']), key=str.lower)[current_class - 1]  # extract the class name for matching the lables
        fixed_labels = _create_labels(data_labels, class_name)  # appending -1 or 1 if the class matched
        all_svms.append(_svm(hog_data, fixed_labels, params['train']))
    return all_svms


def _m_classes_predict(hog_data, m_classes_svm, class_indices, data_path):
    '''
                   This function applies a multi classes prediction according to a m_svm classifier
                   the function claculates a score for each image to belong to each class
                   an image will be predicted as belongs to the class with thew highst score
    '''
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
    '''
    Compute the results - returns the error rate and the confusion matrix
    '''
    error = sum(1 for predict, real in zip(predictions, test_labels) if predict != real)
    return (error/len(test_labels)), sklearn.metrics.confusion_matrix(test_labels, predictions)


def _class_index(class_number, class_indices):
    return class_indices.index(class_number)


def _calc_margins(score_matrix, test_labels, data_path, class_indices):
    '''
        The function calcs the margin for each image by:
        margin = real class prob score - the max score for the image
        The max score for the image represent the predictive class
        Negative margin will point on the image misclassification
    '''
    margins = np.zeros((len(test_labels),))  # list of margins
    for i in range(len(test_labels)):
        class_number = sorted(os.listdir(data_path), key=str.lower).index(test_labels[i])
        margins[i] = score_matrix[i, _class_index(class_number+1, class_indices)] - np.amax(score_matrix[i, :])
    return margins


def _list_worst_images(margins, img_path, number_of_img):
    '''
    The function lists all the eorst pictures and returns a list of thier path
    the function chooses the images with the lowest negative margin.
    if there is no such error - the list will contain teh value "None"
    the function is built for different sizes of test data
    '''
    error_images = []
    current_class = 0
    num_of_images= 0
    for i in range(0, len(margins), number_of_img[current_class]):
        num_of_images = num_of_images + number_of_img[current_class]
        worst_img_value = min(margins[i:num_of_images]) #worst image margin
        val, = np.where(margins == worst_img_value)
        error_images.append(img_path[val[0]]) if worst_img_value != 0.0 else error_images.append("None")
        margins[val[0]] = 0
        worst_img_value2 = min(margins[i:num_of_images]) #second worst image margin
        val, = np.where(margins == worst_img_value2)
        error_images.append(img_path[val[0]]) if worst_img_value2 != 0.0 else error_images.append("None")
        current_class = current_class+1
    return error_images


def _present_images(list_of_2_worst_images, class_indices, data_path):
    '''
    The function presents the worst images. If there are no errors, the function declares that.
    The function gets the list of the most errored images and examined every two images to check if there were errors.
    for every valid value (not "None") of an image the function presents it to the screen
    '''
    for i in range(len(list_of_2_worst_images)):
        if (list_of_2_worst_images[i] != 'None'): #there is an image
            image = cv2.imread(list_of_2_worst_images[i])
            plt.imshow(image, cmap='gray', interpolation='bicubic')
            plt.show()
        else: #image in none
            if (i % 2 != 0): # if the none if for the second pic in the list
                class_name = sorted(os.listdir(data_path), key=str.lower)[class_indices[round((i-1)/2)]-1]
                if list_of_2_worst_images[i-1] == 'None': #check if the previous was also none
                    print(f'There were no errors for the class: {class_name}')


def report_results(predictions, score_matrix, data_path, test_labels, img_path, number_of_img, class_indices):
    '''
    this is an helping function which calls to all relevant function to present the results.
    i.e - the function reports all the results.
    First - it presents the error rate and the confusion matrix.
    Then - it calculates the margins and uses them to present the worst images
    '''
    error_rate, confusion_matrix = _evaluate(predictions, test_labels)
    print(f'error rate is: {error_rate*100} %')
    print(f'confusion_matrix is:\n {confusion_matrix}')
    margins = _calc_margins(score_matrix, test_labels, data_path, class_indices)
    worst_images = _list_worst_images(margins, img_path, number_of_img)
    _present_images(worst_images,class_indices,data_path)



################# main ####################
def main():
    params = get_default_parameters()
    np.random.seed(0)  # seed
    train, test = set_and_split_data(params)
    # tuning_error_per_set, errors = tuning(params, train)
    #tuning_error_per_set, errors = tuning_blocks(params, train)
    train_data = prepare(params, train['data'], train['labels'])
    test_data = prepare(params, test['data'], test['labels'])
    model = train_model(train_data['data'], train_data['labels'], params)
    score_matrix, predictions = test_model(test_data['data'], model, params['data'])
    report_results(predictions, score_matrix, params['data']['data_path'],
                   test_data['labels'], params['data']['image_path'], params['data']['number_of_test_img'],params['data']['class_indices'])



if __name__ == "__main__":
    main()
