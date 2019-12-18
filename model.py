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


def train(train_data, training_parameters, class_indices):
    # the functions implementing the actual learning algorithm and the classifier
    m_classes_svm = _m_classes_svm_train(class_indices)
    results = _m_classes_predict(m_classes_svm)
    return results


def tuning(train_data):
    # If hyper parameter tuning is required, replace the call to train() in main,
    # with a call to TrainWithTuning() which internally also does the hyper parameter tuning (see slides 11-14)
    # do on class 1-10
    pass


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
    pass


def _m_classes_svm_train():
    _svm()
    pass


def _m_classes_predict(m_classes_svm):
    pass


def _svm():
    pass

