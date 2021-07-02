import math
import random
from typing import Tuple, Iterable

import numpy as np
from sting.data import AbstractDataSet

"""
This is where you will implement helper functions and utility code which you will reuse from project to project.
Feel free to edit the parameters if necessary or if it makes it more convenient.
Make sure you read the instruction clearly to know whether you have to implement a function for a specific assignment.
"""


def contingencyTable(y: Iterable[int], y_hat: Iterable[int]):
    tp, fp, fn, tn = 0
    for i in range(1, len(y)):
        predicted_label = y_hat[i]
        true_label = y[i]
        # True Positive
        if predicted_label == 1 & true_label == 1:
            tp += 1
        # False Positive
        if predicted_label == 1 & true_label == 0:
            fp += 1
        # False Negative
        if predicted_label == 0 & true_label == 1:
            fn += 1
        # True Negative
        else:
            tn += 1
    return [tp, fp, fn, tn]


def cv_split(dataset: AbstractDataSet, folds: int, stratified: bool = False) -> Tuple[AbstractDataSet, ...]:
    """
    You will implement this function.
    Perform a cross validation split on a dataset and return the cross validation folds.

    :param dataset: Dataset to be split.
    :param folds: Number of folds to create.
    :param stratified: If True, create the splits with stratified cross validation.
    :return: A tuple of the dataset splits.
    """
    dataset.shuffle()
    foldatasize = math.floor(dataset.size / folds)
    newdataset = AbstractDataSet()
    folddata = ()
    if stratified:
        # percentage of 1's
        x = 0
        dataset1, dataset0 = AbstractDataSet()
        for data in dataset:
            if data.labels == 1:
                x += 1
                dataset1.append(data)
            else:
                dataset0.append(data)

        percent1 = x / len(dataset)

        num1s = percent1 * dataset1
        num0s = (1 - percent1) * dataset0

        for i in range(0, folds):
            for j in range(1, num1s):
                newdataset.append(dataset1.drop())

            for j in range(1, num0s):
                newdataset.append(dataset0.drop())

            folddata += newdataset
            newdataset = AbstractDataSet()

    else:
        for data in dataset:
            if len(newdataset) == foldatasize:
                folddata += newdataset
                newdataset = AbstractDataSet()
            else:
                newdataset.append(data)

    return folddata

    # Set the RNG seed to 12345 to ensure repeatability
    # np.random.seed(12345)
    # random.seed(12345)


def accuracy(y: Iterable[int], y_hat: Iterable[int]) -> float:
    """  You will implement this function.
     Evaluate the accuracy of a set of predictions.

     :param y: Labels (true data)
     :param y_hat: Predictions
     :return: Accuracy of predictions
     """
    table = contingencyTable(y, y_hat)
    return (table[0] + table[3]) / sum(table)


def precision(y: Iterable[int], y_hat: Iterable[int]) -> float:
    """  You will implement this function.
    Evaluate the precision of a set of predictions.

    :param y: Labels (true data)
    :param y_hat: Predictions
    :return: Precision of predictions
    """
    table = contingencyTable(y, y_hat)
    return table[0] / (table[0] + table[1])


def recall(y: Iterable[int], y_hat: Iterable[int]) -> float:
    """
    You will implement this function.
    Evaluate the recall of a set of predictions.

    :param y: Labels (true data)
    :param y_hat: Predictions
    :return: Recall of predictions
    """
    table = contingencyTable(y, y_hat)
    return table[0] / (table[0] + table[2])


def roc_curve_pairs(y: Iterable[int], p_y_hat: Iterable[int]) -> Iterable[Tuple[float, float]]:
    """
    You will implement this function.
    Find pairs of FPR and TPR of prediction probabilities based on different decision thresholds.
    You can use this function to implement plot_roc_curve and auc.

    :param y: Labels (true data)
    :param p_y_hat: Classifier predictions (probabilities)
    :return: pairs of FPR and TPR
    """

    return [1, 1]


def auc(y: Iterable[int], p_y_hat: Iterable[int]) -> float:
    """
    You will implement this function.
    Calculate the AUC score of a set of prediction probabilities.

    :param y: Labels (true data)
    :param p_y_hat: Classifier predictions (probabilities)
    :return: AUC score of the predictions
    """
    lhs = 0
    rhs = 0
    roc_pairs = roc_curve_pairs(y, p_y_hat)
    n = len(roc_pairs)
    # SORT IN ORDER OF FPR
    i = 0
    j = 1
    while (j < n):
        rhs += roc_pairs[j][1] * (roc_pairs[j][0] - roc_pairs[i][0])
        lhs += roc_pairs[i][1] * (roc_pairs[j][0] - roc_pairs[i][0])
        i += 1
        j += 1

    return (rhs + lhs) / 2
