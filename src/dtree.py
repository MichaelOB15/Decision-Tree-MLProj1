import argparse
import os.path
import numpy as np
import pandas as pd

from typing import Sequence

from sting.classifier import Classifier
from sting.data import AbstractDataSet


class DecisionTree(Classifier):
    def train(self, data: AbstractDataSet) -> None:
        """
        You will implement this method.

        :param data: Labelled dataset with which to train the decision tree.
        """
        features, labels = data.unzip()

        raise NotImplementedError()

    def predict(self, data: AbstractDataSet) -> Sequence[int]:
        """
        You will implement this method.

        :param data: Unlabelled dataset to make predictions
        :return: Predictions as a sequence of 0s and 1s. Any sequence (list, numpy array, generator, pandas Series) is
        acceptable.
        """
        raise NotImplementedError()

    def nominal_entropy(data, feature):
        sum = 0
        for value in np.unique(data[feature]):
            p = len(data[data.feature == value]) / len(data)
            sum += (p * np.log2(p))
        return sum

    def continuous_entropy(data, test):
        return nominal_entropy(data[data.test[0] <= test[1]], "label") + nominal_entropy(data[data.test[0] > test[1]], "label")

    def entropy(data, test):
        if type(test) is tuple:
            return continuous_entropy(data, test)
        else:
            return nominal_entropy(data, test)

    def split_data_on_test(data, test):
        datasets = []
        if type(test) is tuple:
            datasets.append(data[data.test[0] <= test[1]])
            datasets.append(data[data.test[0] > [test[1]]])
        else:
            for value in np.unique(data[test]):
                datasets.append(data[data.test == value])

        return datasets

    def information_gain(data, test):
        ig = entropy(data, "label")
        for frame in split_data_on_test(data, test):
            ig -= entropy(frame, "label")
        return ig

    def gain_ratio(data, test):
        return information_gain(data, test) / entropy(data, test)


    def get_continuous_splits(data, feature):
        data.sort_values(by=feature, inplace=True)
        splits = []
        for i in range(1, len(data)):
            j = i - 1
            if data[i, 'label'] != data[j, 'label']:
                splits.append((feature, ((data[i, feature] + data[j, feature]) / 2)))
        return splits

    def get_tests(data: AbstractDataSet):
        features = data.schema()
        df, y = data.unzip()
        df["label"] = y
        tests = []
        for feature in features:
            if feature.ftype == Feature.type.CONTINUOUS:
                tests.extend(get_continuous_splits(df, feature))
            elif feature.ftype == Feature.type.NOMINAL or feature.ftype == Feature.type.BINARY:
                tests.append(feature)
        return tests

    def get_best_test(data, tests, gr):
        if gr:
            f = gain_ratio
        else:
            f = information_gain
        best = 0
        for test in tests:
            best = max(f(data, test), best)
        return best

    def id3(data, tests, height, root):
        if len(tests) == 0 or height == 0:
            root.name = data.mode()["label"]
            return
        else:
            test = get_best_test(data, tests, bool)
            if information_gain(data, test) == 0:
                root.name = data.mode()["label"]
                return
            else:
                tests.remove(test)
                root.name = test
                if type(test) is tuple:
                    l = Node('true')
                    r = Node('false')
                    root.add_child(l)
                    id3(data[data.test[0] <= test[1]], tests.copy(), height - 1, l)
                    root.add_child(r)
                    id3(data[data.test[0] > test[1]], tests.copy(), height - 1, r)
                else:
                    for value in np.unique(data[test]):
                        n = Node(value)
                        root.add_child(n)
                        id3(data[data.test == value], tests.copy(), height-1, n)




class Node:
    def __init__(self, edge):
        self.children = np.array([])
        self.edge = edge
        self.name

    def add_child(child):
        self.children.append(child)


def evaluate_dtree(dtree: DecisionTree, dataset: AbstractDataSet):
    """
    You will implement this method.
    Given a trained decision tree and labelled dataset, Evaluate the tree and print metrics.

    :param dtree: Trained decision tree
    :param dataset: Testing set
    """

    acc = 0.3254328
    print(f'Accuracy:{acc:.2f}')
    print('Size:', 0)
    print('Maximum Depth:', 0)
    print('First Feature:', 'asdf')

    raise NotImplementedError()


def dtree(data_path: str, tree_depth_limit: int, use_cross_validation: bool = True, information_gain: bool = True):
    """
    It is highly recommended that you make a function like this to run your program so that you are able to run it
    easily from a Jupyter notebook.

    :param data_path: The path to the data.
    :param tree_depth_limit: Depth limit of the decision tree
    :param use_cross_validation: If True, use cross validation. Otherwise, run on the full dataset.
    :param information_gain: If true, use information gain as the split criterion. Otherwise use gain ratio.
    :return:
    """

    raise NotImplementedError()


if __name__ == '__main__':
    """
    THIS IS YOUR MAIN FUNCTION. You will implement the evaluation of the program here. We have provided argparse code
    for you for this assignment, but in the future you may be responsible for doing this yourself.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Run a decision tree algorithm.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data.')
    parser.add_argument('depth_limit', metavar='DEPTH', type=int,
                        help='Depth limit of the tree. Must be a non-negative integer. A value of 0 sets no limit.')
    parser.add_argument('--no-cv', dest='cv', action='store_false',
                        help='Disables cross validation and trains on the full dataset.')
    parser.add_argument('--use-gain-ratio', dest='gain_ratio', action='store_true',
                        help='Use gain ratio as tree split criterion instead of information gain.')
    parser.set_defaults(cv=True, gain_ratio=False)
    args = parser.parse_args()

    # If the depth limit is negative throw an exception
    if args.depth_limit < 0:
        raise argparse.ArgumentTypeError('Tree depth limit must be non-negative.')

    # You can access args with the dot operator like so:
    data_path = os.path.expanduser(args.path)
    tree_depth_limit = args.depth_limit
    use_cross_validation = args.cv
    use_information_gain = not args.gain_ratio

    dtree(data_path, tree_depth_limit, use_cross_validation, use_information_gain)
