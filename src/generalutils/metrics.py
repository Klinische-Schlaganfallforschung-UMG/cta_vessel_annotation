"""
This file defines the metrics used for the network performance assessment.
"""

import warnings

import numpy as np
import tensorflow as tf
from keras import backend as K
from sklearn.metrics import confusion_matrix


def dice_multiclass(smooth=0, ignore_background=0, n_classes=26):
    """DICE coefficient for multiclass classification for integer inputs (not one-hot-encoded).

    Computes the DICE coefficient as average over multiple classes. Also known as F1-score or F-measure.
    Can ignore background label 0.

    :param y_true: Ground truth target values.
    :param y_pred: The logits.
    :param smooth: Smoothing factor.
    :param ignore_background: 0 for compute the average witch background label. 1 for compute the average without
    background label.
    :param n_classes: Number of classes to predict.
    :return: DICE coefficient as average of all classes in multilabel classification.
    """

    def dice_multi(y_true, y_pred):
        y_true = K.one_hot(K.cast(y_true, 'uint8'), num_classes=n_classes)
        y_true_f = K.reshape(y_true[..., ignore_background:], (-1, n_classes))
        y_pred_f = K.reshape(y_pred[..., ignore_background:], (-1, n_classes))
        intersect = K.sum(y_true_f * y_pred_f, axis=0)
        numerator = 2. * intersect
        denominator = K.sum(y_true_f + y_pred_f, axis=0)
        dice = K.mean((numerator + smooth) / (denominator + smooth))
        return dice

    return dice_multi


def balanced_accuracy_score(y_true, y_pred, sample_weight=None,
                            adjusted=False):
    """Compute the balanced accuracy

    The balanced accuracy in binary and multiclass classification problems to
    deal with imbalanced datasets. It is defined as the average of recall
    obtained on each class.

    The best value is 1 and the worst value is 0 when ``adjusted=False``.

    This is a re-implementation of the balanced_accuracy_score for scikit-learn. The only difference is that this
    function returns also the average class accuracy per each class, which takes place at the end of the code in:
    > return score, per_class

    Parameters
    ----------
    y_true : 1d array-like
        Ground truth (correct) target values.

    y_pred : 1d array-like
        Estimated targets as returned by a classifier.

    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.

    adjusted : bool, default=False
        When true, the result is adjusted for chance, so that random
        performance would score 0, and perfect performance scores 1.

    Returns
    -------
    balanced_accuracy : float

    Notes
    -----
    Some literature promotes alternative definitions of balanced accuracy. Our
    definition is equivalent to :func:`accuracy_score` with class-balanced
    sample weights, and shares desirable properties with the binary case.
    See the :ref:`User Guide <balanced_accuracy_score>`.

    References
    ----------
    .. [1] Brodersen, K.H.; Ong, C.S.; Stephan, K.E.; Buhmann, J.M. (2010).
           The balanced accuracy and its posterior distribution.
           Proceedings of the 20th International Conference on Pattern
           Recognition, 3121-24.
    .. [2] John. D. Kelleher, Brian Mac Namee, Aoife D'Arcy, (2015).
           `Fundamentals of Machine Learning for Predictive Data Analytics:
           Algorithms, Worked Examples, and Case Studies
           <https://mitpress.mit.edu/books/fundamentals-machine-learning-predictive-data-analytics>`_.
    """
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class = np.diag(C) / C.sum(axis=1)
    if np.any(np.isnan(per_class)):
        warnings.warn('y_pred contains classes not in y_true')
        per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score, per_class
