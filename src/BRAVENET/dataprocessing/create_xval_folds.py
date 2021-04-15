"""
This script creates train/test folds for crossvalidation from given dataset.

File: create_xval_folds.py
Author: Jana Rieger
Created on: 18.05.2020
"""

import os

import numpy as np

from src.BRAVENET import bravenet_config
from src.generalutils import helper

print('create_xval_folds.py', os.getcwd())

np.random.seed(1)


def xval_folds(patients, nr_folds=1, save_path=''):
    """
    Splits patients into given number of train/val folds. The folds are saved in a dictionary.

    :param patients: List of strings. List of patient names.
    :param nr_folds: Positive int. Number of folds.
    :param save_path: String. JSON path to where the fold dictionary shall be saved.
    :return: The fold dictionary.
    """

    # Separate 1kplus and PEG patients.
    kplus_patients = [patient for patient in patients if patient.startswith('1')]
    print("Found %d 1kplus patients." % len(kplus_patients))
    peg_patients = [patient for patient in patients if patient.startswith('0')]
    print("Found %d PEGASUS patients." % len(peg_patients))
    print("%d in total." % len(kplus_patients + peg_patients))

    # Shuffle sets.
    np.random.shuffle(kplus_patients)
    np.random.shuffle(peg_patients)

    # Set test set ratio.
    test_ratio = 1.0 / nr_folds

    # Break dataset into train and test folds.
    folds_dict = {}
    kplus_fold_size = int(np.floor(len(kplus_patients) * test_ratio))
    peg_fold_size = int(np.floor(len(peg_patients) * test_ratio))

    for f in range(nr_folds):
        fold = {}
        kplus_test_fold = kplus_patients[f * kplus_fold_size: (f + 1) * kplus_fold_size]
        kplus_train_fold = kplus_patients[0: f * kplus_fold_size] + kplus_patients[
                                                                    (f + 1) * kplus_fold_size: len(kplus_patients)]

        peg_test_fold = peg_patients[f * peg_fold_size: (f + 1) * peg_fold_size]
        peg_train_fold = peg_patients[0: f * peg_fold_size] + peg_patients[(f + 1) * peg_fold_size: len(peg_patients)]

        # Store patients in fold.
        test_fold = peg_test_fold + kplus_test_fold
        fold['test'] = {'working': test_fold,
                        'size': len(test_fold)}
        train_fold = peg_train_fold + kplus_train_fold
        fold['train'] = {'working': train_fold,
                         'size': len(train_fold)}

        # Store fold in folds dict.
        folds_dict[f] = fold

    # Save folds dict as json.
    helper.save_json(folds_dict, save_path)
    return folds_dict


if __name__ == '__main__':
    # Retrieve all patients.
    patients = bravenet_config.PATIENTS['train']['working'] + bravenet_config.PATIENTS['val']['working'] + \
               bravenet_config.PATIENTS['test']['working']

    # Define number of folds.
    nr_folds = 4

    # Define save path.
    save_path = os.path.join(bravenet_config.TOP_LEVEL_DATA, 'src', 'BRAVENET', 'xval_folds.json')

    # Xreate folds.
    xval_folds(patients, nr_folds, save_path)

    # Sanity check.
    folds = helper.load_json(save_path)
    train_set = set()
    test_set = set()
    for f, fold in folds.items():
        train_set.update(fold['train']['working'])
        test_set.update(fold['test']['working'])
    print('train len', len(train_set))
    print('test len', len(test_set))

    print('DONE.')
