#!/usr/bin/python
# -*- coding: utf-8 -*-

import os.path, logging
import numpy as np

from data_file_prep import dataset
from data_labeler import labeler
from data_mel_cepstrograms import mel_specs
from data_refining import refine_data

logger = logging.getLogger("CNN_Training.MELS")


def trainingdata(settings: dict) -> tuple:
    """
    This function creates the main datasets for the CNN. If the dataset exists earlier, it is not created again.
    Args:
        settings:     dict main settings like the type of dataset, if it should be created new, etc.
    Returns:
        bool: True if the dataset was created, False if it already exists.
    """

    #Initialize the working directory and variables
    os.chdir('..')
    os.chdir('files')
    size = settings.get('size', 'bees_1')
    path = os.path.join(os.getcwd(), f'training_data_torch_{size}.npy')
    # Check if the file already exists
    if os.path.isfile(path) and not settings.get('create_new', False):
        logger.info(f'Dataset {size} already exists, skipping creation.')
        return load_trainingdata(settings.get('size', 'bees_1'))

    #Chain of creating the training data
    logger.info(f'Creating new dataset and saving it in {path}.')
    logger.info(f'Downloading and sorting the files...')
    dir_list = dataset(settings.get('size', 'bienen_1'), settings)

    logger.info(f'Labeling each file...')
    labels = labeler(dir_list, settings.get('training_file_extensions', 'wav'))

    logger.info(f'Creating Mel-spectrograms...')
    mels = mel_specs(labels, settings)

    logger.info(f'Splitting into test and training dataset and adding the right dimensions for the model...')
    trained_data = refine_data(mels)

    # Save the training data
    logger.info(f'Saving training data...')
    trained_data_array = np.empty(1, dtype=object)
    trained_data_array[0] = trained_data

    storage_path = settings.get('mel_ceps_storage_location', os.getcwd())

    # Save the array as numpy file; needs pickle, because object is a non-primitive (tuple)
    np.save(os.path.join(storage_path, f"training_data_torch_{size}.npy"), trained_data_array, allow_pickle=True)
    return load_trainingdata(settings.get('size', 'bees_1'))


def load_trainingdata(size:str ='bees_1') -> tuple:
    """
    This function loads the training data from the file. The file is created in the create_trainingdata function. The correct file is found by the specified size.
    Args:
        size: str which one of the dataset is used ("bees_1")
    Returns:
        contents of the file as a tuple
    """

    data = np.load(f"../files/training_data_torch_{size}.npy", allow_pickle=True)[0]
    return data
