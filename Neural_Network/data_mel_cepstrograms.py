#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import librosa.feature as lf
import librosa
from tqdm import tqdm
import matplotlib.pyplot as pl

logger = logging.getLogger("CNN_Training.MELS.cepstrogram")


def mel_cepstrogram_from_file(fn_wav_name, n_fft=1024, hop_length=441, fss = 48000, n_mels=64, stereo:bool=True) :
    """
    Compute mel cepstrogram from audio file with librosa.feature.melspectrogram()
    Args:
        fn_wav_name:  str   Audio file name
        n_fft:        int   FFT size
        hop_length:   int   Hop size in samples
        fss:          float Sample rate in Hz
        n_mels:       int   Number of mel-bands
        stereo:       bool  If False, convert to mono
    Returns:
        np.ndarray: Mel cepstrogram
    """

    try:
        x_new, fss = librosa.load(fn_wav_name, sr=fss, mono=not stereo)
    except Exception as e:
        logger.error(e)
        return None

    if stereo:
        x_new = np.append(x_new[0], x_new[1])

    # normalize to the audio file to a maximum absolute value of 1
    if np.max(np.abs(x_new)) > 0:
        x_new = x_new / np.max(np.abs(x_new))

    x_new = lf.melspectrogram(y=x_new,
                                sr=fss,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                n_mels=n_mels,
                                fmin=0.0,
                                fmax=fss / 2,
                                power=1.0,
                                htk=True,
                                norm=None,
                                pad_mode="reflect"
                              )

    # apply dB normalization
    x_new = librosa.amplitude_to_db(x_new)

    return x_new


def mel_specs(data: tuple, setting) -> tuple:
    """
    Create mel cepstrogram from audio files and splits them into segments to generate more training data.
    Args:
        data:       tuple of data from previous step
        setting:      dict  with main settings like the type of dataset and injection of other labeled data.
    Returns:
        segment_list:        list of mel cepstrogram segments
        segment_class_id:    list of class ids of the segments
        data[2]:             list of class names
        data[5]:             int number of classes
    """

    # Initialize variables
    fn_wav_list = data[0]
    class_id = data[1]
    size = setting.get('size', 'bees_1')
    segments_per_spectrogram = int(setting.get('segments_per_spectrogram', 10))
    segment_length_frames = int(setting.get('segment_length_frames', 100))

    all_mel_specs = []
    segment_list = []
    segment_class_id = []
    error_files = []

    # Create mel cepstrogram
    for count in tqdm(range(len(fn_wav_list)), desc='Mel-Cepstrogram'):
        mel_spec = mel_cepstrogram_from_file(fn_wav_list[count], stereo=(size == 'bees_1'))
        if mel_spec is None or count != 0 and mel_spec.shape != all_mel_specs[-1].shape:
            error_files.append(fn_wav_list[count])
        else:
            all_mel_specs.append(mel_spec)
    # check for faulty files (can very rarely happen, when recording in flac)
    if not error_files == []:
        error_files = [(str(i) + "\n") for i in error_files]
        logger.warning(f'Wrong file size or faulty file(s), please remove the file(s): {error_files}')

    # combine the mel specs into one np.array
    all_mel_specs = np.stack(all_mel_specs, axis=0)
    max_segment_start_offset = all_mel_specs.shape[-1] - segment_length_frames

    # Create segments from the mel cepstrogram with random start points
    for i in range(len(all_mel_specs)):
        # create n segments from each cepstrogram
        for s in range(segments_per_spectrogram):
            # Use as many parts of the mel cepstrogram as possible, if more are needed, select a random start point
            if (s+1) * segment_length_frames <= all_mel_specs.shape[-1]:
                segment_list.append(all_mel_specs[i, :, s*segment_length_frames:(s+1)*segment_length_frames])
            else:
                segment_start_frames = int(np.random.rand(1).item() * max_segment_start_offset)
                segment_list.append(all_mel_specs[i, :, segment_start_frames:segment_start_frames + segment_length_frames])
            segment_class_id.append(class_id[i])

    # conversion from the list of cepstrogram segments into a np.array for faster processing
    segment_list = np.array(segment_list)
    segment_class_id = np.array(segment_class_id)

    pl.figure(figsize=(10, 8))
    pl.imshow(all_mel_specs[2, :, :], origin="lower", aspect="auto", interpolation="None")
    pl.xticks([], [])
    pl.yticks([], [])
    pl.tight_layout()
    pl.show()

    return segment_list, segment_class_id, data[2], data[5]


