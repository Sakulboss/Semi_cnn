# Overview of functions
  "project_dir_name": "Semi10",

## Logging settings
Log Levels:
- 0: DEBUG
- 1: INFO
- 2: WARNING
- 3: ERROR
- 4: CRITICAL
- 5: None

log to console: stdout
lot to file: save log to log_file
log_file: path to log file

    "log_level": 0,
    "log_to_console": true,
    "log_to_file": false,
    "log_file": "Semi_10.log",

log_format: standard, wouldn't change it

    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

## Training settings
### General settings
Raw training data storage location

    "training_files_storage_location": "D:\\Daten_Semi10\\26-27_April",

Sorted training data, path to main folder containing subfolders for each class

    "sorted_files_storage_location": "D:\\Daten_Semi10",

Place to save complete training data

    "mel_specs_storage_location": "D:\\Daten_Semi10",

Name of dataset

    "size": "bees_1",

Create new folder structure from unsorted data, use-case specific, using function in data_file_prep with size name

    "create_new_source": false,

Create new mel cepstrograms and features and save them

    "create_new": false,

File extensions used for training data, like wav or flac

    "training_file_extensions": "flac",


### feature extraction settings
Number of segments extracted from each file

    "segments_per_spectrogram": 10,

Length of each spectrogram segment

    "segment_length_frames": "100",
Maximum number of files per class to use for training
!Use only for short testing, otherwise set to 0, if 0, all files are used

    "max_file_num" : 100

    
    

## Model settings
### Data splitting
Number of data points per batch

    "batch_size": 60,
Ratio of swarming to non-swarming data points

    "swarm_event_ratio": 0.5,
Ratio of training to testing data points

    "test_size": 0.2,

### training settings 

Minimum number of epochs to counter early-stopping (against overfitting)

    "min_epoch": 5,

Maximum number of epochs in case something went wrong 

    "max_epochs": 100,
Learning rate for the optimizer, idk should be a good value

    "learning_rate": 0.01,
    

### multi-training settings


    "model_structure_file": "C:\\Users\\sdose\\Documents\\GitHub\\Semi10\\CNN_clean\\Neural_Network",
    
    "use_server": false,
    "train_once": true,
    "model_text": "l; conv2d; (1, 16); (3, 3); 1; (1, 1);; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (16, 48); (3, 3); 1; (1, 1);; p; avgpool; (3, 3); 1; (1, 1);; l; conv2d; (48, 48); (3, 3); 1; (1, 1);; p; maxpool; (3, 3); 1; (1, 1);; v: extra;; l; linear; (307200, 10);; l; linear; (10, 10);; l; linear; (10, 2);",
    "dropbox": ""
