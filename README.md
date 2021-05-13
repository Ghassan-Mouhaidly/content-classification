# content-classification

## Abstract
A fully modular pipeline for multi-output content classification - EECE 690 - AUB -
The model is built using keras, with yacs for configurations. A data generator is also included.
Training and visualisation is made with jupyter notebooks for simplicity.

## Getting started
To get the package running, simply make sure to install the python libraries indicated in requirements.txt (a virtual environment would be very beneficial here)

## Training for a custom application
To train for a custom application, simply process your dataset into a dataframe, which has to include the image path and labels for each class/each output.
As for the model architecture, specify the backbone (feature extractor) name you wish to use, number of outputs and their properties (loss, metric, number of classes, activation). All can be easily specified in the configuration file as dictionaries, which will automatically build and compile the desired model.
Note: the data generator is still not modular as I am limited on time - coming soon !

## Coding standards
This project follows the Python Style Guide (PSG) provided by Google. The PSG is available at http://google.github.io/styleguide/pyguide.html

## Directory structure
The directory structure of this project is as follows:

```
├── README.md
│
├── data_processing
│   ├── process_raw_data.py
│   ├── data_generator.py
│   ├── processed_data.pkl
│   └── balanced_data.pkl 
│
├── cfgs
│   ├── content_classification_v1.py
│   └── assets
│       └── default_configs.py
│
├──model
│  └──architecture
│
├──logs
│
├──checkpoints
│
├──model.png
│
├──requirements.txt
│
├──train.ipynb
│
└──LICENSE
```
