
## Install the requirements

If using Anaconda and Windows, the Conda environment file (isbi2013_env.yml) used for development (with all the required dependencies installed) is provided. Import the environment using:

> conda env create -f isbi2013_env.yml

Otherwise, the pip requirements file is provided as well. The requirements may be installed using (need git to be installed first):

> pip install -r requirements.txt

Note: The requirements file specifies Tensorflow without GPU support. If the code is to be run on a GPU enabled environment, then Tensorflow GPU must be installed separately:

> pip install --upgrade tensorflow-gpu

## Get the data

The data can be obtained [here](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) from the competition database.

## Using the Code

## Load the data

The data can be loaded, preprocessed, and prepared using:

> python main.py -l

The paths to the DICOM and NRRD image data need to be edited in the file. The code assumes the files to be present and saved in the "data" directory under three sub-folders named: train, validate, and test. The DICOM files should be organized in patient-wise sub-directories and the NRRD files should be present at the level of the sub-directories.

## Train the network

Once the data has been prepared, the network can be trained using:

> python main.py -t

The paths to the preprocessed data and the choice of the loss function and network model have to be edited in the file.

## Test the network and compute all the evaluation metrics

After the network has been trained, the saved weights file can be used to make predictions on the test set and evaluate the performance. Pre-trained weights may also be used, which are available [here](https://drive.google.com/open?id=1qO2BaMGzC3n9OsZ--1kGYGeDUNXmhxUL)

The path to the weight file which is to be used for evaluating the network has to be edited in the file. The network model name may also need to be changed (the default is standard U-Net with image size 224 x 224).

> python main.py -e
