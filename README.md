# Flowers_ExtractedFeatures_Tabpfn
This involves testing the HuggingFace Tabpfn model with the Roboflow flower dataset (https://universe.roboflow.com/un6oj918voact7ae2ql63t5q4ft1/flowers_classification/dataset/6). It is performed in two phases: first, the image features are extracted using a CNN with EfficientNetB0 as the pretrained model, and then the TabPFNClassifier is applied.

Installation:

Download the project to disk.

Download the dataset (you will need a Roboflow account, which is free): https://universe.roboflow.com/un6oj918voact7ae2ql63t5q4ft1/flowers_classification/dataset/6
In your project folder, it should appear as a folder named Flowers_Classification.v6i.folder with the subfolders: train, valid, and test.

The following modules must been installed.

python pip-script.py install numpy

python pip-script.py install packaging

python pip-script.py install Pillow

python pip-script.py install pyparsing

python pip-script.py install cycler

python pip-script.py install python-dateutil

python pip-script.py install kiwisolver

python pip-script.py install importlib_resources

python pip-script.py install pandas

python pip-script.py install scikit-learn

python pip-script.py install tensorflow

python pip-script.py install opencv-contrib-python

python pip-script.py install tabpfn


You need a HuggingFace account and a token to download a trial version of the TabPFNClassifier model: Prior-Labs/tabpfn_2_6

Prior-Labs/tabpfn_2_6 · Hugging Face
https://huggingface.co/Prior-Labs/tabpfn_2_6

And create a token

https://huggingface.co/settings/tokens

Once the token is created, in the permissions column, under the FINEGRAINED button, expand the button marked with three vertical dots and grant the token all possible permissions.

In the development environment, type:

hf auth login

This will display the following message:

_| _| _| _| _|_|_| _|_|_| _|_|_| _| _| _|_|_| _|_|_|_| _|_| _|_|_| _|_|_|_|
_| _| _| _| _| _| _| _|_| _| _| _| _| _| _|
_|_|_|_| _| _| _| _|_| _| _| _| _| _| _|_| _|_|_| _|_|_|_| _| _|_|_|
_| _| _| _| _| _| _| _| _| _| _| _| _| _| _| _| _|
_| _| _|_| _|_|_| _|_|_| _|_|_| _| _| _|_|_| _| _| _| _|_|_| _|_|_|_| 


To log in, `huggingface_hub` requires a token generated from https://huggingface.co/settings/tokens .
Token can be pasted using 'Right-Click'.
Enter your token (input will not be visible):

Once the token is entered, run the program that trains the model and performs an evaluation with unseen data

python Train_Flowers_FeaturesExtracted_Tabpfn.py

Resulting in a report that shows an accuracy of 0.945.

Accuracy with unseen data: 0.945054945054945

          Precision recall f1-score support

Daisy      0.92     0.95    0.94     77

Dandelion  0.96     0.94    0.95    105


Accuracy                    0.95    182

Macro avg    0.94   0.95    0.94    182

Weighted avg 0.95   0.95    0.95    182


For subsequent runs, the token is stored on the machine, and the session can be closed using the command:

hf auth logout

CONCLUSIONS:

- The model performs accurately and quickly, and can be run on a personal computer with 16GB of RAM without hindering other tasks. Even with very few training records, results with a precision of 0.8 are obtained.

- Special care has been taken to isolate unseen data, and the accuracy train and test are practically the same, so the model lacks overfitting.

- On the other hand, slow inference and the inability to save and retrieve models from disk were encountered. It should be noted that a trial version was downloaded. Testing with more complex datasets will be attempted.

References and citations:

https://universe.roboflow.com/un6oj918voact7ae2ql63t5q4ft1/flowers_classification/dataset/6

https://huggingface.co/Prior-Labs/tabpfn_2_6

https://huggingface.co/settings/tokens

https://www.kaggle.com/code/saadmohamed99/plant-disease-classification
Basically, this project model is used, changing SVM to Tabpfn.

https://medium.com/@koshurai/tabpfn-2-5-the-foundation-model-that-beat-xgboost-without-a-single-line-of-tuning-f02a515e169e

@misc{TabPFN-2.5,\ 
title={TabPFN-2.5},\ 
author={Léo Grinsztajn and Klemens Flöge and Oscar Key and Felix Birkel and Brendan Roof and Phil Jund and Benjamin Jäger and Adrian Hayler and Dominik Safaric and Simone Alessi, Felix Jablonski and Mihir Manium and Rosen Yu and Anurag Garg and Jake Robertson and Shi Bin (Liam) Hoo and Vladyslav Moroshan and Magnus Bühler and Lennart Purucker and Clara Cornu and Lilly Charlotte Wehrhahn and Alessandro Bonetto and Sauraj Gambhir and Noah Hollmann and Frank Hutter},\ 
year={2025}\
}
