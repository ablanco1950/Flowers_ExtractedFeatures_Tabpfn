# https://universe.roboflow.com/un6oj918voact7ae2ql63t5q4ft1/flowers_classification/dataset/6
# 

# 1. IMPORT LIBRARIES

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
#from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# 2. Extract Data and EDA

DATASET_DIR = "Flowers_Classification.v6i.folder/train"
DATASET_DIR_UNSEEN_DATA = "Flowers_Classification.v6i.folder/test"
IMG_SIZE=224

classes = os.listdir(DATASET_DIR)
print("Number of classes:", len(classes))

print(classes)

class_counts = {cls: len(os.listdir(os.path.join(DATASET_DIR, cls))) for cls in classes}

plt.figure(figsize=(10,4))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.xticks(rotation=90)
plt.title("Class Distribution (Imbalanced Dataset)")
plt.show()

def show_sample_images():
    plt.figure(figsize=(12,6))
    for i, cls in enumerate(np.random.choice(classes, 6)):
        img_path = os.path.join(DATASET_DIR, cls, 
                                np.random.choice(os.listdir(os.path.join(DATASET_DIR, cls))))
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(2,3,i+1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis("off")
    plt.show()

show_sample_images()

# 3. Image Loading & Cleaning

def load_images(dataset_dir, img_size=IMG_SIZE):
    images = []
    labels = []
    
    for cls in os.listdir(dataset_dir):
        cls_path = os.path.join(dataset_dir, cls)
        ContCls=0
        for img_name in os.listdir(cls_path):
            ContCls=ContCls+1
            #if ContCls > 10: break # to limit number of images
            img_path = os.path.join(cls_path, img_name)
            try:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (img_size, img_size)) 
                images.append(img)
                labels.append(cls)
            except:
                pass
    
    return np.array(images), np.array(labels)

# 4. Encode Labels & Train/Test Split

X, y = load_images(DATASET_DIR, IMG_SIZE)
X_unseen, y_unseen = load_images(DATASET_DIR_UNSEEN_DATA, IMG_SIZE)

print("Images shape:", X.shape)

le = LabelEncoder() # encoder syring classes names into int values
y_encoded = le.fit_transform(y)
y_unseen= le.fit_transform(y_unseen)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# 5. Load EfficientNet as Feature Extractor

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)

feature_extractor.trainable = False

# 6. Feature Extraction

X_train_pre = preprocess_input(X_train)
X_test_pre = preprocess_input(X_test)
X_unseen_pre= preprocess_input(X_unseen)

train_features = feature_extractor.predict(X_train_pre, batch_size=32)
test_features = feature_extractor.predict(X_test_pre, batch_size=32)
unseen_features=feature_extractor.predict(X_unseen_pre, batch_size=32)

print("Feature vector shape:", train_features.shape)

# 7. Machine Learning Classifier (SVM)

from tabpfn import TabPFNClassifier
from tabpfn import save_fitted_tabpfn_model

tabpfn = TabPFNClassifier(device='cpu',  # or 'cuda' if you have a GPU
        # N_ensemble_configurations=8,
        ignore_pretraining_limits=True)   # No hyperparameters. That's the point.
        #progress_bar=True,
        #verbose=1)

#tabpfn.fit(train_features.values, y_train)
tabpfn.fit(train_features, y_train)

#tab_pred = tabpfn.predict(test_features.values)
tab_pred = tabpfn.predict(test_features)
#tab_prob = tabpfn.predict_proba(test_features.values)[:, 1]
tab_prob = tabpfn.predict_proba(test_features)[:, 1]

print(tab_pred)
print(tab_prob)
print(y_test)


print("Accuracy train:", accuracy_score(y_test, tab_pred))
print(classification_report(y_test, tab_pred, target_names=le.classes_))


tab_pred_unseen = tabpfn.predict(unseen_features)

tab_prob_unseen = tabpfn.predict_proba(unseen_features)[:, 1]

print(tab_pred_unseen)
print(tab_prob_unseen)
print(y_unseen)

tab_pred_unseen = tabpfn.predict(unseen_features)

tab_prob_unseen= tabpfn.predict_proba(unseen_features)[:, 1]

print("")
print("")
print("")

print("Accuracy with unseen data:", accuracy_score(y_unseen, tab_pred_unseen))
print(classification_report(y_unseen, tab_pred_unseen, target_names=le.classes_))

#import joblib

#joblib.dump(tabpfn, "tabpfn_flowers.pkl")

save_fitted_tabpfn_model(tabpfn, "tabpfn_flowers.tabpfn_fit") # not useful










