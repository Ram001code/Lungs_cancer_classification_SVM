
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm, metrics, datasets
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage
from skimage.io import imread
from skimage.transform import resize
import os

from tqdm import tqdm


data_dir= r'D:\Albot\Medical Imaging-Cancer\code\Jupyter notebook\datasets'


print("Directories: ", os.listdir(data_dir))

print("\n No. of images in Bengin cases:", len(os.listdir(data_dir+'\\Bengin cases')))
print("\n No. of images in Malignant cases:", len(os.listdir(data_dir+'\\Malignant cases')))
print("\n No. of images in Normal cases:", len(os.listdir(data_dir+'\\Normal cases')))


def load_image_files(container_path, dimension=(64, 64)):
    image_dir = Path(container_path)
    #print("Data folder name:", image_dir)

    folder = os.listdir(image_dir)
    print("\n Each Categories Name:", folder)

    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    print("\n Name of folders:", folders)

    images = []
    flat_data = []
    target = []

    for i, direc in enumerate(folders):
        # print(direc)
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            # print(img)
            img_resized = resize(img, dimension, mode='reflect')
            flat_data.append(img_resized.flatten())
            images.append(img_resized)
            target.append(i)

    flat_data = np.array(flat_data)
    # print("flat_data:", flat_data)
    print("\n shape of flat_data:", flat_data.shape)

    target = np.array(target)
    print("\n shape of target:", target.shape)
    print("Target:", target)

    images = np.array(images)
    print("\n shape of images:", images.shape)
    # print("images:", images)

    return Bunch(data=flat_data, Target=target)

image_dataset= load_image_files('Jupyter notebook/datasets/')


print("\n Shape of flat data", image_dataset.data.shape)
print("\n Shape of Target data", image_dataset.Target.shape)

#Putting data into dataframe

df=pd.DataFrame(image_dataset.data) #dataframe

df['Target']=image_dataset.Target

X=df.iloc[:,:-1] #input data

y=df.iloc[:,-1] #output data

####Spliting into test and train


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=109)

print("\n Data splitted ")


# look at the distrubution of labels in the train set
print("y_train: ", len(y_train), )
print("y_test:", len(y_test))
print("X_train: ", len(X_train))
print("X_test:", len(X_test))


## Training data with parameter optimization


## Training data with parameter optimization

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC(probability= True)
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# calculate accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print('\n Model accuracy is: ', accuracy)

# save the model to disk
import pickle

filename = 'model_Without_Augm_SVM.sav'
pickle.dump(clf, open(filename, 'wb'))


