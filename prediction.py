import skimage
from skimage.io import imread
from skimage.transform import resize
import pandas as pd
import matplotlib.pyplot as plt


import pickle
filename = 'model_Without_Augm_SVM.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print("\n .........Model Loaded Successfully......... ")


### Model Evaluation

input_image = 'D:\Albot\Medical Imaging-Cancer\Datasets\lung cancer dataset\Test\Bengin cases\Bengin case (91).jpg'  #Bengin cases(class=0)
#input_image = 'D:\Albot\Medical Imaging-Cancer\Datasets\lung cancer dataset\Test\Malignant cases\Malignant case (458).jpg'  #Malignant cases(class=1)
#input_image= r'D:\Albot\Medical Imaging-Cancer\Datasets\lung cancer dataset\Test\Normal cases\Normal case (289).jpg'   #Normal cases(class=2)

img= imread(input_image)
print("\n shape of test image:", img.shape)
#plt.imshow(img)
#plt.show()

img_resize=resize(img,(64,64,3))  #resizing an image
print("\n shape after resizing the image:", img_resize.shape)

f =[img_resize.flatten()]   #flattening the image

Test_Prediction= loaded_model.predict(f)[0]

print("\n Predicted class: ",Test_Prediction )

if Test_Prediction ==0:
    print("\n Type of cancer: Bengin Lung Cancer")
elif Test_Prediction ==1:
    print("\n Type of cancer: Malignant Lung Cancer")
else:
    print("\n Type of cancer: Normal Lung(No Cancer)")


