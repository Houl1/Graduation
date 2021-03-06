# How to load and use weights from a checkpoint
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import os
import os.path
import glob

from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils, plot_model

np.random.seed(1)

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import preprocess_input

from keras.models import Sequential, Model
from keras.layers import Input, Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
imagedir = "new-14-family"


cur_dir = os.getcwd()
os.chdir(imagedir)  # the parent folder with sub-folders

# Get number of samples per family
list_fams = sorted(os.listdir(os.getcwd()), key=str.lower)  # vector of strings with family names
no_imgs = []  # No. of samples per family
for i in range(len(list_fams)):
    os.chdir(list_fams[i])
    len1 = len(glob.glob('*.png'))  # assuming the images are stored as 'png'
    no_imgs.append(len1)
    os.chdir('..')
num_samples = np.sum(no_imgs)  # total number of all samples

# Compute the labels
y = np.zeros(num_samples)
pos = 0
label = 0
for i in no_imgs:
    print ("Label:%2d\tFamily: %15s\tNumber of images: %d" % (label, list_fams[label], i))
    for j in range(i):
        y[pos] = label
        pos += 1
    label += 1
num_classes = label

# Compute the features
width, height,channels = (224, 224, 3)
X = np.zeros((num_samples, width, height, channels))
cnt = 0
list_paths = [] # List of image paths
print("Processing images ...")

All_imgs = []

for i in range(len(list_fams)):
    for img_file in glob.glob(list_fams[i]+'/*.png'):

        All_imgs.append(img_file)  # 把所有图片名放入All_imgs数组中

        #print("[%d] Processing image: %s" % (cnt, img_file))
        list_paths.append(os.path.join(os.getcwd(),img_file))
        img = image.load_img(img_file, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        X[cnt] = x
        cnt += 1
print("Images processed: %d" %(cnt))

os.chdir(cur_dir)

# Encoding classes (y) into integers (y_encoded) and then generating one-hot-encoding (Y)
encoder = LabelEncoder()
encoder.fit(y)
y_encoded = encoder.transform(y)
Y = np_utils.to_categorical(y_encoded)


# create model

input_shape = (224, 224, 3)
input_tensor = Input(shape=input_shape)

# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# Classification block
x = Flatten(name='flatten')(x)  # input_shape=(7,7,512)
x = Dense(units=2048, activation='relu', name='FC-1')(x)
x = Dense(units=2048, activation='relu', name='FC-2')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

vgg_model = Model(inputs=input_tensor, outputs=predictions)  # 网络链接要注意
#vgg_model.summary()

# load weights
vgg_model.load_weights("/home/mayixuan/mal-visual/checkpoint/New/New-CNN-300epochs-transfer.h5")
# Compile model (required to make predictions)
vgg_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")


# estimate accuracy on whole dataset using loaded weights
y_prob = vgg_model.predict(X, verbose=1)  # Testing
y_pred = np.argmax(y_prob, axis=1)



print("acurracy: %.4f" % (accuracy_score(y, y_pred)))
print(classification_report(y, y_pred, digits=4))

# 将图片名、ground truth label、predict label放入excel表格中（分别为1/2/3列）
data = pd.DataFrame({'name': All_imgs,
                     'label': y,
                     'predict label': y_pred})

writer = pd.ExcelWriter('predict-result.xlsx')
data.to_excel(writer, 'page_1')
writer.save()

writer.close()
print('DataFrame is written successfully to the Excel File.')