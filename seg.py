#Brain Tumor Segmentation with U-net model
#Import required packages

from glob import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from skimage.color import rgb2gray
from tensorflow.keras import Input
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Activation, BatchNormalization, Dropout, Lambda, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.model_selection import train_test_split

# Load mask and brain image into array

img_width = 256
img_height = 256
data_path = "/home/pl2313/dataset/kaggle_3m"


img = []
masks = glob(data_path+'/*/*_mask*')
for file in masks:
  img.append(file.replace('_mask',''))  

print(masks)

#Separate each dataset into training set, test set and validation set, store them into Dataframe object

data = pd.DataFrame(data={'brain': img, 'mask': masks})
train_dataset, test_dataset = train_test_split(data, test_size = 0.1)
train_dataset, validation_dataset = train_test_split(train_dataset,test_size = 0.2)

#Data normalize function

def normalize_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

#Training Data generator. Generate image and mask

def data_generator(data_frame, batch_size, aug_dict,image_color_mode="rgb",
                   mask_color_mode="grayscale", image_save_prefix="image",
                  mask_save_prefix="mask",save_to_dir=None, target_size=(256,256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "brain",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    for (img, mask) in train_gen:
        img, mask = normalize_data(img, mask)
        yield (img,mask)

#Define loss function

SMOOTH = 10

def dice_coef(target, pred):
    y_targetf=K.flatten(target)
    y_predf=K.flatten(pred)
    And=K.sum(y_targetf* y_predf)
    return((2* And + SMOOTH) / (K.sum(y_targetf) + K.sum(y_predf) + SMOOTH))

def dice_coef_loss(target, pred):
    return -dice_coef(target, pred)

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=K.flatten(y_true)
    y_predf=K.flatten(y_pred)

    return - iou(y_true, y_pred)    


#Unet model using keras

def unet(input_size=(img_width, img_height, 3)):
    inputs = Input(input_size)
    
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = Activation('relu')(conv1)
    conv1 = Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = BatchNormalization(axis=3)(conv1)
    bn1 = Activation('relu')(bn1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = Activation('relu')(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = BatchNormalization(axis=3)(conv2)
    bn2 = Activation('relu')(bn2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = Activation('relu')(conv3)
    conv3 = Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = BatchNormalization(axis=3)(conv3)
    bn3 = Activation('relu')(bn3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = Activation('relu')(conv4)
    conv4 = Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = BatchNormalization(axis=3)(conv4)
    bn4 = Activation('relu')(bn4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = Activation('relu')(conv5)
    conv5 = Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = BatchNormalization(axis=3)(conv5)
    bn5 = Activation('relu')(bn5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = Activation('relu')(conv6)
    conv6 = Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = BatchNormalization(axis=3)(conv6)
    bn6 = Activation('relu')(bn6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = Activation('relu')(conv7)
    conv7 = Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = BatchNormalization(axis=3)(conv7)
    bn7 = Activation('relu')(bn7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = Activation('relu')(conv8)
    conv8 = Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = BatchNormalization(axis=3)(conv8)
    bn8 = Activation('relu')(bn8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = Activation('relu')(conv9)
    conv9 = Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = BatchNormalization(axis=3)(conv9)
    bn9 = Activation('relu')(bn9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return Model(inputs=[inputs], outputs=[conv10])

unet_model = unet()
unet_model.summary()

EPOCHS = 40
BATCH_SIZE = 8
learning_rate = 0.001

decay = learning_rate / EPOCHS
# optimizer
optimizer = Adam(lr=learning_rate, decay=decay)

#Data preparation

train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')

train_data = data_generator(train_dataset, BATCH_SIZE, train_generator_args, target_size=(img_height, img_width))

val_data =  data_generator(validation_dataset, BATCH_SIZE,dict(),target_size=(img_height, img_width))

callbacks = [ModelCheckpoint('unet_brain_mri_seg.hdf5', verbose=1, save_best_only=True)]

#Train the model using dataset

unet_model.compile(optimizer=optimizer, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])
result = unet_model.fit(train_data,
                    steps_per_epoch=len(train_dataset) / BATCH_SIZE, 
                    epochs=EPOCHS, 
                    callbacks=callbacks,
                    validation_data = val_data,
                    validation_steps=len(validation_dataset) / BATCH_SIZE)

a = result.history

list_traindice = a['dice_coef']
list_testdice = a['val_dice_coef']

list_trainjaccard = a['iou']
list_testjaccard = a['val_iou']

list_trainloss = a['loss']
list_testloss = a['val_loss']

data = {
    'list_traindice' : a['dice_coef'],
    'list_testdice' : a['val_dice_coef'],

    'list_trainjaccard' : a['iou'],
    'list_testjaccard' : a['val_iou'],

    'list_trainloss' : a['loss'],
    'list_testloss' : a['val_loss'],
}

with open('data.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# plt.figure(1)
# plt.plot(list_testloss, 'b-')
# plt.plot(list_trainloss,'r-')
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.title('loss graph', fontsize = 15)
# plt.figure(2)
# plt.plot(list_traindice, 'r-')
# plt.plot(list_testdice, 'b-')
# plt.xlabel('iteration')
# plt.ylabel('accuracy')
# plt.title('accuracy graph', fontsize = 15)
# plt.show()

model = load_model('unet_brain_mri_seg.hdf5', custom_objects={'dice_coef_loss': dice_coef_loss, 'iou': iou, 'dice_coef': dice_coef})

test_gen = data_generator(test_dataset, BATCH_SIZE,
                                dict(),
                                target_size=(img_height, img_width))
results = model.evaluate(test_gen, steps=len(test_dataset) / BATCH_SIZE)
print("Test lost: ",results[0])
print("Test IOU: ",results[1])
print("Test Dice Coefficent: ",results[2])

# for i in range(30):
    # index=np.random.randint(1,len(test_dataset.index))
    # img = cv2.imread(test_dataset['filename'].iloc[index])
    # img = cv2.resize(img ,(img_height, img_width))
    # img = img / 255
    # img = img[np.newaxis, :, :, :]
    # pred=model.predict(img)

    # plt.figure(figsize=(12,12))
    # plt.subplot(1,3,1)
    # plt.imshow(np.squeeze(img))
    # plt.title('Original Image')
    # plt.subplot(1,3,2)
    # plt.imshow(np.squeeze(cv2.imread(test_dataset['mask'].iloc[index])))
    # plt.title('Original Mask')
    # plt.subplot(1,3,3)
    # plt.imshow(np.squeeze(pred) > .5)
    # plt.title('Prediction')
    # plt.show()