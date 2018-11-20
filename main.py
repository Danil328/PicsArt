import numpy as np
from skimage.io import imread, imshow
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from model import Nest_Net
from losses import  dice_coef_loss_bce, dice_coef, hard_dice_coef, binary_crossentropy
from my_tools import rle_encoding, rle_decode, rle_encode, visualize
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import load_model
from my_tools import dice
import tensorflow as tf
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    Blur,
    ElasticTransform,
    GridDistortion,
    ShiftScaleRotate,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    JpegCompression
)

# path = 'PicsArt/data/'
path = '/media/danil/Data/Datasets/PicsArt/data/'
BATCH = 12
supervision = False

import gc
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

def load_train_data(path):
    print('===LOAD DATA===')
    train_images = os.listdir(os.path.join(path, 'train'))
    train_images_list = [imread(os.path.join(path, 'train', img)) for img in train_images]
    train_mask_list = [imread(os.path.join(path, 'train_mask', img.split('.')[0]+'.png')) for img in train_images]
    return np.array(train_images_list), np.expand_dims(np.array(train_mask_list),-1)

def load_test_data(path):
    print('===LOAD TEST DATA===')
    image_names = os.listdir(os.path.join(path, 'test'))
    test_images_list = [imread(os.path.join(path, 'test', img)) for img in image_names]
    return image_names, np.array(test_images_list)

def make_predict(model):
    image_names, test_images_array = load_test_data(path)
    test_images_array = test_images_array / 255.
    print('===PREDICT===')
    if supervision:
        predict_mask = model.predict(test_images_array, batch_size=1)[-1]
    else:
        predict_mask = model.predict(test_images_array, batch_size=1)
        predict_mask_tta = model.predict(np.fliplr(test_images_array), batch_size=1)
        predict_mask = (predict_mask + predict_mask_tta) / 2.
    return test_images_array, predict_mask, image_names

def evaluate(model, X_val, y_val, threshold=0.5):
    pred = model.predict(X_val)
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    dice_list = [dice(y_val[i], pred[i]) for i in range(X_val.shape[0])]
    print("Evaluate Dice coefficient: {}".format(np.mean(dice_list)))

def create_submission(image_names, predicted_mask, threshold=0.5):
    print('===CREATE SUBMISSION===')
    predicted_mask[predicted_mask >= threshold] = 1
    predicted_mask[predicted_mask < threshold] = 0
    rle_mask = [rle_encode(x) for x in predicted_mask]
    sub = pd.DataFrame()
    sub['image'] = image_names
    sub['image'] = sub['image'].map(lambda x: x.split('.')[0])
    sub['rle_mask'] = rle_mask
    sub.to_csv('submission/submission.csv', index=False)

def create_train_image_generator(X_train, y_train, batch = BATCH, supervision=False):
    aug = Compose([
        VerticalFlip(p=0.25),
        HorizontalFlip(p=0.25),
        OneOf([
            ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=50),
            GridDistortion(p=0.5),
            ShiftScaleRotate(p=0.5),
        ], p=0.5),
        CLAHE(p=0.5),
        RandomContrast(p=0.5),
        RandomBrightness(p=0.5),
        RandomGamma(p=0.5),
        JpegCompression(p=0.5),
        Blur(p=0.5)
        ])

    while True:
        image_rgb = []
        image_mask = []
        k = np.random.randint(0, 100)
        np.random.seed(k)
        np.random.shuffle(X_train)
        np.random.seed(k)
        np.random.shuffle(y_train)
        for i in range(X_train.shape[0]):
            augmented = aug(image=X_train[i], mask=y_train[i, ..., 0])
            image_rgb += [augmented['image']]
            image_mask += [augmented['mask']]
            if len(image_rgb) == batch:
                if supervision:
                    m = np.expand_dims(np.stack(image_mask, 0),-1)/255.
                    yield np.stack(image_rgb,0)/255., {'output_1': m,
                                                        'output_2': m,
                                                        'output_3': m,
                                                        'output_4': m}
                else:
                    yield np.stack(image_rgb,0)/255., np.expand_dims(np.stack(image_mask, 0),-1)/255.

                image_rgb, image_mask = [], []

    return train_generator

def create_callbaks(model_name='unet++.h5'):
    checkpoint = ModelCheckpoint('weights/' + model_name, monitor='val_dice_coef', mode='max', save_best_only=True, verbose=1)
    return [checkpoint]

def train_model(train_generator):
    if supervision:
        loss = {'output_1': dice_coef_loss_bce,
                'output_2': dice_coef_loss_bce,
                'output_3': dice_coef_loss_bce,
                'output_4': dice_coef_loss_bce}

        val_data = (X_val, {'output_1': y_val,
                                  'output_2': y_val,
                                  'output_3': y_val,
                                  'output_4': y_val})
        path_to_pretrained_model = 'weights/unet_with_car_data_supervision.h5'
        callback_name = 'unet++supervision.h5'
    else:
        loss = dice_coef_loss_bce
        val_data = (X_val, y_val)
        path_to_pretrained_model = 'weights/unet_with_car_data.h5'
        callback_name = 'unet++.h5'

    callbacks = create_callbaks(callback_name)
    model = Nest_Net(320, 240, 3)
    # model = load_model(path_to_pretrained_model, compile=False)
    model.compile(optimizer=Adam(1e-3, decay=1e-5), loss=loss, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])

    print('===FIT MODEL===')
    model.fit_generator(train_generator,
                        steps_per_epoch = X_train.shape[0]/BATCH,
                        epochs=20,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_data,
                        initial_epoch=0)

    model = load_model('weights/' + callback_name, compile=False)
    model.compile(optimizer=Adam(0.0005, decay=1e-5), loss=loss, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])

    model.fit_generator(train_generator,
                        steps_per_epoch = X_train.shape[0]/BATCH,
                        epochs=30,
                        verbose=2,
                        callbacks=callbacks,
                        validation_data=val_data,
                        initial_epoch=20)

    model = load_model('weights/' + callback_name, compile=False)
    model.compile(optimizer=Adam(1e-4, decay=1e-5), loss=loss, metrics=[dice_coef, hard_dice_coef, binary_crossentropy])

    if supervision:
        model.fit(X_train / 255., {'output_1': y_train / 255., 'output_2': y_train / 255., 'output_3': y_train / 255., 'output_4': y_train / 255.},
                  batch_size=BATCH, epochs=50, verbose=2, callbacks=callbacks,
                  validation_data=val_data, initial_epoch=30)
    else:
        model.fit(X_train/255., y_train/255., batch_size=BATCH, epochs=40, verbose=2, callbacks=callbacks,
                  validation_data=val_data, initial_epoch=30)

    return model

if __name__ == '__main__':
    train_images, train_mask = load_train_data(path)
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_mask, test_size = 0.15, random_state = 17)
    X_val = X_val / 255.
    y_val = y_val / 255.

    train_generator = create_train_image_generator(X_train, y_train, supervision=supervision)
    # x,y = next(train_generator)
    # plt.figure()
    # imshow(x[0])
    # plt.show(block=False)
    # plt.figure()
    # imshow(y[0,...,0])
    # plt.show(block=False)

    model = train_model(train_generator)
    model = load_model('weights/unet++.h5', compile = False)

    evaluate(model, X_val, y_val, 0.5)

    test_image_array, predicted_mask, test_image_names = make_predict(model)
    create_submission(test_image_names, predicted_mask, threshold=0.5)

    # plt.figure()
    # imshow(test_image_array[0])
    # plt.show()
    # plt.figure()
    # imshow(predicted_mask[0,...,0])
    # plt.show()
