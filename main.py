import numpy as np
from skimage.io import imread, imshow
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from model import Nest_Net
from losses import  dice_coef_loss_bce, dice_coef, hard_dice_coef, binary_crossentropy
from my_tools import rle_encoding, rle_decode, rle_encode, visualize
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam, SGD
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
BATCH = 16
supervision = True
CV = 0

MODE = 'stack'  # train predict stack

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

def split_train_data_to_cv():
    skf = KFold(n_splits=5, shuffle=True, random_state=17)
    train_images = os.listdir(os.path.join(path, 'train'))
    train_images = np.array(train_images)
    X_train_cv = skf.split(train_images)

    index_train_dict = {}
    index_test_dict = {}
    for i in range(5):
        ind = next(X_train_cv)
        index_train_dict['split_{}'.format(i)] = ind[0]
        index_test_dict['split_{}'.format(i)] = ind[1]
        np.save('cross_val/train_split_{}'.format(i), train_images[index_train_dict['split_{}'.format(i)]])
        np.save('cross_val/val_split_{}'.format(i), train_images[index_test_dict['split_{}'.format(i)]])

def read_cv_data(N=0):
    train_names = np.load('cross_val/train_split_{}.npy'.format(N))
    val_names = np.load('cross_val/val_split_{}.npy'.format(N))

    path = '/media/danil/Data/Datasets/PicsArt/data/'
    train_images_list = [imread(os.path.join(path, 'train', img)) for img in train_names]
    train_mask_list = [imread(os.path.join(path, 'train_mask', img.split('.')[0] + '.png')) for img in train_names]

    val_images_list = [imread(os.path.join(path, 'train', img)) for img in val_names]
    val_mask_list = [imread(os.path.join(path, 'train_mask', img.split('.')[0] + '.png')) for img in val_names]

    return np.array(train_images_list), np.expand_dims(np.array(train_mask_list), -1),\
           np.array(val_images_list), np.expand_dims(np.array(val_mask_list), -1)

def stack_prediction(N=5):
    files = []
    for i in range(N):
        files += [np.load('cross_val_predicts/predicted_mask_cv_{}.npy'.format(i)).item()]

    image_names = list(files[0].keys())
    union_predict = np.empty((len(image_names), 320, 240, 1), dtype=np.float32)
    for i, image_name in enumerate(image_names):
        temp = np.zeros((320, 240, 1))
        for k in range(N):
            temp += files[k][image_name]
        temp = temp/N
        union_predict[i] = temp

    create_submission(image_names, union_predict, threshold=0.59, stack=True)

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
        # predict_mask_tta = model.predict(np.fliplr(test_images_array), batch_size=1)[-1]
        # predict_mask_tta = np.fliplr(predict_mask_tta)
        # predict_mask = (predict_mask*0.7 + predict_mask_tta*0.3)
    else:
        predict_mask = model.predict(test_images_array, batch_size=1)
        # predict_mask_tta = model.predict(np.fliplr(test_images_array), batch_size=1)
        # predict_mask_tta = np.fliplr(predict_mask_tta)
        # predict_mask = (predict_mask*0.7 + predict_mask_tta*0.3)
    return test_images_array, predict_mask, image_names

def evaluate(model, X_val, y_val):
    pred = model.predict(X_val)[-1]
    score = []
    thresholds = np.arange(0.25, 0.75, 0.05)
    for threshold in thresholds:
        temp = pred.copy()
        temp[temp >= threshold] = 1
        temp[temp < threshold] = 0
        dice_list = [dice(y_val[i], temp[i]) for i in range(X_val.shape[0])]
        score.append(np.mean(dice_list))

    print("Evaluate Dice coefficient CV {}: Best_score = {} Best_threshold = {}".format(CV, np.max(score), thresholds[np.argmax(score)]))

def create_submission(image_names, predicted_mask, threshold=0.5, stack=False):
    print('===CREATE SUBMISSION===')
    predicted_mask[predicted_mask >= threshold] = 1
    predicted_mask[predicted_mask < threshold] = 0
    rle_mask = [rle_encode(x) for x in predicted_mask]
    sub = pd.DataFrame()
    sub['image'] = image_names
    sub['image'] = sub['image'].map(lambda x: x.split('.')[0])
    sub['rle_mask'] = rle_mask
    if stack:
        sub.to_csv('submission/supervision_submission_stack.csv', index=False)
    else:
        sub.to_csv('submission/supervision_submission_cv{}.csv'.format(CV), index=False)

def create_train_image_generator(X_train, y_train, batch = BATCH, supervision=False):
    aug = Compose([
        VerticalFlip(p=0.25),
        HorizontalFlip(p=0.5),
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

def create_callbaks(model_name='unet++.h5', monitor='val_dice_coef'):
    checkpoint = ModelCheckpoint('weights/' + model_name, monitor=monitor, mode='max', save_best_only=True, verbose=1)
    return [checkpoint]

def train_model(train_generator):
    if supervision:
        loss = {'output_1': binary_crossentropy,
                'output_2': binary_crossentropy,
                'output_3': binary_crossentropy,
                'output_4': dice_coef_loss_bce}

        val_data = (X_val, {'output_1': y_val,
                                  'output_2': y_val,
                                  'output_3': y_val,
                                  'output_4': y_val})
        path_to_pretrained_model = 'weights/real_unet++supervision.h5'
        callback_name = 'my_unet++supervision_cv{}'.format(CV) + '.h5'
        monitor = 'val_output_4_dice_coef'
        metric = {'output_4': [dice_coef, hard_dice_coef, binary_crossentropy]}
        loss_weight = [0.25, 0.25, 0.5, 1.]
    else:
        loss = dice_coef_loss_bce
        val_data = (X_val, y_val)
        path_to_pretrained_model = 'weights/unet++supervision.h5'
        callback_name = 'unet++_cv{}'.format(CV) + '.h5'
        monitor = 'val_dice_coef'
        metric = [dice_coef, hard_dice_coef, binary_crossentropy]
        loss_weight = [1.]

    callbacks = create_callbaks(callback_name, monitor)
    # model = Nest_Net(320, 240, 3, deep_supervision=supervision)
    model = load_model(path_to_pretrained_model, compile=False)
    model.compile(optimizer=Adam(1e-3, decay=1e-5), loss=loss, metrics=metric, loss_weights=loss_weight)

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
                  batch_size=BATCH, epochs=40, verbose=2, callbacks=callbacks, validation_data=val_data, initial_epoch=30)
    else:
        model.fit(X_train/255., y_train/255., batch_size=BATCH, epochs=40, verbose=2, callbacks=callbacks,
                  validation_data=val_data, initial_epoch=30)

    #SGD
    model = load_model('weights/' + callback_name, compile=False)
    model.compile(optimizer=SGD(1e-4), loss=loss,
                  metrics=[dice_coef, hard_dice_coef, binary_crossentropy])

    if supervision:
        model.fit(X_train / 255.,
                  {'output_1': y_train / 255., 'output_2': y_train / 255., 'output_3': y_train / 255., 'output_4': y_train / 255.},
                  batch_size=BATCH, epochs=50, verbose=2, callbacks=callbacks, validation_data=val_data, initial_epoch=40)
    else:
        model.fit(X_train / 255., y_train / 255., batch_size=BATCH, epochs=50, verbose=2, callbacks=callbacks,
                  validation_data=val_data, initial_epoch=40)


    return model

if __name__ == '__main__':
    # train_images, train_mask = load_train_data(path)
    # split_train_data_to_cv()

    if MODE == 'train':
        X_train, y_train, X_val, y_val = read_cv_data(CV)
        # X_train, X_val, y_train, y_val = train_test_split(train_images, train_mask, test_size = 0.15, random_state = 17)
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

        model = load_model('weights/my_unet++supervision_cv{}'.format(CV) + '.h5', compile=False)

        evaluate(model, X_val, y_val)

        test_image_array, predicted_mask, test_image_names = make_predict(model)
        predicts = dict(zip(test_image_names, predicted_mask))
        np.save('cross_val_predicts/predicted_mask_cv_{}'.format(CV), predicts)
        create_submission(test_image_names, predicted_mask.copy(), threshold=0.5)

    else:
        stack_prediction()

    # plt.figure()
    # imshow(test_image_array[0])
    # plt.show()
    # plt.figure()
    # imshow(predicted_mask[0,...,0])
    # plt.show()


"""
Evaluate Dice coefficient CV 0: Best_score = 0.9589819573059134 Best_threshold = 0.5 LB = 0,956846 (0.5)
Evaluate Dice coefficient CV 1: Best_score = 0.955407403962216 Best_threshold = 0.6499999999999999  LB = 0,954470 (0.5)
Evaluate Dice coefficient CV 2: Best_score = 0.9536193533506331 Best_threshold = 0.5499999999999999 LB = 0,956182 (0.5)
Evaluate Dice coefficient CV 3: Best_score = 0.9529933907900953 Best_threshold = 0.5999999999999999 LB = 0,957851 (0.5)
Evaluate Dice coefficient CV 4: Best_score = 0.9556511903047544 Best_threshold = 0.5499999999999999 LB = 0,957374 (0.5)

Stack = 0,958633 (0.59)
"""