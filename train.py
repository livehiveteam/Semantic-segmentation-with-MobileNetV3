import numpy as np
import cv2
import os
from os.path import join
import tensorflow as tf

from modules.datasets import ImageTargetDataset, RandomConcatDataset
from modules.segm_transforms import train_transforms, test_transforms, ToTensorColor
from modules.metrics import FbSegm
from train.train import Model

import matplotlib.pyplot as plt

train_batch_size = 16
val_batch_size = 8
INPUT_SIZE = (224, 224)
# INPUT_SIZE = (224, 224)
INPUT_SIZE = (448, 448)
AUG_PARAMS = [0.75, 1.25, 0.75, 1.25, 0.6, 1.4]
ANG_RANGE = 15
device = 'GPU:0' #'CPU:0'

train_trns = train_transforms(dataset='picsart', scale_size=INPUT_SIZE, ang_range=ANG_RANGE,
                                      augment_params=AUG_PARAMS, add_background=False,
                                      crop_scale=0.02)
val_trns = test_transforms(dataset='picsart', scale_size=INPUT_SIZE)

data_dirs_hq = [
    '/workdir/data/datasets/picsart/',
    '/workdir/data/datasets/supervisely/',
]

# data_dirs_coco = [
#     '/workdir/data/datasets/coco_person/'
# #     '/workdir/data/datasets/cityscapes_person/',
# ]

train_dirs_hq = [join(d, 'train') for d in data_dirs_hq]
val_dirs_hq = [join(d, 'val') for d in data_dirs_hq]
# train_dirs_coco = [join(d, 'train') for d in data_dirs_coco]
# val_dirs_coco = [join(d, 'val') for d in data_dirs_coco]

train_dataset_hq = ImageTargetDataset(train_dirs_hq,
                                           train_batch_size,
                                           shuffle=True,
                                           device=device,
                                           **train_trns,
                                           IMG_EXTN='.jpg',
                                           TRGT_EXTN='.png')
val_dataset_hq = ImageTargetDataset(val_dirs_hq,
                                           val_batch_size,
                                           shuffle=False,
                                           device=device,
                                           **val_trns,
                                           IMG_EXTN='.jpg',
                                           TRGT_EXTN='.png')

# train_dataset_coco = ImageTargetDataset(train_dirs_coco,
#                                            train_batch_size,
#                                            shuffle=True,
#                                            device=device,
#                                            **train_trns,
#                                            IMG_EXTN='.jpg',
#                                            TRGT_EXTN='.png')
# val_dataset_coco = ImageTargetDataset(val_dirs_coco,
#                                            val_batch_size,
#                                            shuffle=False,
#                                            device=device,
#                                            **val_trns,
#                                            IMG_EXTN='.jpg',
#                                            TRGT_EXTN='.png')

train_dataset = RandomConcatDataset([train_dataset_hq],
                                    [1.0], size=427)

# for x in train_dataset:
#     img, target = x[0], x[1]
#     for i in range(8):
#         plt.imshow(img[i])
#         plt.show()
#         plt.imshow(np.squeeze(target[i]))
#         plt.show()
#     break

# for x in val_dataset_hq:
#     img, target = x[0], x[1]
#     for i in range(8):
#         plt.imshow(img[i])
#         plt.show()
#         plt.imshow(np.squeeze(target[i]))
#         plt.show()
#     break


model_name = 'mobilenet_large'
# model_name = 'test_model'
n_class=1

# Train params
n_train = len(train_dataset)
n_val = len(val_dataset_hq)

print(n_train, n_val)
input("...")

# loss_name='bce'
loss_name = 'fb_combined'
optimizer = 'Adam'
lr = 0.00001
batch_size = train_batch_size
max_epoches = 1000
save_directory = '/workdir/data/experiments/picsart_supervisely'
reduce_factor = 0.95 # 0.75
epoches_limit = 3
early_stoping = 100
metrics = [FbSegm(channel_axis=-1)]
old_model_path = None # '/workdir/data/experiments/fb_combined_mobilenetv3_tf_coco_pixart_supervisely/model_best_0.14590999484062195.h5'

mobilenet_model = Model(device=device,
                        model_name=model_name,
                        n_class=n_class,
                        input_shape=(1,INPUT_SIZE[0],INPUT_SIZE[1],3),
                        shape=INPUT_SIZE,
                        old_model_path=old_model_path)

mobilenet_model.prepare_train(train_loader=train_dataset,
                              val_loader=val_dataset_hq,
                              n_train=n_train,
                              n_val=n_val,
                              loss_name=loss_name,
                              optimizer=optimizer,
                              lr = lr,
                              batch_size = batch_size,
                              max_epoches = max_epoches,
                              save_directory = save_directory,
#                               use_bce=True,
                              reduce_factor=reduce_factor,
                              epoches_limit=epoches_limit,
                              early_stoping=early_stoping,
                              metrics=metrics)

mobilenet_model.fit()