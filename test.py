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

TEST_DIR = '/workdir/data/datasets/picsart/test/images'
RESULT_DIR = '/workdir/data/results/test/'

# TEST_DIR = '/workdir/data/datasets/picsart/train/images'
# RESULT_DIR = '/workdir/data/results/train/'

test_imgs = os.listdir(TEST_DIR)

device = 'GPU:0'
model_name = 'mobilenet_small' # 'mobilenet_small'
input_size = 448

#old_model_path = '/workdir/data/experiments/fb_combined_mobilenetv3_tf_coco_pixart_supervisely/model_best_0.12529000639915466.h5'
#old_model_path = '/workdir/data/experiments/picsart_supervisely/model_best_0.3700900077819824.h5' # large
#old_model_path = '/workdir/data/experiments/picsart_supervisely/model_best_0.26719000935554504.h5' # small
old_model_path = '/workdir/data/experiments/picsart/model_best_0.16599999368190765.h5' # small

mobilenet_model = Model(device=device,
                        model_name=model_name,
                        n_class=1,
                        input_shape=(1,input_size,input_size,3),
                        old_model_path=old_model_path)
                        
for i, img_path in enumerate(test_imgs):
    img_name = img_path
    img_name = img_name.split('.')[0]
    img_path = os.path.join(TEST_DIR, img_path)

    print(img_path)

    test_img = cv2.imread(img_path)
    test_img = test_img[:,:,::-1]
    test_img = cv2.resize(test_img, (input_size,input_size))
    test_tensor = ToTensorColor()(test_img)
    test_tensor = tf.expand_dims(test_tensor, 0)
    out = mobilenet_model.predict(test_tensor)
    out_img = np.squeeze(out)
    print(out_img.shape)
    print("--")

    test_img = cv2.resize(test_img, (224,224))
    plt.imshow(test_img)
    plt.imshow((out_img>0.5)*255, alpha=0.4)
    plt.show()

    plt.savefig(RESULT_DIR + img_name + '.png')

    if i > 29:
        break