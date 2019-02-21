from keras.callbacks import ModelCheckpoint, TensorBoard

import LoadBatches
from Models import FCN8,FCN32,SegNet,UNet
from keras import optimizers
import math

#############################################################################
# train_images_path = "D:\FCN\data\SegNet-Tutorial-master\CamVid\\train"
# train_segs_path = "D:\FCN\data\SegNet-Tutorial-master\CamVid\\trainannot"
train_images_path = 'train_image.npy'
train_segs_path = 'train_annotation.npy'
train_batch_size = 5
n_classes =2

epochs = 10

input_height=320
input_width=320



val_images_path = 'validation_image.npy'
val_segs_path = 'validation_annotation.npy'
val_batch_size = 4

key="fcn32"


##################################

method={"fcn32":FCN32.FCN32,"fcn8":FCN8.FCN8,'segnet':SegNet.SegNet,'unet':UNet.UNet}

model = method[key](n_classes,input_height=input_height,input_width=input_width)
model.compile(loss='categorical_crossentropy',optimizer="adadelta",metrics=['acc'])

G = LoadBatches.imageSegmentationGenerator(train_images_path,
                                   train_segs_path, train_batch_size, n_classes=n_classes, input_height=input_height,input_width=input_width)

G_test=LoadBatches.imageSegmentationGenerator(val_images_path,
                                   val_segs_path, val_batch_size, n_classes=n_classes, input_height=input_height,input_width=input_width)

checkpoint = ModelCheckpoint(filepath="output/%s_model.h5" % key, monitor='acc', mode='auto', save_best_only='True')
tensorboard = TensorBoard(log_dir='output/log_%s_model' % key)

model.fit_generator(generator=G,
                      steps_per_epoch=math.ceil(367./train_batch_size),
                      epochs=epochs,callbacks=[checkpoint,tensorboard],
                      verbose=2,
                      validation_data=G_test,
                      validation_steps=8,
                      shuffle=True)