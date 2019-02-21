from model import *
from data import *
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.callbacks import ModelCheckpoint

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
#myGene = trainGenerator(1,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(1,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=100,epochs=1,callbacks=[model_checkpoint])

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene,1,verbose=1)
saveResult("data/membrane/val",results)

# plt.imshow(new_label)
# plt.show()
#
#
# # model.load_weights('unet_membrane.hdf5')
# # print(testGene)
# # pr = model.predict(np.expand_dims(testGene, 0))[0]
# # pr = pr.reshape((256, 256, 1)).argmax(axis=2)
