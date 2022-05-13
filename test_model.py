from  model.code.age_model_2D import age_model_2DCNN
import sys
sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import CSVLogger

from keras import backend as K
from Data_loader.MRI_DataProvider import  BrainMRI_DataProvider
from Data_loader.DataGenerator import DataGenerator
from setting import SettingRepository
from opt import opt
import os
import math
import time


# model = age_model_2DCNN("Gray", "Sagittal", "./Log")
#
#
# model.run()

#
# import sys
# sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')
# from keras.callbacks import TensorBoard, ModelCheckpoint
# from keras.callbacks import CSVLogger
# from .models import model_2D
# from keras import backend as K
# from Data_loader.MRI_DataProvider import  BrainMRI_DataProvider
# from Data_loader.DataGenerator import DataGenerator
# from setting import SettingRepository
# from opt import opt
# import os
# import math
# import time
#
#
# class age_model_2DCNN:
#
#     def __init__(self, data_type, axis, log_path):
#
#         self.data_provider = BrainMRI_DataProvider()
#         self.data_type = data_type
#         self.axis = axis
#         self.log(log_path)
#
#
#
#     def log(self, log_path):
#
#         self.CSV_logger = CSVLogger(os.path.join(log_path, "log.csv"), append=True,
#                                     separator=';')
#         self.TensorBoard = TensorBoard(log_dir=log_path, batch_size=opt.batch_size_2D, update_freq='epoch')
#         print(type(self.axis),self.axis,  type(self.data_type),  self.data_type)
#         checkpoint_path = os.path.join(log_path,
#                                        self.axis + "_"+ self.data_type + "_2DCNN.{epoch:02d}-{loss:.2f}.hdf5")
#         self.checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True,
#                                           save_weights_only=False,
#                                           mode='auto', period=1)
#
#     def coeff_determination(self, y_true, y_pred):
#         SS_res = K.sum(K.square(y_true - y_pred))
#         SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#         return (1 - SS_res / (SS_tot + K.epsilon()))
#
#     def age_model(self):
#
#         model = model_2D()
#         return model
#
#     def create_model(self):
#         self.model = self.age_model()
#         self.model.compile(optimizer='adam', loss='mae', metrics=["mae", "mse", self.coeff_determination])  # optimizer = adam
#
#     def train(self, data_generator_Train, data_generator_valid, epochs= opt.epochs_2D, optimizer= opt.optimizer, loss= opt.loss, model_path=None):
#
#         STEP_SIZE_TRAIN = opt.Train_num // opt.batch_size_2D
#         STEP_SIZE_VALID = opt.Valid_num // opt.batch_size_2D
#
#         self.model = self.age_model()
#         if model_path:
#             self.model.load_weights(model_path)
#
#         self.model.compile(optimizer=optimizer, loss=loss, metrics=["mae", "mse", self.coeff_determination])  # optimizer = adam
#         self.model.fit_generator(data_generator_Train,epochs= epochs)#, validation_data= data_generator_valid, validation_steps= STEP_SIZE_VALID, steps_per_epoch= STEP_SIZE_TRAIN, epochs= epochs, callbacks=[self.TensorBoard, self.checkpoint, self.CSV_logger])
#
#
#     def evaluate(self, test_data, y_data, model_path=None):
#
#         if model_path:
#             self.create_model()
#             self.model.load_weights(model_path)
#
#         score = self.model.evaluate(test_data, y_data)
#         return score
#
#     def predict(self, test_data):
#
#         test_pred = self.model.predict(test_data)
#         return test_pred
#
#     def create_data_test(self):
#
#         return  self.data_provider.get_test_data(self.data_type, self.axis)
#

data_provider = BrainMRI_DataProvider()

def create_data_train():
    data_generator_Train = DataGenerator(SettingRepository(data_provider= data_provider,
                                                           num_sample=100,
                                                           batch_size=3,
                                                           data_type="Gray",
                                                           type_="train", ax_="Sagittal"))

    return  data_generator_Train,
#
#     def test(self, model_path=None):
#
#         x, y = self.create_data_test()
#         scores = self.evaluate(x, y, model_path)
#         print("---------------------------------------------------------------------------------------")
#         print("---------------------------------------------------------------------------------------")
#         print(" %s matter/3D_model score, MAE, MSE, R2, RMSE" % self.data_type, scores[1], scores[2],
#               scores[3], math.sqrt(scores[2]))
#         print("---------------------------------------------------------------------------------------")
#         print("---------------------------------------------------------------------------------------")
#
#
#     def run (self):
#         data_generator_Train, data_generator_Valid = self.create_data_train()
#         start_time = time.time()
#         self.train( data_generator_Train, data_generator_Train)
#         stop_time = time.time()
#         print("Training_Time_Duration   :", stop_time - start_time)
#         self.test()
#

from keras.models import  Sequential
from keras.layers import *

model = Sequential()
model.add(Conv3D(16, kernel_size=(3, 3, 3), input_shape=(80, 80, 80, 1)))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(32, kernel_size=(3, 3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(64, kernel_size=(3, 3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Conv3D(128, kernel_size=(3, 3, 3)))
model.add(LeakyReLU())
model.add(Conv3D(256, kernel_size=(3, 3, 3)))
model.add(LeakyReLU())
model.add(Conv3D(512, kernel_size=(3, 3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Flatten())
model.add(Dense(128))
model.add(LeakyReLU())
model.add(Dense(1))
model.add(LeakyReLU())

data_generator_Train= create_data_train()
#
# import sys
# sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')
from utils.Utils import Util
from  Data_loader.MRI_metadata import MetaData
import nibabel as nib
import numpy as np
import os
helper = Util()
import random
from opt import opt
from setting import SettingRepository
random.seed(20)
# class BrainMRI_DataProvider():
#
def get_sample_IDs(type_):
    if type_ == "train":
             sample_IDs = helper.get_list_files(opt.Train_WhitePath)
    elif type_ == "valid":
             sample_IDs = helper.get_list_files(opt.Valid_WhitePath)
    elif type_ == "test":
             sample_IDs = helper.get_list_files(opt.Test_whitePath)
    else:
        print("error")

    return sample_IDs

def Read_MRIData(Data_path):
    n1_img = nib.load(Data_path)
    tmp = np.array(n1_img.get_data())
    if tmp.shape != (80, 80, 80):
        tmp = tmp[helper.constants.width_Range, helper.constants.height_Range, helper.constants.depth_Range]
    return tmp


import numpy as np


def get_test_3DCNN(self, data_type):
    MRIMetaData = MetaData()
    batch_x = []
    batch_y = []
    if data_type == "Gray":
        pth = opt.Test_GrayPath
    elif data_type == "White":
        pth = opt.Test_whitePath

    Sample_IDs = self.get_sample_IDs("test")
    for ID in Sample_IDs:

        pth_ = opt.meta_data_test
        Age_Sample = MRIMetaData.get_MetaData(pth_, ID)

        IDs = helper.get_list_files(os.path.join(pth, ID))
        for MRI_ID in IDs:
            Data = self.Read_MRIData(os.path.join(pth, ID, MRI_ID))
            batch_x.append(Data)
            batch_y.append(Age_Sample)

    batch_x = np.expand_dims(batch_x, axis=-1)
    batch_x = np.asarray(batch_x)
    batch_y = np.asarray(batch_y)
    return batch_x, batch_y

x, y = get_test_3DCNN()
STEP_SIZE_TRAIN = opt.Train_num // opt.batch_size_3D
STEP_SIZE_VALID = opt.Valid_num // opt.batch_size_3D

# x = np.random.rand(10,80,80,80)
# y= np.random.rand(10,1)
model.compile(optimizer='adam', loss='mse', metrics=["mae", "mse"])  # optimizer = adam
model.fit(x,y,epochs= 1)#, validation_data= data_generator_valid, validation_steps= STEP_SIZE_VALID, steps_per_epoch= STEP_SIZE_TRAIN, epochs= epochs, callbacks=[self.TensorBoard, self.checkpoint, self.CSV_logger])
