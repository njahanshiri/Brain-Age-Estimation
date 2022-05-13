import sys
sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import CSVLogger
from .models import model_2D
from keras import backend as K
from Data_loader.MRI_DataProvider import  BrainMRI_DataProvider
from Data_loader.DataGenerator import DataGenerator
from setting import SettingRepository
from opt import opt
import os
import math
import time

class age_model_2DCNN:

    def __init__(self, data_type, axis, log_path):
        self.data_provider = BrainMRI_DataProvider()
        self.data_type = data_type
        self.axis = axis
        self.log(log_path)

    def log(self, log_path):

        self.CSV_logger = CSVLogger(os.path.join(log_path, "log.csv"), append=True,
                                    separator=';')
        self.TensorBoard = TensorBoard(log_dir=log_path, update_freq='epoch')

        checkpoint_path = os.path.join(log_path,
                                       self.axis + "_"+ self.data_type + "_2DCNN.{epoch:02d}-{loss:.2f}.hdf5")

        self.checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=opt.save_best_only,
                                          save_weights_only=opt.save_weights_only,
                                          mode='auto')

    def coeff_determination(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))


    def age_model(self):

        model = model_2D()
        return model

    def create_model(self):
        self.model = self.age_model()
        self.model.compile(optimizer='adam', loss='mae', metrics=["mae", "mse", self.coeff_determination])  # optimizer = adam

    def train(self, data_generator_Train, data_generator_valid, epochs= opt.epochs_2D, optimizer= opt.optimizer, loss= opt.loss, model_path=None):

        STEP_SIZE_TRAIN = opt.Train_num // opt.batch_size_2D
        STEP_SIZE_VALID = opt.Valid_num // opt.batch_size_2D

        self.model = self.age_model()
        if model_path:
            self.model.load_weights(model_path)

        self.model.compile(optimizer=optimizer, loss=loss, metrics=["mae", "mse", self.coeff_determination])  # optimizer = adam
        self.model.fit(data_generator_Train, validation_data= data_generator_valid, validation_steps= STEP_SIZE_VALID, steps_per_epoch= STEP_SIZE_TRAIN, epochs= epochs, callbacks=[self.TensorBoard, self.checkpoint, self.CSV_logger])


    def evaluate(self, test_data, y_data, model_path=None):

        if model_path:
            self.create_model()
            self.model.load_weights(model_path)

        score = self.model.evaluate(test_data, y_data)
        return score

    def predict(self, test_data):

        test_pred = self.model.predict(test_data)
        return test_pred

    def create_data_test(self):

        return  self.data_provider.get_test_data(self.data_type, self.axis)

    def create_data_train(self):

        data_generator_Train = DataGenerator(SettingRepository(data_provider=self.data_provider,
                                                               num_sample=opt.Train_num,
                                                               batch_size=opt.batch_size_2D,
                                                               data_type= self.data_type,
                                                               type_="train", ax_=self.axis))

        data_generator_Valid = DataGenerator(SettingRepository(data_provider=self.data_provider,
                                                               num_sample=opt.Valid_num,
                                                               batch_size=opt.batch_size_2D,
                                                               data_type=self.data_type,
                                                               type_="valid", ax_=self.axis))
        return  data_generator_Train, data_generator_Valid

    def test(self, model_path=None):

        x, y = self.create_data_test()
        scores = self.evaluate(x, y, model_path)
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")
        print(" %s matter/3D_model score, MAE, MSE, R2, RMSE" % self.data_type, scores[1], scores[2],
              scores[3], math.sqrt(scores[2]))
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")


    def run (self):
        data_generator_Train, data_generator_Valid = self.create_data_train()
        start_time = time.time()
        self.train( data_generator_Train, data_generator_Train)
        stop_time = time.time()
        print("Training_Time_Duration   :", stop_time - start_time)
        self.test()






