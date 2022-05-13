import sys
sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.callbacks import CSVLogger
from keras import backend as K
from keras.models import Model
from tensorflow.keras import optimizers
from keras.layers import *
from  Data_loader.MRI_DataProvider import BrainMRI_DataProvider
from Data_loader.DataGenerator import DataGenerator
from setting import SettingRepository
from .models import model_2D, model_3D
from opt import opt
import os
import math
import glob
import time




class age_model_ECNN:
    def __init__(self, log_path):

        self.data_provider = BrainMRI_DataProvider()
        self.log(log_path)


    def log(self, log_path):
        self.CSV_logger = CSVLogger(os.path.join(log_path, "log.csv"), append=True,
                                    separator=';')
        self.TensorBoard = TensorBoard(log_dir=log_path,  update_freq='epoch')
        checkpoint_path = os.path.join(log_path,
                                       "ECNN.{epoch:02d}-{loss:.2f}.hdf5")
        self.checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=opt.save_best_only,
                                          save_weights_only=opt.save_weights_only,
                                          mode='auto')

    def coeff_determination(self, y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    def find_path(self, type):
        path = os.path.join(opt.log, type)
        list_of_files = glob.glob(path+ "/*.hdf5")
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    def age_model(self):

        Sagittal_gray_model = model_2D(Dense_128_name="layer_128_Sagittal_gray")

        Axial_gray_model = model_2D(Dense_128_name="layer_128_Axial_gray")
        Coronal_gray_model = model_2D(Dense_128_name="layer_128_Coronal_gray")

        Sagittal_white_model = model_2D(Dense_128_name="layer_128_Sagittal_white")
        Axial_white_model = model_2D(Dense_128_name="layer_128_Axial_white")
        Coronal_white_model = model_2D(Dense_128_name="layer_128_Coronal_white")

        gray_3D_model = model_3D(Dense_128_name="layer_128_gray_3D")
        white_3D_model = model_3D(Dense_128_name="layer_128_white_3D")

        Sagittal_gray_model.load_weights(self.find_path("Sagittal_Gray"))

        Axial_gray_model.load_weights(self.find_path("Axial_Gray"))
        Coronal_gray_model.load_weights(self.find_path("Coronal_Gray"))

        Sagittal_white_model.load_weights(self.find_path("Sagittal_White"))
        Axial_white_model.load_weights(self.find_path("Axial_White"))
        Coronal_white_model.load_weights(self.find_path("Coronal_White"))

        gray_3D_model.load_weights(self.find_path("Gray_3DCNN"))
        white_3D_model.load_weights(self.find_path("White_3DCNN"))

        concat_layer = concatenate(
            [Sagittal_gray_model.get_layer("layer_128_Sagittal_gray").output, Axial_gray_model.get_layer("layer_128_Axial_gray").output, Coronal_gray_model.get_layer("layer_128_Coronal_gray").output,
             Sagittal_white_model.get_layer("layer_128_Sagittal_white").output, Axial_white_model.get_layer("layer_128_Axial_white").output, Coronal_white_model.get_layer("layer_128_Coronal_white").output,
             gray_3D_model.get_layer("layer_128_gray_3D").output, white_3D_model.get_layer("layer_128_white_3D").output])

        Dense256 = Dense(256)(concat_layer)
        LeakyReLU256 = LeakyReLU()(Dense256)
        Dense1 = Dense(1)(LeakyReLU256)
        output = LeakyReLU()(Dense1)
        model = Model(
            [Sagittal_gray_model.input, Axial_gray_model.input, Coronal_gray_model.input, Sagittal_white_model.input, Axial_white_model.input, Coronal_white_model.input, gray_3D_model.input,
             white_3D_model.input], output)

        return model

    def create_model(self):
        self.model = self.age_model()
        self.model.compile(optimizer='adam', loss='mae',
                           metrics=["mae", "mse", self.coeff_determination])  # optimizer = adam

    def train(self, data_generator_Train, data_generator_valid, epochs= opt.epochs_ECNN, lr = opt.lr_ECNN, loss= opt.loss, model_path=None):

        STEP_SIZE_TRAIN = opt.Train_num // opt.batch_size
        STEP_SIZE_VALID = opt.Valid_num // opt.batch_size

        Sagittal_gray_model = model_2D(Dense_128_name="layer_128_Sagittal_gray")

        Axial_gray_model = model_2D(Dense_128_name="layer_128_Axial_gray")
        Coronal_gray_model = model_2D(Dense_128_name="layer_128_Coronal_gray")

        Sagittal_white_model = model_2D(Dense_128_name="layer_128_Sagittal_white")
        Axial_white_model = model_2D(Dense_128_name="layer_128_Axial_white")
        Coronal_white_model = model_2D(Dense_128_name="layer_128_Coronal_white")

        gray_3D_model = model_3D(Dense_128_name="layer_128_gray_3D")
        white_3D_model = model_3D(Dense_128_name="layer_128_white_3D")

        Sagittal_gray_model.load_weights(self.find_path("Sagittal_Gray"))

        Axial_gray_model.load_weights(self.find_path("Axial_Gray"))
        Coronal_gray_model.load_weights(self.find_path("Coronal_Gray"))

        Sagittal_white_model.load_weights(self.find_path("Sagittal_White"))
        Axial_white_model.load_weights(self.find_path("Axial_White"))
        Coronal_white_model.load_weights(self.find_path("Coronal_White"))

        gray_3D_model.load_weights(self.find_path("Gray_3DCNN"))
        white_3D_model.load_weights(self.find_path("White_3DCNN"))

        concat_layer = concatenate(
            [Sagittal_gray_model.get_layer("layer_128_Sagittal_gray").output,
             Axial_gray_model.get_layer("layer_128_Axial_gray").output,
             Coronal_gray_model.get_layer("layer_128_Coronal_gray").output,
             Sagittal_white_model.get_layer("layer_128_Sagittal_white").output,
             Axial_white_model.get_layer("layer_128_Axial_white").output,
             Coronal_white_model.get_layer("layer_128_Coronal_white").output,
             gray_3D_model.get_layer("layer_128_gray_3D").output,
             white_3D_model.get_layer("layer_128_white_3D").output])

        Dense256 = Dense(256)(concat_layer)
        LeakyReLU256 = LeakyReLU()(Dense256)
        Dense1 = Dense(1)(LeakyReLU256)
        output = LeakyReLU()(Dense1)
        self.model = Model(
            [Sagittal_gray_model.input, Axial_gray_model.input, Coronal_gray_model.input, Sagittal_white_model.input,
             Axial_white_model.input, Coronal_white_model.input, gray_3D_model.input,
             white_3D_model.input], output)

        if model_path:
            self.model.load_weights(model_path)
        optimizer = optimizers.Adam(lr)
        self.model.compile(optimizer=optimizer, loss=loss,  metrics=["mae", "mse", self.coeff_determination])  # optimizer = adam
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

        return  self.data_provider.get_test_data(None, None)

    def create_data_train(self):

        data_generator_Train = DataGenerator(SettingRepository(data_provider=self.data_provider,
                                                               num_sample=opt.Train_num,
                                                               batch_size=opt.batch_size,
                                                               data_type=None,
                                                               type_="train", ax_=None))

        data_generator_Valid = DataGenerator(SettingRepository(data_provider=self.data_provider,
                                                               num_sample=opt.Valid_num,
                                                               batch_size=opt.batch_size,
                                                               data_type=None,
                                                               type_="valid", ax_=None))
        return  data_generator_Train, data_generator_Valid

    def test(self, model_path=None):
        x, y = self.create_data_test()
        scores = self.evaluate(x, y, model_path)
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")
        print(" ECNN score, MAE, MSE, R2, RMSE" , scores[1], scores[2],
              scores[3], math.sqrt(scores[2]))
        print("---------------------------------------------------------------------------------------")
        print("---------------------------------------------------------------------------------------")

    def run (self):
        data_generator_Train, data_generator_Valid = self.create_data_train()
        start_time = time.time()
        self.train( data_generator_Train, data_generator_Valid)
        stop_time = time.time()
        print("Training_Time_Duration   :", stop_time - start_time)
        self.test()






