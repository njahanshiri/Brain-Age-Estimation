from Utils import Util
from  MRI_metadata import MetaData
import nibabel as nib
import numpy as np
import os
helper = Util()
import random
from opt import opt
from setting import SettingRepository
random.seed(20)
class BrainMRI_DataProvider():

    def get_sample_IDs(self,type_):
        if type_ == "train":
                 sample_IDs = helper.get_list_files(opt.Train_WhitePath)
        elif type_ == "valid":
                 sample_IDs = helper.get_list_files(opt.Valid_WhitePath)
        elif type_ == "test":
                 sample_IDs = helper.get_list_files(opt.Test_whitePath)
        else:
            print("error")

        return sample_IDs

    def Read_MRIData(self,Data_path):
        n1_img = nib.load(Data_path)
        tmp = np.array(n1_img.get_data())
        if tmp.shape != (80, 80, 80):
            tmp = tmp[helper.constants.width_Range, helper.constants.height_Range, helper.constants.depth_Range]

        return tmp

    def get_test_data(self, data_type=None, ax=None):

        if ax:
            batch_x, batch_y = self.get_test_2DCNN(data_type, ax)
        elif data_type:
            batch_x, batch_y = self.get_test_3DCNN(data_type)
        else:
            batch_x, batch_y = self.get_test_ECNN()
        return batch_x, batch_y

    def get_test_2DCNN(self, data_type, ax):

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
                if ax == "Axial":
                    tmp = Data
                if ax == "Sagittal":
                    tmp = np.swapaxes(Data, 0, 2)
                if ax == "Coronal":
                    tmp = np.swapaxes(Data, 1, 2)
                if ax == "all":
                    tmp = Data
                batch_x.append(tmp)
                batch_y.append(Age_Sample)

        batch_x = np.asarray(batch_x)
        return batch_x, batch_y

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
        return batch_x, batch_y

    def get_test_ECNN(self):
        Sagittal_gray, y = self.get_test_2DCNN("Gray", "Sagittal")
        Coronal_gray, _ = self.get_test_2DCNN("Gray", "Coronal")
        Axial_gray, _ = self.get_test_2DCNN("Gray", "Axial")

        Sagittal_White, _ = self.get_test_2DCNN("White", "Sagittal")
        Coronal_White, _ = self.get_test_2DCNN("White", "Coronal")
        Axial_White, _ = self.get_test_2DCNN("White", "Axial")

        gray_3DCNN, _ = self.get_test_3DCNN("Gray")
        White_3DCNN, _ = self.get_test_3DCNN("White")

        return [Sagittal_gray, Axial_gray, Coronal_gray, Sagittal_White, Axial_White, Coronal_White, gray_3DCNN, White_3DCNN], y


    def get_batch_data(self, setting, index):
        if setting.ax:
            batch_x, batch_y = self.get_train_2DCNN(setting, index)
        elif setting.data_type:
            batch_x, batch_y = self.get_train_3DCNN(setting, index)
        else:
            batch_x, batch_y = self.get_train_ECNN(setting, index)

        return  batch_x, batch_y

    def get_train_2DCNN(self, setting, index):

        MRIMetaData = MetaData()
        batch_x = []
        batch_y = []
        if index==0:
            index+=1
        loop_range = slice(setting.batch_size * (index - 1), setting.batch_size * index)

        if setting.type_ == "train":

            if setting.data_type == "Gray":
                pth = opt.Train_GrayPath
            elif setting.data_type == "White":
                pth = opt.Train_WhitePath

        elif setting.type_ == "valid":

            if setting.data_type == "Gray":
                pth = opt.Valid_GrayPath
            elif setting.data_type == "White":
                pth = opt.Valid_WhitePath

        Sample_IDs = self.get_sample_IDs(setting.type_)
        Sample_IDs  = random.sample(Sample_IDs, len(Sample_IDs))
        Sample_IDs = Sample_IDs[loop_range]
        for ID in Sample_IDs:

            if setting.type_ == "train":
                pth_ = opt.meta_data_train
            else:
                pth_ = opt.meta_data_test

            Age_Sample = MRIMetaData.get_MetaData(pth_, ID)
            IDs = helper.get_list_files(os.path.join(pth, ID))
            for MRI_ID in IDs:

                Data = self.Read_MRIData(os.path.join(pth, ID, MRI_ID))
                if setting.ax == "Axial":
                        tmp = Data
                if setting.ax == "Sagittal":

                    tmp = np.swapaxes(Data, 0, 2)
                if setting.ax == "Coronal":

                    tmp = np.swapaxes(Data, 1, 2)
                if setting.ax == "all":
                    tmp = Data

                batch_x.append(tmp)
                batch_y.append(Age_Sample)

        batch_x = np.asarray(batch_x)
        return batch_x , batch_y

    def get_train_3DCNN(self, setting, index):

        MRIMetaData = MetaData()
        batch_x = []
        batch_y = []
        if index == 0:
            index += 1
        loop_range = slice(setting.batch_size * (index - 1), setting.batch_size * index)

        if setting.type_ == "train":

            if setting.data_type == "Gray":
                pth = opt.Train_GrayPath
            elif setting.data_type == "White":
                pth = opt.Train_WhitePath

        elif setting.type_ == "valid":

            if setting.data_type == "Gray":
                pth = opt.Valid_GrayPath
            elif setting.data_type == "White":
                pth = opt.Valid_WhitePath

        Sample_IDs = self.get_sample_IDs(setting.type_)
        Sample_IDs = random.sample(Sample_IDs, len(Sample_IDs))
        Sample_IDs = Sample_IDs[loop_range]
        for ID in Sample_IDs:

            if setting.type_ == "train":
                pth_ = opt.meta_data_train
            else:
                pth_ = opt.meta_data_test

            Age_Sample = MRIMetaData.get_MetaData(pth_, ID)
            IDs = helper.get_list_files(os.path.join(pth, ID))
            for MRI_ID in IDs:

                Data = self.Read_MRIData(os.path.join(pth, ID, MRI_ID))
                batch_x.append(Data)
                batch_y.append(Age_Sample)

        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_x = np.asarray(batch_x)
        return batch_x, batch_y

    def get_train_ECNN(self, setting, index):

        MRIMetaData = MetaData()
        Sagittal_gray = []
        Axial_gray = []
        Coronal_gray = []
        Sagittal_White = []
        Axial_White = []
        Coronal_White = []
        gray_3DCNN = []
        White_3DCNN = []

        batch_y = []
        if index==0:
            index+=1
        loop_range = slice(setting.batch_size * (index - 1), setting.batch_size * index)

        if setting.type_ == "train":

            pth_Gray = opt.Train_GrayPath
            pth_White = opt.Train_WhitePath

        elif setting.type_ == "valid":

            pth_Gray = opt.Valid_GrayPath
            pth_White = opt.Valid_WhitePath

        Sample_IDs = self.get_sample_IDs(setting.type_)
        Sample_IDs  = random.sample(Sample_IDs, len(Sample_IDs))
        Sample_IDs = Sample_IDs[loop_range]
        for ID in Sample_IDs:

            if setting.type_ == "train":
                pth_ = opt.meta_data_train
            else:
                pth_ = opt.meta_data_test

            Age_Sample = MRIMetaData.get_MetaData(pth_, ID)
            IDs = helper.get_list_files(os.path.join(pth_Gray, ID))
            for MRI_ID in IDs:

                Data = self.Read_MRIData(os.path.join(pth_Gray, ID, MRI_ID))

                Axial_gray.append(Data)
                Sagittal_gray.append(np.swapaxes(Data, 0, 2))
                Coronal_gray.append(np.swapaxes(Data, 1, 2))
                gray_3DCNN.append(Data)
                batch_y.append(Age_Sample)

            IDs = helper.get_list_files(os.path.join(pth_White, ID))
            for MRI_ID in IDs:
                Data = self.Read_MRIData(os.path.join(pth_White, ID, MRI_ID))
                Axial_White.append(Data)
                Sagittal_White.append(np.swapaxes(Data, 0, 2))
                Coronal_White.append(np.swapaxes(Data, 1, 2))
                White_3DCNN.append(Data)

        Sagittal_gray = np.asarray(Sagittal_gray)
        Axial_gray = np.asarray(Axial_gray)
        Coronal_gray = np.asarray(Coronal_gray)
        Sagittal_White = np.asarray(Sagittal_White)
        Axial_White = np.asarray(Axial_White)
        Coronal_White = np.asarray(Coronal_White)
        gray_3DCNN = np.expand_dims(gray_3DCNN, axis=-1)
        gray_3DCNN = np.asarray(gray_3DCNN)
        White_3DCNN = np.expand_dims(White_3DCNN, axis=-1)
        White_3DCNN = np.asarray(White_3DCNN)

        return [Sagittal_gray, Axial_gray, Coronal_gray, Sagittal_White, Axial_White, Coronal_White, gray_3DCNN,
                White_3DCNN], batch_y