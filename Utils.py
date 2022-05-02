import os
import xlrd
import nibabel as nib
import numpy as np
import random
import itertools
import shutil
from opt import opt
import glob

class Util:


    class constants:
        width = 80
        height = 80
        depth = 80
        width_Range = slice(6, 86)
        height_Range = slice(16, 96)
        depth_Range = slice(6, 86)
        rotate = ([1, 0], [0, 2])
        translate = ([10, 10, 10], [-10, -10, -10], [-40, -40, -40], [40, 40, 40])
        sp = False

    def find_path(self, type):
        path = os.path.join(opt.log, type)
        list_of_files = glob.glob(path+ "/*.hdf5")
        latest_file = max(list_of_files, key=os.path.getctime)
        return latest_file

    @classmethod
    def get_IDs(self, path):

        IDs = []
        filenames = os.listdir(path)
        for filename in filenames:
            IDs.append(filename)
        return IDs

    def get_len(self, path):
        count1 = 0
        for root, dirs, files in os.walk(path):
            count1 += len(dirs)

        return count1

    def get_ID_MRIFile(self, pth, DataType, ID):

        MRI_Files = []
        pth = self.get_pth(DataType, ID)
        MRI_Files = self.get_IDs(pth)

        return MRI_Files

    def get_label(self, ID):
        # print("input",ID)
        label = -1;
        worksheet = self.open_exl()
        for idx, mt in enumerate(worksheet.col_values(0)):
            if mt != worksheet.cell_value(0, 0):
                if mt == ID:
                    label = worksheet.cell_value(idx, 1)
                    # print("excel, label-------------",mt, label)
        if (label == -1):
            print("not label***********", ID)
        return label

    def open_exl(self):

        workbook = xlrd.open_workbook(opt.Label_Path)
        worksheet = workbook.sheet_by_name('Sheet1')

        return worksheet

    def get_pth(self, DataType, ID):

        Path_MRI_Files = []
        if DataType == 'Grey':
            Path_MRI_Files = os.path.join(self.Path.Data_GreyPath, ID)
        elif DataType == 'White':
            Path_MRI_Files = os.path.join(self.Path.Data_WhitePath, ID)
        elif DataType == 'CSF':
            Path_MRI_Files = os.path.join(self.Path.Data_CSF_Path, ID)
        else:
            Path_MRI_Files = os.path.join(self.Path.Data_RawPath, ID)

        return Path_MRI_Files

    def load_MRI_data(self, pth):

        n1_img = nib.load(pth)
        tmp = np.array(n1_img.get_data())
        if (tmp.shape != (80, 80, 80)):
            tmp = tmp[self.constants.width_Range, self.constants.height_Range, self.constants.depth_Range]

        return tmp

    def get_list_files(self, pth):
        if not os.path.exists(pth):
            print("path not exist", pth)

        files = os.listdir(pth)
        return files

    def spilit_train_test_valid(self):

        list_files = self.get_list_files(self.Path.Data_GreyPath)
        random.seed(10)
        random.shuffle(list_files)

        test_data = list_files[:200]
        valid_data = list_files[200:400]
        train_data = list_files[400:]

        return test_data, train_data, valid_data

    def make_train_test_valid_DIR(self, datatype):

        if not os.path.exists(self.Path.Pth_Split):
            os.mkdir(self.Path.Pth_Split)
        if datatype == 'Grey':
            if not os.path.exists(os.path.join(self.Path.Pth_Split, 'Grey')):
                self.make_subDIR(datatype)
        else:
            if not os.path.exists(os.path.join(self.Path.Pth_Split, 'White')):
                self.make_subDIR("White")

    def make_subDIR(self, datatype):
        os.mkdir(os.path.join(self.Path.Pth_Split, datatype))
        os.mkdir(os.path.join(self.Path.Pth_Split, datatype, 'Train'))
        os.mkdir(os.path.join(self.Path.Pth_Split, datatype, 'Test'))
        os.mkdir(os.path.join(self.Path.Pth_Split, datatype, 'Valid'))

    def copy_folder(self, datatype, name, Stype):
        if datatype == 'Grey':
            src = os.path.join(self.Path.Data_GreyPath, name)
            dest = os.path.join(self.Path.Pth_Split, datatype, Stype, name)

        else:
            src = os.path.join(self.Path.Data_WhitePath, name)
            dest = os.path.join(self.Path.Pth_Split, datatype, Stype, name)
        try:
            shutil.copytree(src, dest)
        # Directories are the same
        except shutil.Error as e:
            print('Directory not copied. Error: %s' % e)
        # Any error saying that the directory doesn't exist
        except OSError as e:
            print('Directory not copied. Error: %s' % e)

    def Data_Split(self, datatype):
        testfiles, trainfiles, validfiles = self.spilit_train_test_valid()
        self.make_train_test_valid_DIR(datatype)
        for i in range(0, self.constants.Test_num):
            self.copy_folder(datatype, testfiles[i], 'Test')

        for i in range(0, self.constants.Valid_num):
            self.copy_folder(datatype, validfiles[i], 'Valid')

        for i in range(0, self.constants.Train_num):
            self.copy_folder(datatype, trainfiles[i], 'Train')

    def get_sample_IDs(self, type_):
        if type_ == "train":
            sample_IDs = self.get_list_files(self.Path.Train_WhitePath)
        elif type_ == "valid":
            sample_IDs = self.get_list_files(self.Path.Valid_WhitePath)
        elif type_ == "test":
            sample_IDs = self.get_list_files(self.Path.Test_whitePath)
        else:
            print("error")

        return sample_IDs

