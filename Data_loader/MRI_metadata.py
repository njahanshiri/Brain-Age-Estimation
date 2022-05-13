import sys
sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')
import numpy as np
from utils.Utils import Util

helper =Util()

class MetaData:

    def creat_MetaDatafile(self,data_pth):
        MRI_metadata = {}
        list_files = helper.get_list_files(data_pth)
        for ID in list_files:
            age=helper.get_label(ID)
            MRI_metadata[ID] = age
        np.save('/media/jsh/Data/age_estimation/age_estimation_code/code/data/metadata/MRI_meta_valid.npy', MRI_metadata)#helper.Path.MetaData_Path, MRI_metadata)

    def get_MetaData(self,pth, ID):
        MRI_MetaDatas = np.load(pth,allow_pickle=True).item()
        ID = ID.split("_")[0]
        Age = MRI_MetaDatas[str(ID)]
        return Age

    def get_MetaData_all(self, pth):
        Ages = np.load(pth, allow_pickle=True).item()
        return Ages
#
if __name__ == '__main__':
    metadata = MetaData()
    metadata.creat_MetaDatafile("/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/White/valid")