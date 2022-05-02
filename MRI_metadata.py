import numpy as np
from Utils import Util

helper =Util()

class MetaData:

    def creat_MetaDatafile(self,data_pth):
        MRI_metadata = {}
        list_files = helper.get_list_files(data_pth)
        for ID in list_files:
            age=helper.get_label(ID)
            MRI_metadata[ID] = age
        np.save('/mnt/sdb2/age_estimation/dataset/metadata/MRI_meta_Train.npy', MRI_metadata)#helper.Path.MetaData_Path, MRI_metadata)

    def get_MetaData(self,pth, ID):
        MRI_MetaDatas = np.load(pth,allow_pickle=True).item()
        Age = MRI_MetaDatas[str(ID)]
        return Age

    def get_MetaData_all(self, pth):
        Ages = np.load(pth, allow_pickle=True).item()

        return Ages


