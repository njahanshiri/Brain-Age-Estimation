from age_model import age_model
from opt import opt
from Utils import Util

helper =Util()


'''-------------------------------------------2D CNN models------------------------------'''

print("##################################")
#
age_model("2DCNN", level ="test", model_path = helper.find_path("Sagittal_Gray"), data_type="Gray", ax="Sagittal", log_pth=opt.Sagittal_Gray_log) # 2d CNN models on Gray matter/Sagittal

age_model("2DCNN", level ="test", model_path = helper.find_path("Coronal_Gray"), data_type="Gray", ax="Coronal", log_pth=opt.Coronal_Gray_log) # 2d CNN models on Gray matter/Coronal

age_model("2DCNN", level ="test", model_path = helper.find_path("Axial_Gray"), data_type="Gray", ax="Axial", log_pth=opt.Axial_Gray_log) # 2d CNN models on Gray matter/Axial


age_model("2DCNN", level ="test", model_path = helper.find_path("Sagittal_White"), data_type="White", ax="Sagittal", log_pth=opt.Sagittal_White_log) # 2d CNN models on White matter/Sagittal

age_model("2DCNN", level ="test", model_path = helper.find_path("Coronal_White"), data_type="White", ax="Coronal", log_pth=opt.Coronal_White_log) # 2d CNN models on White matter/Coronal

age_model("2DCNN", level ="test", model_path = helper.find_path("Axial_White"), data_type="White", ax="Axial", log_pth=opt.Axial_White_log) # 2d CNN models on White matter/Axial



'''-------------------------------------------3D CNN models------------------------------'''

age_model("3DCNN", level ="test", model_path = helper.find_path("White_3DCNN"), data_type="White", log_pth=opt.White_3DCNN_log) # White matter
age_model("3DCNN", level ="test", model_path = helper.find_path("Gray_3DCNN"), data_type="Gray", log_pth=opt.Gray_3DCNN_log) # Gray matter


'''-------------------------------------------ECNN models------------------------------'''

age_model("ECNN", level ="test", model_path = helper.find_path("ECNN"), log_pth=opt.ECNN_log)



