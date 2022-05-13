import argparse

import sys
sys.path.insert(1, '/media/jsh/Data/age_estimation/age_estimation_code/code/')

parser = argparse.ArgumentParser(description='brain age estimation')

"----------------------------- General options -----------------------------"


"----------------------------- Model options -----------------------------"

parser.add_argument('--optimizer', default= "Adam")
parser.add_argument('--lr_ECNN', default=0.00005)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--loss', default="mae")
parser.add_argument('--save_best_only', default=True, help="save best checkpoint")
parser.add_argument('--save_weights_only', default=False, help="save only weights of model or no")
parser.add_argument('--Train_num', default= 15)
parser.add_argument('--Valid_num', default= 2)
parser.add_argument('--Test_num', default= 2)

parser.add_argument('--batch_size', default= 15)
parser.add_argument('--batch_size_2D', default= 1)
parser.add_argument('--batch_size_3D', default= 1)

parser.add_argument('--epochs_3D', default= 1)
parser.add_argument('--epochs_ECNN', default= 4)
parser.add_argument('--epochs_2D', default=1)
parser.add_argument('--model_pth', default='/media/jsh/Data/age_estimation/age_estimation_code/code/model/weight')

"---------------------------------------Data_path--------------------------------------------------------"

parser.add_argument('--Train_GrayPath_root', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/Gray/train")
parser.add_argument('--Train_WhitePath_root', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/White/train")
parser.add_argument('--Train_GrayPath', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/Gray/train_aug")
parser.add_argument('--Train_WhitePath', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/White/train_aug")
parser.add_argument('--Test_GrayPath', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/Gray/test")
parser.add_argument('--Test_whitePath', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/White/test")
parser.add_argument('--Valid_GrayPath', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/Gray/valid")
parser.add_argument('--Valid_WhitePath', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/MRI_data/White/valid")

parser.add_argument('--Label_Path', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/2001Data.xlsx")
parser.add_argument('--meta_data_train', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/metadata/MRI_meta_train.npy")
parser.add_argument('--meta_data_test', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/metadata/MRI_meta_test.npy")
parser.add_argument('--meta_data_valid', default="/media/jsh/Data/age_estimation/age_estimation_code/code/data/metadata/MRI_meta_valid.npy")

"-----------------------------------------------LOG_path-------------------------------------------------"
parser.add_argument('--log', default="./Log/")
parser.add_argument('--Sagittal_Gray_log', default="./Log/Sagittal_Gray/")
parser.add_argument('--Coronal_Gray_log', default="./Log/Coronal_Gray/")
parser.add_argument('--Axial_Gray_log', default="./Log/Axial_Gray/")
parser.add_argument('--Sagittal_White_log', default="./Log/Sagittal_White/")
parser.add_argument('--Coronal_White_log', default="./Log/Coronal_White/")
parser.add_argument('--Axial_White_log', default="./Log/Axial_White/")
parser.add_argument('--White_3DCNN_log', default="./Log/White_3DCNN/")
parser.add_argument('--Gray_3DCNN_log', default="./Log/Gray_3DCNN/")
parser.add_argument('--ECNN_log', default="./Log/ECNN/")

opt = parser.parse_args()















