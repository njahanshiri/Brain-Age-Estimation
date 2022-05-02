import argparse


parser = argparse.ArgumentParser(description='brain age estimation')

"----------------------------- General options -----------------------------"


"----------------------------- Model options -----------------------------"

parser.add_argument('--optimizer', default= "Adam")
parser.add_argument('--lr_ECNN', default=0.00005)
parser.add_argument('--lr', default=0.01)
parser.add_argument('--loss', default="mae")

parser.add_argument('--Train_num', default= 7995)
parser.add_argument('--Valid_num', default= 199)
parser.add_argument('--Test_num', default= 200)

parser.add_argument('--batch_size', default= 15)
parser.add_argument('--batch_size_2D', default= 20)
parser.add_argument('--batch_size_3D', default= 20)

parser.add_argument('--epochs_3D', default= 200)
parser.add_argument('--epochs_ECNN', default= 4)
parser.add_argument('--epochs_2D', default=200)

"---------------------------------------Data_path--------------------------------------------------------"

parser.add_argument('--Train_GrayPath_root', default="/mnt/sdb2/age_estimation/dataset/data_split/Grey/Train")
parser.add_argument('--Train_WhitePath_root', default="/mnt/sdb2/age_estimation/dataset/data_split/White/Train")
parser.add_argument('--Train_GrayPath', default="./data/MRI_data/White/train_aug")
parser.add_argument('--Train_WhitePath', default="./data/MRI_data/Gray/train_aug")
parser.add_argument('--Test_GrayPath', default="/mnt/sdb2/age_estimation/dataset/data_split/Grey/Test")
parser.add_argument('--Test_whitePath', default="/mnt/sdb2/age_estimation/dataset/data_split/White/Test")
parser.add_argument('--Valid_GrayPath', default="/mnt/sdb2/age_estimation/dataset/data_split/Grey/Valid")
parser.add_argument('--Valid_WhitePath', default="/mnt/sdb2/age_estimation/dataset/data_split/White/Valid")

parser.add_argument('--Label_Path', default="./data/2001Data.xlsx")
parser.add_argument('--meta_data_train', default="./data/metadata/MRI_meta_train.npy")
parser.add_argument('--meta_data_test', default="./data/metadata/MRI_meta_test.npy")
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















